"""Rank-local unquantized inference Linear on the Vendor TP group."""
import torch
import torch.distributed as dist
from torch.nn import functional as F

from integration.vllm_native_mlu.phase12_4.projection import PackedProjection

from .shards import column_spans, row_span


NATIVE_MAX = 10944


def validate_tp(group, device):
    if (
        device.type != "mlu" or device.index != torch.mlu.current_device()
        or group.world_size not in (1, 2, 4, 8)
        or dist.get_backend(group.device_group) != "cncl"
        or (group.world_size > 1 and type(group.device_communicator).__name__ != "MLUCommunicator")
    ):
        raise ValueError("Expected a Vendor CNCL TP group on the current MLU")


class LocalGemm:
    def __init__(self, weight, library):
        if weight.ndim != 2 or not all(weight.shape) or weight.device.type != "mlu":
            raise ValueError("Expected a nonempty MLU weight [N, K]")
        self.n, self.k = weight.shape
        self.device, self.dtype = weight.device, weight.dtype
        self.packed = None
        self.packed_tiles = []
        if self.dtype == torch.bfloat16:
            pad_k = -self.k % 64
            if self.k + pad_k > NATIVE_MAX:
                raise ValueError("Padded BF16 input width exceeds native limit")
            if self.n <= NATIVE_MAX:
                self.packed = PackedProjection(library, F.pad(weight, (0, pad_k, 0, -self.n % 64)))
            else:
                try:
                    for tile in weight.split(NATIVE_MAX, dim=0):
                        width = tile.shape[0]
                        packed = PackedProjection(library, F.pad(tile, (0, pad_k, 0, -width % 64)))
                        self.packed_tiles.append((width, packed))
                except Exception:
                    self.close()
                    raise
        elif self.dtype == torch.float32:
            self.weight = weight.contiguous()
        else:
            raise ValueError("Expected BF16 or FP32 weights")

    def __call__(self, x):
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.k:
            raise ValueError(f"Expected input [M, {self.k}], got {tuple(x.shape)}")
        if x.device != self.device or x.dtype != self.dtype:
            raise ValueError("Input must match shard device and dtype")
        if self.dtype == torch.float32:
            return torch.mm(x, self.weight.T)
        padded = F.pad(x, (0, -self.k % 64)).contiguous()
        if self.packed is not None:
            return self.project_rows(padded, self.packed, self.n)
        return torch.cat(
            [self.project_rows(padded, packed, width) for width, packed in self.packed_tiles],
            dim=1,
        )

    @staticmethod
    def project_rows(x, packed, width):
        chunks = [packed(part)[:, :width] for part in x.split(128)]
        return chunks[0].contiguous() if len(chunks) == 1 else torch.cat(chunks, dim=0)

    def close(self):
        if self.packed is not None:
            self.packed.close()
        for _, packed in self.packed_tiles:
            packed.close()


def check_bias(bias, shape, weight):
    if bias is not None and (
        bias.shape != shape or bias.device != weight.device or bias.dtype != weight.dtype
    ):
        raise ValueError("Bias must match the expected output shape, device and dtype")


class ColumnParallelLinear:
    def __init__(self, weight, sections, group, library, bias=None):
        validate_tp(group, weight.device)
        self.group = group
        self.sections = tuple(sections)
        spans = column_spans(self.sections, group.rank_in_group, group.world_size)
        self.local_widths = tuple(width for _, width in spans)
        if weight.ndim != 2 or weight.shape[0] != sum(self.local_widths):
            raise ValueError("Weight does not have the expected local output width")
        check_bias(bias, (weight.shape[0],), weight)
        self.bias = bias
        self.gemm = LocalGemm(weight, library)

    def __call__(self, x):
        output = self.gemm(x)
        return output if self.bias is None else output + self.bias

    def gather(self, output):
        if output.ndim != 2 or output.shape[1] != sum(self.local_widths):
            raise ValueError("Output does not have the expected local width")
        if output.device != self.gemm.device or output.dtype != torch.float32:
            raise ValueError("Expected FP32 output on the local MLU")
        parts = output.split(self.local_widths, dim=1)
        gathered = [self.group.all_gather(part.contiguous(), dim=1) for part in parts]
        return gathered[0] if len(gathered) == 1 else torch.cat(gathered, dim=1)

    def close(self):
        self.gemm.close()


class RowParallelLinear:
    def __init__(self, weight, input_width, group, library, bias=None):
        validate_tp(group, weight.device)
        self.group = group
        self.input_width = input_width
        self.offset, self.width = row_span(input_width, group.rank_in_group, group.world_size)
        if weight.ndim != 2 or weight.shape[1] != self.width:
            raise ValueError("Weight does not have the expected local input width")
        check_bias(bias, (weight.shape[0],), weight)
        self.bias = bias
        self.gemm = LocalGemm(weight, library)

    def local_input(self, x, input_is_parallel):
        width = self.width if input_is_parallel else self.input_width
        if x.ndim != 2 or x.shape[1] != width:
            raise ValueError(f"Expected input width {width}")
        return x if input_is_parallel else x.narrow(1, self.offset, self.width)

    def __call__(self, x, *, input_is_parallel=False, residual=None):
        if residual is not None and (
            x.ndim != 2 or residual.shape != (x.shape[0], self.gemm.n)
            or residual.device != self.gemm.device
            or residual.dtype not in (self.gemm.dtype, torch.float32)
        ):
            raise ValueError("Residual must match batch size, output width, device and dtype")
        local = self.gemm(self.local_input(x, input_is_parallel))
        output = self.group.all_reduce(local)
        if self.bias is not None:
            output = output + self.bias
        if residual is not None:
            output = output + residual
        return output

    def close(self):
        self.gemm.close()
