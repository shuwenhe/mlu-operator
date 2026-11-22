"""Eight-card checks of the vendor cache, expert dispatch and CNCL path."""
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_mlu  # noqa: F401
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment, destroy_model_parallel, get_tp_group,
    init_distributed_environment, initialize_model_parallel,
)
from vllm_mlu.v1.kv_cache_interface import MLUFullAttentionSpec

from ...phase13_2.linear import validate_tp
from ...phase13_3_2.probe import LIB, ROOT, bf16
from ...phase13_4.full_model import ReplicatedDecoder
from ...phase13_4.mla import Cache, NativeCalls
from ...phase14_1.model import WorkerCache
from .cache import TP8Backend
from .contracts import expert_owner, layout
from .model import ExpertParallelDecoder


def check(group, device, name, ok):
    failures = group.all_reduce(torch.tensor([not ok], dtype=torch.int32, device=device)).item()
    print(json.dumps(dict(gate=name, rank=group.rank_in_group, failures=failures,
                          status="FAIL" if failures else "PASS")), flush=True)
    if failures:
        raise AssertionError(f"{name}: {failures} failing ranks")


def validate(group, device):
    rank = group.rank_in_group
    validate_tp(group, device)
    value = torch.tensor([rank + 1], dtype=torch.float32, device=device)
    check(group, device, "cncl_sum_gather", group.all_gather(value).tolist() == list(range(1, 9))
          and group.all_reduce(value).item() == 36)

    cos, sin = (bf16(ROOT / "rope_yarn" / f"{name}.bin", (131, 64)).to(device) for name in ("cos", "sin"))
    scale = float((ROOT / "rope_yarn/scale.txt").read_text())
    ops = NativeCalls(group, device, LIB, Path(os.environ["NEURX_PAGED_TWO"]), cos, sin, scale)
    slots, split, words = layout(3, 16)
    spec = MLUFullAttentionSpec(block_size=16, num_kv_heads=2, head_size=192,
                                head_size_v=128, dtype=torch.bfloat16)
    raw = torch.empty(words * 2, dtype=torch.int8, device=device)
    views = TP8Backend.reshape_kv_cache(raw, spec, 3, 16, "auto")
    cache = WorkerCache(ops, views)
    check(group, device, "zero_copy_cache_views",
          cache.slots == slots and views[0].data_ptr() == raw.data_ptr()
          and views[1].data_ptr() == raw.data_ptr() + 2 * split)
    ref = Cache(ops, 3)
    key, val = (torch.randn((2, 17, width), device=device, dtype=torch.bfloat16) for width in (192, 128))
    table = [2, 0]
    mapping = list(range(32, 48)) + [0]
    for page in (cache, ref):
        page.write(key[:, :16].contiguous(), val[:, :16].contiguous(), mapping[:16])
        page.write(key[:, 16:].contiguous(), val[:, 16:].contiguous(), mapping[16:])
    actual = cache.read(table, 17)
    check(group, device, "cache_cross_page", torch.equal(actual[0], key) and torch.equal(actual[1], val))

    decoder = ExpertParallelDecoder(ops)
    reference = ReplicatedDecoder(ops)
    try:
        prefix = "model.layers.1.mlp."
        hidden = bf16(ROOT / "native-dump-yarn-scale/request2/0-layer0.bin", (-1, 2048)).to(device)
        name = "model.layers.1.post_attention_layernorm.weight"
        hidden = decoder.norm.forward_native(hidden, decoder._resident(name), 1e-6)
        actual = decoder._moe(prefix, hidden)
        expected = reference._moe(prefix, hidden)
        check(group, device, "expert_dispatch_combine", torch.equal(actual, expected))
        check(group, device, "expert_pack_ownership", all(
            expert_owner(int(name.split(".")[5])) == rank
            for name, _, _ in decoder.projections if ".experts." in name
        ))
    finally:
        decoder.close()
        reference.close()


if __name__ == "__main__":
    rank = int(os.environ["LOCAL_RANK"])
    torch.mlu.set_device(rank)
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=8, disable_custom_all_reduce=True))
    with torch.inference_mode(), set_current_vllm_config(config):
        init_distributed_environment(world_size=8, rank=rank, local_rank=rank,
                                     distributed_init_method="env://", backend="cncl")
        try:
            initialize_model_parallel(tensor_model_parallel_size=8, backend="cncl")
            validate(get_tp_group(), torch.device("mlu", rank))
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
