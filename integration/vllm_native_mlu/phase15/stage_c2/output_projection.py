"""Output-axis MLA projection with a complete K accumulation per output stripe."""
from ...phase13_2.linear import ColumnParallelLinear
from ...phase13_2.shards import row_span


WIDTH = 2048


class OutputAxisProjection:
    def __init__(self, group, weight, library):
        self.group = group
        self.linear = ColumnParallelLinear(weight, (WIDTH,), group, library)

    def __call__(self, av, residual):
        local = av.transpose(0, 1).contiguous().flatten(1)
        complete = self.group.all_gather(local, dim=1)
        partial = self.linear(complete.contiguous())
        wide = self.linear.gather(partial)
        rounded = wide.to(residual.dtype)
        return rounded, rounded + residual, wide

    def close(self):
        self.linear.close()


def output_stripes(weights, model, stripes):
    """Capture the disjoint output-axis checkpoint slices during the normal load."""
    offset, size = row_span(WIDTH, model.rank, model.group.world_size)
    for name, tensor in weights:
        fields = name.split(".")
        if len(fields) == 6 and fields[:2] == ["model", "layers"] and fields[3:] == ["self_attn", "o_proj", "weight"]:
            if tensor.shape != (WIDTH, WIDTH):
                raise ValueError(f"Invalid output projection shape: {name}")
            stripes[int(fields[2])] = tensor.narrow(0, offset, size).clone()
        yield name, tensor


def install(model, stripes):
    if set(stripes) != set(range(len(model.mla))):
        raise ValueError("Missing MLA output projection stripes")
    library = model.libraries / "projection/libprojection.so"
    for number, layer in enumerate(model.mla):
        output = OutputAxisProjection(model.group, stripes.pop(number).to(model.device), library)
        layer.parts.append(output)
        layer.o.close()
        layer.parts.remove(layer.o)
        layer.o = output
        layer.output = output
