#!/usr/bin/env bash
set -euxo pipefail
dir=${1:?experimental output}
old=${2:?frozen Phase 8 build}
sdk=/usr/local/neuware
torch=$(find /torch/venv3/pytorch_infer -type d -path '*/site-packages/torch' -print -quit)
python3 - "$dir" <<'PY'
from pathlib import Path
import sys
out=Path(sys.argv[1])
source=Path("operators/rmsnorm/native/rmsnorm.mlu").read_text()
source=source.replace("bang_rmsnorm_", "phase82_finish_")
source=source.replace("Low *rounded, int rows, int width, float eps", "Low *rounded, const float *inverses, int rows, int width")
source=source.replace("uint16_t *rounded, int rows, int width, float eps", "uint16_t *rounded, const float *inverses, int rows, int width")
begin=source.index("        float squares")
end=source.index("        for (int i = 0; i < tile; ++i)\n            aux[i] = inverse;",begin)
source=source[:begin]+"        const float inverse = inverses[row];\n"+source[end:]
source=source.replace(" ||\n        !(eps > 0.0f && eps <= __FLT_MAX__)", " || !inverses")
source=source.replace("rows, width, eps);","inverses, rows, width);")
(out/"finish.mlu").write_text(source)
fixture=Path("runtime/native/r6_test.cpp").read_text()
anchor="using namespace native_mlu;"
assert anchor in fixture
fixture=fixture.replace(anchor,anchor+"\nstd::vector<float> checkpoint_attention(const std::vector<float>&, const std::vector<float>&, const std::vector<uint16_t>&, int, int, bool);")
anchor="const auto out_attention=reference_mla(actual_q,actual_cache,actual_rot,r.length,batch);"
assert anchor in fixture
fixture=fixture.replace(anchor,"const auto out_attention=checkpoint_attention(actual_q,actual_cache,actual_rot,r.length,batch, !fp16);")
(out/"experiment.cpp").write_text(fixture)
PY
norm_def=()
if [[ ${3:-} == device_only || ${3:-} == exact_rsqrt ]]; then
  opts=()
  if [[ ${3:-} == exact_rsqrt ]]; then opts=(--exact-rsqrt); fi
  python3 runtime/native/phase82/native_norm.py "$dir" "${opts[@]}"
  norm_def=(-DPHASE82_DEVICE_ONLY)
  norm_source=native_norm
else
  norm_source=finish
fi
for dtype in fp16 bf16; do
 defs=()
 if [[ $dtype == bf16 ]]; then defs=(-DRMSNORM_BF16); fi
 "$sdk/bin/cncc" --bang-mlu-arch=mtp_592 -std=c++11 -O2 -ffp-contract=off -Ioperators/rmsnorm/native -Iruntime/native/phase82 "${defs[@]}" -c "$dir/$norm_source.mlu" -o "$dir/finish-$dtype.o"
done
g++ -no-pie -O2 -std=c++17 -ffp-contract=off "${norm_def[@]}" -I"$sdk/include" -Ioperators/gemm/native/prepacked -Iruntime/native \
 -Ioperators/rmsnorm/native -Icommon/reference -isystem "$torch/include" \
 runtime/native/{runtime,workspace,dispatch}.cpp "$dir/experiment.cpp" runtime/native/phase8/candidate_wrap.cpp \
 runtime/native/phase82/checkpoint_arithmetic.cpp /results/prepacked-final-build/{fp16,bf16,packed}.o \
 /results/runtime/{rope-fp16,rope-bf16,kv-cast}.o "$old"/{fp16,bf16}/gemm.o "$dir"/finish-{fp16,bf16}.o \
 -Wl,--wrap=_ZN3mlu14gemm_prepackedEPKtRKNS_12PackedWeightEPfiP10CNqueue_st \
 -Wl,--wrap=bang_rmsnorm_fp16 -Wl,--wrap=bang_rmsnorm_bf16 \
 -L"$sdk/lib64" -Wl,-rpath-link,"$sdk/lib64" -lcnrt -lcnnl -L"$torch/lib" -Wl,-rpath,"$torch/lib" \
 -ltorch_cpu -lc10 -o "$dir/layer0_experiment"
