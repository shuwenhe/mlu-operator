"""Generate trace-only copies of accepted fixture and RMSNorm source."""
from pathlib import Path
import sys

out = Path(sys.argv[1])
s = Path("operators/rmsnorm/native/rmsnorm.mlu").read_text()
s = s.replace("bang_rmsnorm_", "phase82_")
s = s.replace("Low *rounded, int rows", "Low *rounded, float *squared, float *stats, int rows")
s = s.replace("uint16_t *rounded, int rows", "uint16_t *rounded, float *squared, float *stats, int rows")
s = s.replace("            for (int i = 0; i < count; ++i)\n                squares", """            __memcpy(squared + base + start, work, count * sizeof(float), NRAM2GDRAM);
            for (int i = 0; i < count; ++i)
                squares""")
s = s.replace("        const float inverse = work[0];", """        const float inverse = work[0];
        __nram__ float scalars[4];
        scalars[0] = squares;
        scalars[1] = squares / static_cast<float>(width);
        scalars[2] = variance_eps;
        scalars[3] = inverse;
        __memcpy(stats + row * 4, scalars, sizeof(scalars), NRAM2GDRAM);""")
s = s.replace("reinterpret_cast<Low *>(rounded),\n        rows", "reinterpret_cast<Low *>(rounded), squared, stats,\n        rows")
(out / "probe.mlu").write_text(s)
s = Path("runtime/native/r6_test.cpp").read_text()
anchor = "                const auto ref_attention=reference_mla"
assert anchor in s
s = s.replace(anchor, """                if(std::getenv("LAYER0_TRACE_DIR") && !fp16 && prefill==128) {
                    std::vector<float> rope(actual_rot.size());
                    std::transform(actual_rot.begin(),actual_rot.end(),rope.begin(),decode);
                    trace("attention_rotary",i,r.length,rope);
                    trace("attention_cache",i,r.length,rounded(actual_cache));
                }
"""+anchor)
(out / "trace.cpp").write_text(s)
