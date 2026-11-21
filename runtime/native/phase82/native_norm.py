"""Generate an independent BANG C AVX2-cascade RMSNorm experiment."""
from pathlib import Path
import sys

directory=Path(sys.argv[1])
text=Path("operators/rmsnorm/native/rmsnorm.mlu").read_text().replace("bang_rmsnorm_", "phase82_native_")
text=text.replace("    __nram__ float values[tile], aux[tile], work[tile];", """    __nram__ float values[tile], aux[tile], work[tile], ones[tile];""")
text=text.replace("        float squares = 0.0f;", """        float chunk[32]={}, total[32]={};""")
text=text.replace("""            for (int i = 0; i < count; ++i)
                squares += work[i];""","""            for (int i = 0; i < count; ++i)
                chunk[i % 32] += work[i];
            if ((start + tile) % 512 == 0) {
                for (int i=0;i<32;++i) {
                    total[i]+=chunk[i];
                    chunk[i]=0.f;
                }
            }""")
text=text.replace("        const float variance_eps", """        float squares=0.f;
        for(int i=0;i<8;++i)
            squares+=(total[i]+total[i+8]+total[i+16]+total[i+24]);
        const float variance_eps""")
text=text.replace("        __bang_rsqrt(work, aux, tile);", """        __bang_sqrt(work, aux, tile);
        for(int i=0;i<tile;++i)ones[i]=1.f;
        __bang_div(aux,ones,work,tile);""").replace("        const float inverse = work[0];","        const float inverse = aux[0];")
text=text.replace("rows <= 0 || width <= 0", "rows <= 0 || (width != 512 && width != 2048)")
if "--exact-rsqrt" in sys.argv[2:]:
    text = '#include "rsqrt_round.h"\n' + text
    begin = text.index("        __bang_sqrt(")
    end = text.index("        for (int i = 0; i < tile; ++i)\n            aux[i] = inverse;", begin)
    text = text[:begin] + "        const float inverse = phase82_rsqrt_round(variance_eps);\n" + text[end:]
    text = text.replace(", ones[tile]", "")
(directory/"native_norm.mlu").write_text(text)
