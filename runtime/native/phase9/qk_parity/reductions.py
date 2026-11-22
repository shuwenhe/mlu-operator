"""Generate isolated device experiments from the frozen baseline."""
from pathlib import Path
import sys

parent = Path(__file__).resolve().parents[1]
baseline = (parent / "baseline_9_1" / "attention_qk.mlu").read_text()
start = baseline.index("            float dot =")
end = baseline.index("            scores[j] = dot;", start)
strategies = {"serial": baseline[start:end],
              "fma": """            float dot = 0.f;
            for (uint32_t p = 0; p < dim; ++p)
                dot = __builtin_fmaf(query[p], keys[p], dot);
""",
              "tree": """            for (uint32_t stride = 1; stride < dim; stride *= 2)
                for (uint32_t p = 0; p + stride < dim; p += 2 * stride)
                    products[p] += products[p + stride];
            float dot = products[0];
""",
              "chunk16": """            float dot = 0.f;
            for (uint32_t start = 0; start < dim; start += 16) {
                float chunk = 0.f;
                for (uint32_t p = start; p < start + 16; ++p)
                    chunk += products[p];
                dot += chunk;
            }
""",
              "lanes32": """            float dot = 0.f;
            for (uint32_t lane = 0; lane < 32; ++lane) {
                float sum = 0.f;
                for (uint32_t p = lane; p < dim; p += 32)
                    sum += products[p];
                dot += sum;
            }
""",
              "avx2_64": """            for (uint32_t p = 0; p < 64; ++p)
                products[p] = (products[p] + products[p + 64]) + products[p + 128];
            for (uint32_t p = 0; p < 8; ++p)
                products[p] = ((products[p] + products[p + 32]) +
                               (products[p + 16] + products[p + 48])) +
                              ((products[p + 24] + products[p + 56]) +
                               (products[p + 8] + products[p + 40]));
            for (uint32_t p = 0; p < 4; ++p)
                products[p] += products[p + 4];
            float dot = (products[0] + products[2]) + (products[1] + products[3]);
"""}

out = Path(sys.argv[1])
for name, reduction in strategies.items():
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "attention_qk.mlu").write_text(baseline[:start] + reduction + baseline[end:])
    for filename in ("attention_qk.h", "build_qk.sh", "qk_device_test.cpp"):
        (folder / filename).write_bytes((parent / "baseline_9_1" / filename).read_bytes())
    harness = folder / "qk_device_test.cpp"
    source = harness.read_text()
    point = "    check(launch(queue, q.data, k.data, pos.data, base, scale, raw.data,"
    source = source.replace(point, """    cnrtNotifier_t start, end;
    check(cnrtNotifierCreate(&start));
    check(cnrtNotifierCreate(&end));
    check(cnrtPlaceNotifier(start, queue));
""" + point)
    point = "    return write_output(raw.download())"
    source = source.replace(point, """    check(cnrtPlaceNotifier(end, queue));
    check(cnrtQueueSync(queue));
    float microseconds = 0.f;
    check(cnrtNotifierDuration(start, end, &microseconds));
    check(cnrtNotifierDestroy(start));
    check(cnrtNotifierDestroy(end));
    std::fprintf(stderr, "latency_us=%.3f\\n", microseconds);
""" + point)
    source = source.replace("scaled.data, masked.data, heads, seq, length, length));\n    check(cnrtQueueSync(queue));",
                            "scaled.data, masked.data, heads, seq, length, length));")
    harness.write_text(source)

# Instrument the selected order without changing the production launcher ABI.
production = out / "production"
diagnostic = out / "diagnostic"
for folder in (production, diagnostic):
    folder.mkdir(exist_ok=True)
    for filename in ("attention_qk.h", "build_qk.sh", "qk_device_test.cpp"):
        (folder / filename).write_bytes((parent / filename).read_bytes())
    source = (parent / "attention_qk.mlu").read_text()
    if folder == diagnostic:
        raw_copy = "        __memcpy(raw + output, scores, count * sizeof(float), NRAM2GDRAM);\n"
        source = source.replace(raw_copy, "", 1)
        pos = source.index("        for (uint32_t j = 0; j < tile; ++j) {\n            scores[j] *= scale;")
        source = source[:pos] + ("        __memcpy(masked + output, scores, count * sizeof(float), NRAM2GDRAM);\n"
                                 + source[pos:])
        pos = source.index("        NARROW(scores_low, scores, tile);", pos)
        source = source[:pos] + raw_copy + source[pos:]
        pos = source.index("        WIDEN(scores, scores_low, tile);", pos)
        source = source[:pos] + "    }\n}\n" + source[source.index('\nextern "C"', pos):]
    (folder / "attention_qk.mlu").write_text(source)
