#include "adapter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace deepseek;
using namespace native_mlu;

namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

template<class T> std::vector<T> download(View view) {
    std::vector<T> data(size_t(view.rows) * view.cols);
    check(cnrtMemcpy(data.data(), view.data, data.size() * sizeof(T), cnrtMemcpyDevToHost), "download");
    return data;
}

template<class T> void upload(View view, const std::vector<T> &data) {
    require(data.size() == size_t(view.rows) * view.cols, "upload shape");
    check(cnrtMemcpy(view.data, const_cast<T *>(data.data()), data.size() * sizeof(T), cnrtMemcpyHostToDev), "upload");
}

bool compare(const char *stage, const char *backend, const std::vector<float> &expected,
             const std::vector<float> &actual) {
    require(expected.size() == actual.size(), "comparison shape");
    size_t bad = 0, nonfinite = 0;
    double maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double ref = expected[i], got = actual[i];
        const double error = std::abs(ref - got);
        nonfinite += !std::isfinite(got) || !std::isfinite(ref);
        bad += !std::isfinite(got) || !std::isfinite(ref) || error > 1e-5 + 1e-5 * std::abs(ref);
        maximum = std::max(maximum, error);
    }
    std::printf("backend=%s stage=%s elements=%zu max_abs=%.9g mismatches=%zu nonfinite=%zu\n",
                backend, stage, actual.size(), maximum, bad, nonfinite);
    return bad == 0;
}

std::vector<float> cpu_projection(const std::string &dir, Tensor tensor,
                                  const std::vector<uint16_t> &input, Type dtype) {
    const auto &c = contract(tensor);
    const auto raw = load_words(dir + "/" + c.alias + ".bin", size_t(c.n) * c.k, dtype);
    const int rows = input.size() / c.k;
    std::vector<float> ref(size_t(rows) * c.n);
    for (int row = 0; row < rows; ++row) {
        for (int n = 0; n < c.n; ++n) {
            double sum = 0;
            for (int k = 0; k < c.k; ++k)
                sum += double(decode(input[size_t(row) * c.k + k], dtype)) *
                       decode(raw[size_t(n) * c.k + k], dtype);
            ref[size_t(row) * c.n + n] = static_cast<float>(sum);
        }
    }
    return ref;
}

std::vector<float> widen(const std::vector<uint16_t> &words, Type type) {
    std::vector<float> result;
    result.reserve(words.size());
    for (auto word : words) result.push_back(decode(word, type));
    return result;
}

std::vector<uint16_t> expected_norm(const std::vector<uint16_t> &words,
                                   const std::vector<uint16_t> &gamma, Type type) {
    float squares = 0;
    for (float x : widen(words, type)) squares += x * x;
    const float scale = 1.f / std::sqrt(squares / words.size() + 1e-6f);
    std::vector<uint16_t> output(words.size());
    for (size_t j = 0; j < words.size(); ++j) {
        const float normalized = decode(encode(decode(words[j], type) * scale, type), type);
        output[j] = encode(normalized * decode(gamma[j], type), type);
    }
    return output;
}
}

int main(int argc, char **argv) try {
    if (argc != 6 || std::string(argv[1]) != "--authorized") return 2;
    const std::string dir = argv[3], mode = argv[4], dtype = argv[5];
    if ((mode != "q" && mode != "kv" && mode != "projections") || (dtype != "bf16" && dtype != "fp16")) return 2;
    const Type type = dtype == "bf16" ? Type::bf16 : Type::fp16;
    Worker worker(argv[2]);
    Registry registry(worker);
    Memory memory(worker);
    Workspace workspace(worker, memory);
    Dispatch dispatch(worker, memory, workspace);
    Adapter native(worker, registry, memory, dispatch, dir, Backend::Native, type);
    Adapter cnnl(worker, registry, memory, dispatch, dir, Backend::CNNL, type);
    const size_t slot = workspace.add(worker.add_queue(), 2, 4);
    std::vector<void *> buffers;
    auto make = [&](int rows, int cols, Type format) {
        void *ptr = memory.allocate(size_t(rows) * cols *
                                    (format == Type::fp32 || format == Type::u32 ? 4 : 2), Kind::workspace);
        buffers.push_back(ptr);
        return View{ptr, rows, cols, cols, format};
    };
    std::printf("PCI=%s ordinal=%d dtype=%s mode=%s\n", worker.pci().c_str(), worker.ordinal(),
                dtype.c_str(), mode.c_str());

    const auto embedding = load_words(dir + "/token.bin", 2048, type);
    const auto input = make(1, 2048, type);
    upload(input, embedding);
    bool ok = true;
    if (mode != "kv") {
        const auto cases = mode == "q" ? std::vector<Tensor>{Tensor::Q}
                                       : std::vector<Tensor>{Tensor::Output, Tensor::Gate, Tensor::Up};
        for (Tensor tensor : cases) {
            const auto &c = contract(tensor);
            const auto result = make(1, c.n, Type::fp32);
            const auto expected = cpu_projection(dir, tensor, embedding, type);
            for (Adapter *adapter : {&native, &cnnl}) {
                adapter->load(tensor);
                workspace.start(slot);
                auto padded = input;
                ++padded.stride;
                bool rejected = false;
                try {
                    adapter->project(slot, tensor, padded, result);
                } catch (const std::invalid_argument &) {
                    rejected = true;
                }
                require(rejected, "invalid stride accepted");
                adapter->project(slot, tensor, input, result);
                workspace.finish(slot);
                ok &= compare(c.alias, adapter->backend() == Backend::Native ? "native" : "cnnl",
                              expected, download<float>(result));
            }
        }
    } else {
        for (Tensor tensor : {Tensor::InputNorm, Tensor::KvA, Tensor::KvB, Tensor::KvNorm})
            native.load(tensor);
        const auto normalized_input = make(1, 2048, type);
        const auto wide = make(1, 576, Type::fp32);
        const auto latent = make(1, 512, type);
        const auto normalized = make(1, 512, type);
        const auto output = make(1, 4096, Type::fp32);
        workspace.start(slot);
        native.norm(slot, Tensor::InputNorm, input, normalized_input);
        native.kv(slot, normalized_input, wide, latent, normalized, output);
        bool rejected = false;
        try {
            cnnl.kv(slot, normalized_input, wide, latent, normalized, output);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        require(rejected, "CNNL KV chain unexpectedly available");
        workspace.finish(slot);

        const auto input_gamma = load_words(dir + "/input_gamma.bin", 2048, type);
        const auto kv_gamma = load_words(dir + "/kv_gamma.bin", 512, type);
        const auto input_words = download<uint16_t>(normalized_input);
        const auto latent_words = download<uint16_t>(latent);
        const auto norm_words = download<uint16_t>(normalized);
        const auto expected_a = cpu_projection(dir, Tensor::KvA, input_words, type);
        const auto actual_a = download<float>(wide);
        std::vector<uint16_t> expected_latent(512);
        for (size_t i = 0; i < expected_latent.size(); ++i)
            expected_latent[i] = encode(actual_a[i], type);
        const auto expected_b = cpu_projection(dir, Tensor::KvB, norm_words, type);
        const auto actual_b = download<float>(output);
        ok &= compare("input_norm", "native", widen(expected_norm(embedding, input_gamma, type), type),
                      widen(input_words, type));
        ok &= compare("kv_a", "native", expected_a, actual_a);
        require(latent_words == expected_latent, "latent cast or stride differs");
        ok &= compare("kv_norm", "native", widen(expected_norm(latent_words, kv_gamma, type), type),
                      widen(norm_words, type));
        ok &= compare("kv_b", "native", expected_b, actual_b);
        const auto expected_input = expected_norm(embedding, input_gamma, type);
        const auto ref_wide = cpu_projection(dir, Tensor::KvA, expected_input, type);
        std::vector<uint16_t> ref_latent(512);
        for (size_t i = 0; i < ref_latent.size(); ++i) ref_latent[i] = encode(ref_wide[i], type);
        ok &= compare("kv_chain", "native", cpu_projection(dir, Tensor::KvB,
                      expected_norm(ref_latent, kv_gamma, type), type), actual_b);
        for (Tensor tensor : {Tensor::KvA, Tensor::KvB}) cnnl.load(tensor);
        workspace.start(slot);
        cnnl.project(slot, Tensor::KvA, normalized_input, wide);
        cnnl.project(slot, Tensor::KvB, normalized, output);
        workspace.finish(slot);
        ok &= compare("kv_a_matched_input", "cnnl", expected_a, download<float>(wide));
        ok &= compare("kv_b_matched_input", "cnnl", expected_b, download<float>(output));

        constexpr int sequence = 2, width = 64, heads = 17;
        const auto rotary = make(heads * sequence, width, type);
        const auto cosine = make(8, width, type);
        const auto sine = make(8, width, type);
        const auto positions = make(1, sequence, Type::u32);
        const auto rotated = make(heads * sequence, width, type);
        std::vector<uint16_t> rope_input(heads * sequence * width);
        for (size_t i = 0; i < rope_input.size(); ++i)
            rope_input[i] = embedding[i % embedding.size()];
        const std::vector<uint32_t> indices{7, 1};
        std::vector<uint16_t> cos_words(8 * width), sin_words(8 * width);
        const auto correction = [](double rotations) {
            return width * std::log(4096 / (rotations * 2 * std::acos(-1.0))) /
                   (2 * std::log(10000.0));
        };
        const float lo = std::max(0., std::floor(correction(32)));
        const float hi = std::min(63., std::ceil(correction(1)));
        for (int channel = 0; channel < width / 2; ++channel) {
            const float divisor = std::pow(10000.f, float(2 * channel) / width);
            const float ramp = std::clamp((channel - lo) / (hi - lo), 0.f, 1.f);
            const float freq = (1 - ramp) / divisor + ramp / (40.f * divisor);
            for (int pos = 0; pos < 8; ++pos) {
                const size_t i = pos * width + channel;
                cos_words[i] = cos_words[i + 32] = encode(std::cos(pos * freq), type);
                sin_words[i] = sin_words[i + 32] = encode(std::sin(pos * freq), type);
            }
        }
        upload(rotary, rope_input);
        upload(cosine, cos_words);
        upload(sine, sin_words);
        upload(positions, indices);
        workspace.start(slot);
        native.rope(slot, rotary, cosine, sine, positions, rotated, sequence);
        workspace.finish(slot);
        const auto result = download<uint16_t>(rotated);
        for (int row = 0; row < heads * sequence; ++row) {
            const auto table = indices[row % sequence] * width;
            for (int channel = 0; channel < width; ++channel) {
                const bool second = channel >= width / 2;
                const size_t pair = row * width + 2 * (channel % (width / 2));
                const float direct = decode(rope_input[pair + second], type);
                const float crossed = (second ? 1.f : -1.f) * decode(rope_input[pair + !second], type);
                const float left = decode(encode(direct * decode(cos_words[table + channel], type), type), type);
                const float right = decode(encode(crossed * decode(sin_words[table + channel], type), type), type);
                require(result[row * width + channel] == encode(left + right, type), "YaRN output bits");
            }
        }
        std::puts("backend=native stage=yarn_rope positions=7/1 elements=2176 bit_exact=1");
    }
    workspace.close();
    native.close();
    cnnl.close();
    registry.close();
    for (auto &buffer : buffers) memory.release(buffer);
    require(memory.usage().live == 0 && registry.bytes() == 0, "device resource leak");
    std::puts("cleanup live_bytes=0 packed_weight_bytes=0");
    memory.close();
    worker.close();
    return ok ? 0 : 1;
} catch (const std::exception &error) {
    std::fprintf(stderr, "adapter error: %s\n", error.what());
    return 1;
}
