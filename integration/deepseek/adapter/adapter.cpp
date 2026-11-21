#include "adapter.h"

#include <array>
#include <cstring>
#include <fstream>

namespace deepseek {
using namespace native_mlu;
namespace {
constexpr std::array<Contract, 9> contracts{{
    {"q", "model.layers.0.self_attn.q_proj.weight", 2048, 3072},
    {"kv_a", "model.layers.0.self_attn.kv_a_proj_with_mqa.weight", 2048, 576},
    {"kv_b", "model.layers.0.self_attn.kv_b_proj.weight", 512, 4096},
    {"o", "model.layers.0.self_attn.o_proj.weight", 2048, 2048},
    {"gate", "model.layers.0.mlp.gate_proj.weight", 2048, 10944},
    {"up", "model.layers.0.mlp.up_proj.weight", 2048, 10944},
    {"input_gamma", "model.layers.0.input_layernorm.weight", 2048, 1},
    {"kv_gamma", "model.layers.0.self_attn.kv_a_layernorm.weight", 512, 1},
    {"post_gamma", "model.layers.0.post_attention_layernorm.weight", 2048, 1},
}};

void require(bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(message);
}

bool low(Type type) { return type == Type::bf16 || type == Type::fp16; }

void validate(View view, int columns, Type type) {
    require(view.rows > 0 && view.cols == columns && view.stride == columns && view.type == type,
            "DeepSeek tensor shape, stride or dtype");
}
}

const Contract &contract(Tensor tensor) { return contracts.at(static_cast<size_t>(tensor)); }

float decode(uint16_t word, Type type) {
    if (type == Type::fp16) {
        _Float16 value;
        std::memcpy(&value, &word, sizeof(word));
        return value;
    }
    require(type == Type::bf16, "unsupported activation dtype");
    uint32_t bits = uint32_t(word) << 16;
    float value;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

uint16_t encode(float value, Type type) {
    if (type == Type::fp16) {
        const _Float16 half = static_cast<_Float16>(value);
        uint16_t word;
        std::memcpy(&word, &half, sizeof(word));
        return word;
    }
    require(type == Type::bf16, "unsupported activation dtype");
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

std::vector<uint16_t> load_words(const std::string &path, size_t count, Type type) {
    require(low(type), "unsupported activation dtype");
    std::vector<uint16_t> words(count);
    std::ifstream stream(path, std::ios::binary);
    if (!stream.read(reinterpret_cast<char *>(words.data()), count * sizeof(uint16_t)) ||
        stream.peek() != EOF) throw std::runtime_error("invalid checkpoint export: " + path);
    if (type == Type::fp16) {
        for (auto &word : words) word = encode(decode(word, Type::bf16), type);
    }
    return words;
}

Adapter::Adapter(Worker &worker, Registry &registry, Memory &memory, Dispatch &dispatch,
                 const std::string &dir, Backend backend, Type type)
    : worker_(worker), registry_(registry), memory_(memory), dispatch_(dispatch),
      dir_(dir), backend_(backend), type_(type) {
    require(low(type) && (backend == Backend::Native || backend == Backend::CNNL),
            "unsupported DeepSeek backend or dtype");
    std::ifstream manifest(dir_ + "/manifest");
    require(bool(manifest >> revision_) && revision_.size() == 64 &&
            revision_.find_first_not_of("0123456789abcdef") == std::string::npos,
            "missing checkpoint revision");
    std::string alias, tensor, shard;
    uint64_t offset;
    for (size_t i = 0; i < contracts.size(); ++i) {
        const auto &c = contracts[i];
        require(bool(manifest >> alias >> tensor >> shard >> offset) &&
                alias == c.alias && tensor == c.checkpoint_name && offset > 0,
                "invalid checkpoint weight identity");
        weights_.emplace(static_cast<Tensor>(i), Weight{WeightKey{
            revision_, tensor, shard, "kn-row-major", worker_.pci(), offset,
            type_ == Type::bf16 ? mlu::WeightType::bf16 : mlu::WeightType::fp16, c.k, c.n}});
    }
    require(!(manifest >> alias), "unexpected checkpoint weight");
}

Adapter::~Adapter() {
    try { close(); } catch (...) {
        // Memory still owns any pinned blocks. Workspace drains them before Memory teardown.
    }
}

void Adapter::load(Tensor tensor) {
    require(!closed_, "closed DeepSeek adapter");
    auto &entry = weights_.at(tensor);
    const auto &c = contract(tensor);
    const bool gamma = c.n == 1;
    const bool packed = !gamma && backend_ == Backend::Native;
    if (entry.view.data || (packed && registry_.acquire(entry.key))) return;
    worker_.select();
    auto raw = load_words(dir_ + "/" + c.alias + ".bin", size_t(c.k) * c.n, type_);
    if (!gamma) {
        std::vector<uint16_t> logical(raw.size());
        for (int n = 0; n < c.n; ++n)
            for (int k = 0; k < c.k; ++k)
                logical[size_t(k) * c.n + n] = raw[size_t(n) * c.k + k];
        raw.swap(logical);
    }
    if (packed) {
        registry_.host(entry.key, raw.data());
        return;
    }
    const bool cnnl_gamma = gamma && backend_ == Backend::CNNL;
    std::vector<float> wide;
    if (cnnl_gamma) {
        wide.reserve(raw.size());
        for (auto word : raw) wide.push_back(decode(word, type_));
    }
    const size_t bytes = raw.size() * (cnnl_gamma ? sizeof(float) : sizeof(uint16_t));
    void *buffer = memory_.allocate(bytes, Kind::weight);
    try {
        void *source = cnnl_gamma ? static_cast<void *>(wide.data()) : raw.data();
        check(cnrtMemcpy(buffer, source, bytes, cnrtMemcpyHostToDev), "upload checkpoint weight");
    } catch (...) {
        memory_.release(buffer);
        throw;
    }
    const int columns = gamma ? c.k : c.n;
    entry.view = {buffer, gamma ? 1 : c.k, columns, columns, cnnl_gamma ? Type::fp32 : type_};
}

void Adapter::project(size_t slot, Tensor tensor, View input, View output) {
    require(!closed_, "closed DeepSeek adapter");
    const auto &entry = weights_.at(tensor);
    const auto &c = contract(tensor);
    require(c.n > 1 && input.rows == output.rows, "invalid DeepSeek projection");
    validate(input, c.k, type_);
    validate(output, c.n, Type::fp32);
    auto lease = backend_ == Backend::Native ? registry_.acquire(entry.key) : Registry::Lease{};
    require(lease || entry.view.data, "unloaded DeepSeek projection");
    dispatch_.gemm(slot, backend_, input, std::move(lease), entry.view, output);
}

void Adapter::norm(size_t slot, Tensor tensor, View input, View output) {
    require(!closed_, "closed DeepSeek adapter");
    const auto &c = contract(tensor);
    const auto gamma = weights_.at(tensor).view;
    require(c.n == 1 && gamma.data && input.rows == output.rows, "unloaded or invalid DeepSeek norm");
    validate(input, c.k, type_);
    validate(output, c.k, backend_ == Backend::Native ? type_ : Type::fp32);
    dispatch_.rmsnorm(slot, backend_, input, gamma, output, 1e-6f);
}

void Adapter::rope(size_t slot, View input, View cosine, View sine, View positions, View output, int sequence) {
    require(!closed_, "closed DeepSeek adapter");
    validate(input, 64, type_);
    dispatch_.rope(slot, backend_, input, cosine, sine, positions, output, sequence);
}

void Adapter::kv(size_t slot, View input, View wide, View latent, View normalized, View output) {
    require(!closed_ && backend_ == Backend::Native, "connected DeepSeek KV requires Native");
    validate(input, 2048, type_);
    dispatch_.kv_chain(slot, input, registry_.acquire(weights_.at(Tensor::KvA).key),
                       weights_.at(Tensor::KvNorm).view, registry_.acquire(weights_.at(Tensor::KvB).key),
                       wide, latent, normalized, output);
}

void Adapter::close() {
    if (closed_) return;
    for (auto &item : weights_) memory_.release(item.second.view.data);
    // The caller owns registry invalidation. Other adapters can share these packed entries.
    closed_ = true;
}
}
