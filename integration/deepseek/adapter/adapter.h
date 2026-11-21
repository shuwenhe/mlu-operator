#ifndef DEEPSEEK_MODEL_ADAPTER_H
#define DEEPSEEK_MODEL_ADAPTER_H
#include "dispatch.h"

namespace deepseek {
enum class Tensor { Q, KvA, KvB, Output, Gate, Up, InputNorm, KvNorm, PostNorm };
struct Contract {
    const char *alias;
    const char *checkpoint_name;
    int k, n;
};
const Contract &contract(Tensor);
float decode(uint16_t, native_mlu::Type);
uint16_t encode(float, native_mlu::Type);
std::vector<uint16_t> load_words(const std::string &, size_t, native_mlu::Type);

class Adapter {
public:
    Adapter(native_mlu::Worker &, native_mlu::Registry &, native_mlu::Memory &,
            native_mlu::Dispatch &, const std::string &, native_mlu::Backend, native_mlu::Type);
    Adapter(const Adapter &) = delete;
    ~Adapter();
    native_mlu::Backend backend() const { return backend_; }
    void load(Tensor);
    void project(size_t slot, Tensor, native_mlu::View input, native_mlu::View output);
    void norm(size_t slot, Tensor, native_mlu::View input, native_mlu::View output);
    void rope(size_t slot, native_mlu::View input, native_mlu::View cosine, native_mlu::View sine,
              native_mlu::View positions, native_mlu::View output, int sequence);
    void kv(size_t slot, native_mlu::View input, native_mlu::View wide,
            native_mlu::View latent, native_mlu::View normalized, native_mlu::View output);
    void close();
private:
    struct Weight { native_mlu::WeightKey key; native_mlu::View view{}; };
    native_mlu::Worker &worker_;
    native_mlu::Registry &registry_;
    native_mlu::Memory &memory_;
    native_mlu::Dispatch &dispatch_;
    std::string dir_, revision_;
    native_mlu::Backend backend_;
    native_mlu::Type type_;
    std::map<Tensor, Weight> weights_;
    bool closed_ = false;
};
}
#endif
