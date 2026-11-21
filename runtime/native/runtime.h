#ifndef NATIVE_MLU_RUNTIME_H
#define NATIVE_MLU_RUNTIME_H
#include "packed_weight.h"
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace native_mlu {
class Error : public std::runtime_error {
public:
    Error(cnrtRet_t code, const char *op);
    const cnrtRet_t code;
};
void check(cnrtRet_t status,const char *op);

class Worker {
public:
    explicit Worker(const std::string &pci);
    ~Worker();
    Worker(const Worker&)=delete;
    Worker& operator=(const Worker&)=delete;
    const std::string& pci() const {return pci_;}
    int ordinal() const {return ordinal_;}
    const cnrtDeviceProp_t& properties() const {return properties_;}
    cnrtRet_t fault() const {return fault_;}
    void select() const;
    size_t add_queue();
    cnrtQueue_t queue(size_t id) const;
    void sync(size_t id);
    void recover();
    void close();
private:
    std::thread::id owner_=std::this_thread::get_id();
    std::string pci_;
    int ordinal_=-1;
    cnrtDeviceProp_t properties_{};
    std::vector<cnrtQueue_t> queues_;
    cnrtRet_t fault_=cnrtSuccess;
    bool closed_=false;
};

struct WeightKey {
    std::string revision,tensor,shard,layout,pci;
    uint64_t offset;
    mlu::WeightType type;
    int k,n;
    bool operator<(const WeightKey &other) const;
};
class Registry {
    struct Entry {
        mlu::PackedWeight *weight=nullptr;
        size_t queue=0;
        bool published=false,valid=true;
    };
public:
    using Lease=std::shared_ptr<const mlu::PackedWeight>;
    explicit Registry(Worker &worker):worker_(worker){}
    Registry(const Registry&)=delete;
    Registry& operator=(const Registry&)=delete;
    ~Registry();
    void host(const WeightKey&,const uint16_t*);
    void device(const WeightKey&,const uint16_t*,size_t queue);
    void publish(const WeightKey&);
    Lease acquire(const WeightKey&) const;
    void invalidate(const std::string &revision);
    void collect();
    void close();
    size_t bytes() const;
private:
    Entry& insert(const WeightKey&);
    Worker &worker_;
    std::map<WeightKey,std::shared_ptr<Entry>> entries_;
};
}
#endif
