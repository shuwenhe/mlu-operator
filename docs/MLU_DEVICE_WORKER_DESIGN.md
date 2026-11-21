# MLU device worker

[Worker and registry source](../runtime/native/runtime.cpp) binds one context to one owner thread and physical PCI identifier. Construction resolves the BDF through `cnrtDeviceGetByPCIBusId`, reads it back with `cnrtDeviceGetPCIBusId`, and records CNRT properties. The ordinal is scoped to the device mapping in that process. Every queue operation reselects that worker's ordinal and rejects calls originating on another thread. Queues are indexed within the worker and destroyed only after synchronization. Metadata and registry operations are intended to run on the same owning thread.

Lifecycle is construction, ready, faulted, recovered or closed. A failed synchronization stores the original CNRT status and blocks queue acquisition and managed allocation. Recovery checks all surviving queues before clearing the recorded fault. If a workspace cannot guarantee safe teardown on destruction, the process terminates without freeing potentially referenced resources. This is a fail-stop behavior. Runtime restart and unrecoverable device-fault remediation are external responsibilities.

**Hardware PASS, R1.** In a disposable SDK context mapping one card, the worker resolved `0000:4f:00.0` to ordinal 0 and reported MLU590-M9, 98,304 MiB. Both queue initialization and destruction were exercised. The image uses CNRT 7.7.3/CNNL 2.2.18 and CNCC 5.7.6. Existing Team mappings and software were untouched.

[R1/R2 hardware log](../artifacts/logs/runtime_r12_hardware.log)
