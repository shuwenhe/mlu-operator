# MLU memory manager acceptance

[Memory manager](../runtime/native/workspace.cpp) tracks explicitly owned allocations in weight, cache and workspace buckets, with live and peak counters and a configurable driver free-memory floor. The floor is a policy reserve, not an allocation or measurement of CNRT internal reservation. Driver total/free is captured independently with `cnrtMemGetInfo`. The packed registry reports a separate byte total because allocations occur in the accepted packing API. The managed peak excludes registry bytes and vendor internal allocations. Reserve enforcement currently applies only to managed allocation calls.

**Hardware PASS, R4.** The small failure-injection test forced the second slot allocation to return `cnrtErrorNoMem` and observed rollback to zero live bytes. It measured 288 peak managed bytes with the configured 4,096-byte reserve floor. It exercised distinct weight/cache/workspace buckets and released all managed memory. Actual exhaustion was not provoked. Queue fault recovery used an injected sync status and subsequent real synchronization. See the single-layer report for real checkpoint allocation sizes and the final residual counters.

[R3/R4 transcript](../artifacts/logs/runtime_r34_hardware.log) · [Final suite](../artifacts/logs/runtime_final_hardware.log)
