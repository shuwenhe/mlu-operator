# Scheduler metadata

The new adapter consumes vendor-style `query_start_loc`, `seq_lens`, `block_table_tensor` and `slot_mapping`, aligned with the worker's active request IDs and scheduled positions. It validates all active spans before writing any cache data. It rejects duplicate request IDs, repeated or cross-request page ownership, out-of-range physical pages and slots that disagree with the block table and position. Its current contract requires exclusive pages and contiguous scheduled positions at the end of each committed sequence. The validator reads small metadata arrays on the host. Projected activations and KV cache remain device-resident.

The vendor control instruments `SchedulerWithProfiler.schedule` and `KVCacheManager.free` in the test process and records request IDs, token counts and physical blocks. The native adapter probe consumes deterministic fixture metadata in a separate process. Binding actual worker metadata, including its request ordering, to the native model is **BLOCKED**. [Vendor hardware log](../artifacts/logs/phase12_3_scheduler_hardware.log)

The final vendor control recorded a standalone five-token prompt and a batch containing five- and 65-token prompts. All three requests generated four tokens and logged a cache free. The 65-token request received blocks 2–6. Block 1 was reclaimed from the first request and reassigned to the next short request. The trace records three decode scheduling steps for each request. A retry that imported the vendor scheduler before the engine raised a vendor `check_stop` signature error. The final control attaches the trace after vendor engine initialization.

Run the vendor control in the pinned SDK with the writable cache environment from the compatibility audit:

```bash
/torch/venv3/pytorch_infer/bin/python3 -m integration.vllm_native_mlu.phase12_3.vendor_trace
```
