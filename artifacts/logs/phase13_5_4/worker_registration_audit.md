# Phase 13.5.4 Worker Registration Audit

## Scope
- Runtime image: `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`
- Runtime vLLM source:
  - `/torch/venv3/pytorch_infer/lib/python3.12/site-packages/vllm/v1/executor/multiproc_executor.py`
  - `/torch/venv3/pytorch_infer/lib/python3.12/site-packages/vllm/model_executor/models/registry.py`
  - `/torch/venv3/pytorch_infer/lib/python3.12/site-packages/vllm/plugins/__init__.py`
- NeurX registration path:
  - `/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase12_5/run.py`

## Observed failure (Phase 13.5.3 TP2)
- Error text: `Model architectures ['NeurXDeepSeekV2ForCausalLM'] are not supported for now`
- Source of error string confirmed in `registry.py` (`_ModelRegistry._raise_for_unsupported`).

## Actual Vendor behavior (from installed source)
1. `LLM(..., tensor_parallel_size=2)` enters multiprocess executor path.
2. Worker processes start with independent Python interpreters (`spawn` path).
3. `ModelRegistry` is process-local global singleton in each interpreter (`registry.py`).
4. Unsupported architecture check happens in worker-side registry resolution.
5. Main-process `ModelRegistry.register_model(...)` does not automatically propagate to worker registries.

## Why main registration did not reach Worker
- Existing NeurX code registers model only when `register()` is called explicitly in main flow (`phase12_5/run.py`).
- Worker startup does not replay that explicit call unless the same registration code runs in worker interpreter before model resolution.

## Answers to required questions
1. Why main-process registration did not enter Worker?
   - Worker has independent interpreter/registry; explicit registration call was main-process-only.
2. Does Worker use independent ModelRegistry?
   - Yes (process-local singleton initialized per worker process).
3. Does Worker rebuild allowlist?
   - Yes, from its own registry state (`_VLLM_MODELS` + anything registered in that process).
4. Does custom registration occur before Worker init?
   - Previously: no guaranteed worker-side registration before resolution.
5. Is registration lost at startup?
   - Not "lost"; never executed in worker process.
6. Official extension interface?
   - Vendor includes general plugin hook (`vllm.general_plugins`, loaded by `load_general_plugins()`), but it requires installed Python entry points. Current NeurX repo path is not installed as a plugin entry point in runtime.

## Root cause
- **Root cause:** custom architecture registration was not executed in worker interpreters before worker-side model architecture resolution.

## Minimal fix direction
- Ensure NeurX registration executes in every importing process (including worker subprocess) before `LLM` worker resolves architecture.
- Keep Vendor runtime contracts unchanged; avoid kernel/math changes.
