"""Persistent terminal chat on the accepted synchronous Stage C.3 engine."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
from statistics import mean
import sys
from time import perf_counter


class ContextOverflow(ValueError):
    def __init__(self, prompt_tokens, message):
        super().__init__(message)
        self.prompt_tokens = prompt_tokens


def fit_history(tokenizer, history, text, limit, budget, truncate):
    pending = [*history, {"role": "user", "content": text}]
    first_turn = int(pending[0]["role"] == "system")
    removed = 0
    while True:
        tokens = tokenizer.apply_chat_template(pending, tokenize=True, add_generation_prompt=True,
                                               return_dict=False)
        if len(tokens) + budget <= limit:
            return pending, list(tokens), removed
        if not truncate or len(pending) - first_turn < 3:
            raise ContextOverflow(len(tokens), f"Prompt is {len(tokens)} tokens plus {budget} reserved output tokens. "
                                  f"Context limit is {limit}. Use /reset or shorten the prompt.")
        del pending[first_turn:first_turn + 2]
        removed += 1


class RoleBoundary:
    """Hold just the unresolved prefix of a structural role header."""

    markers = ("User:", "Assistant:")

    def __init__(self):
        self.pending = ""
        self.at_boundary = True
        self.stopped = False

    def feed(self, fragment):
        if self.stopped:
            return ""
        visible = []
        for char in fragment:
            if self.at_boundary:
                candidate = self.pending + char
                if candidate in self.markers:
                    self.pending = ""
                    self.stopped = True
                    break
                if any(marker.startswith(candidate) for marker in self.markers):
                    self.pending = candidate
                    continue
                if self.pending:
                    visible.append(self.pending)
                    self.pending = ""
                    self.at_boundary = False
            visible.append(char)
            self.at_boundary = char in "\r\n.!?。！？" or (self.at_boundary and char in " \t")
        return "".join(visible)

    def finish(self):
        trailing, self.pending = self.pending, ""
        return trailing


class Session:
    def __init__(self, engine, tokenize, sampling, args, emit=print, read=input):
        self.engine = engine
        self.tokenizer, self.sampling, self.args = tokenize, sampling, args
        self.emit, self.read = emit, read
        self.rows = []
        self.active = None
        self.stopping = False
        self.reading = False
        self.signal = None
        self.recovery_failed = False
        self.reset()

    def reset(self):
        self.history = [{"role": "system", "content": self.args.system}] if self.args.system else []

    def interrupt(self, signum, _frame):
        self.signal = signal.Signals(signum).name
        self.stopping = True
        if self.reading:
            raise KeyboardInterrupt

    def stats(self):
        ok = [row for row in self.rows if row["status"] == "ok"]
        tokens = sum(row["generated_tokens"] for row in ok)
        self.emit(f"Requests: {len(self.rows)} ({len(ok)} complete)\nGenerated tokens: {tokens}")
        for key, label in (("ttft_ms", "Average TTFT"), ("mean_tpot_ms", "Average TPOT")):
            values = [row[key] for row in ok if row[key] is not None]
            self.emit(f"{label}: {mean(values):.2f} ms" if values else f"{label}: unavailable")

    def turn(self, text):
        request_id = f"chat-{len(self.rows) + 1}"
        start = perf_counter()
        row = dict(request_id=request_id, phase="first" if not self.rows else "warm",
                   history_messages=len(self.history), prompt_tokens=0, generated_tokens=0,
                   dropped_turns=0, ttft_ms=None, mean_tpot_ms=None, p95_tpot_ms=None,
                   latency_ms=None, decode_tokens_s=None, streaming_fragments=0, boundary_ms=0.0,
                   finish_reason="", status="error", error="")
        arrivals, parts = [], []
        boundary = RoleBoundary()

        def show(fragment):
            if fragment:
                parts.append(fragment)
                self.emit(fragment, end="")
        try:
            pending, tokens, dropped = fit_history(self.tokenizer, self.history, text,
                                                    self.args.max_model_len, self.args.max_tokens,
                                                    self.args.history_policy == "truncate")
            row.update(prompt_tokens=len(tokens), dropped_turns=dropped, history_messages=len(pending) - 1)
            if self.stopping:
                raise KeyboardInterrupt
            self.active = request_id
            self.engine.add_request(request_id, {"prompt_token_ids": tokens}, self.sampling)
            self.emit("DeepSeek: ", end="")
            finished = False
            while not boundary.stopped and self.engine.has_unfinished_requests():
                if self.stopping:
                    raise KeyboardInterrupt
                outputs = self.engine.step()
                now = perf_counter()
                if self.stopping:
                    raise KeyboardInterrupt
                for output in outputs:
                    if output.request_id != request_id:
                        raise RuntimeError(f"Unexpected request output: {output.request_id}")
                    for delta in output.outputs:
                        if delta.token_ids:
                            arrivals.append((now, len(delta.token_ids)))
                        if delta.text:
                            row["streaming_fragments"] += 1
                            begin_filter = perf_counter()
                            visible = boundary.feed(delta.text)
                            row["boundary_ms"] += (perf_counter() - begin_filter) * 1000
                            show(visible)
                        if output.finished:
                            row["finish_reason"] = delta.finish_reason or ""
                finished |= any(output.finished for output in outputs)
            if boundary.stopped:
                row["finish_reason"] = "role_boundary"
                if not finished:
                    self.engine.abort_request([request_id])
                    self.active = None
            elif finished and row["finish_reason"] == "stop":
                show(boundary.finish())
            elif finished and row["finish_reason"] == "length":
                row["status"] = "incomplete"
                self.emit("\nOutput budget reached. This turn was not saved to history.")
                return
            else:
                raise RuntimeError("Engine ended without a normal stop")
            answer = "".join(parts)
            if not answer.strip() or not arrivals:
                row["status"] = "empty"
                self.emit("\nNo assistant reply was saved.")
                return
            self.history = [*pending, {"role": "assistant", "content": answer}]
            row["status"] = "ok"
            self.emit("")
        except ContextOverflow as exc:
            row.update(status="overflow", error=str(exc), prompt_tokens=exc.prompt_tokens)
            self.emit(str(exc))
        except KeyboardInterrupt:
            self.stopping = True
            row["status"] = "cancelled"
            self.emit("\nCancelling request and closing chat.")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            self.emit(f"\nGeneration failed: {row['error']}")
        finally:
            try:
                if self.active and row["status"] != "ok":
                    self.engine.abort_request([self.active])
            except Exception as exc:
                self.stopping = self.recovery_failed = True
                row["error"] += f" Abort failed: {exc}"
                self.emit("Engine recovery failed. Closing chat.")
            finally:
                self.active = None
                row["latency_ms"] = (perf_counter() - start) * 1000
                row["generated_tokens"] = sum(n for _, n in arrivals)
                if arrivals:
                    row["ttft_ms"] = (arrivals[0][0] - start) * 1000
                    elapsed = arrivals[-1][0] - arrivals[0][0]
                    decode_tokens = sum(n for _, n in arrivals[1:])
                    if elapsed > 0 and decode_tokens:
                        row["mean_tpot_ms"] = elapsed * 1000 / decode_tokens
                        row["decode_tokens_s"] = decode_tokens / elapsed
                    if len(arrivals) > 1 and all(n == 1 for _, n in arrivals):
                        values = sorted((b[0] - a[0]) * 1000 for a, b in zip(arrivals, arrivals[1:]))
                        row["p95_tpot_ms"] = values[math.ceil(.95 * len(values)) - 1]
                self.rows.append(row)

    def run(self):
        while not self.stopping:
            try:
                self.reading = True
                text = self.read("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            finally:
                self.reading = False
            if not text:
                continue
            if text in ("/exit", "/quit"):
                break
            if text == "/help":
                self.emit("Available commands: /help /reset /stats /exit /quit")
            elif text == "/reset":
                self.reset()
                self.emit("Conversation history cleared.")
            elif text == "/stats":
                self.stats()
            else:
                self.turn(text)


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def chat_worker_identity(worker):
    from ..phase15.deepseek_infer import worker_identity

    # Terminal Ctrl+C is sent to its foreground process group. Give the
    # worker its own group while preserving the executor's PID ownership.
    if os.getpgrp() != os.getpid():
        os.setpgrp()
    return dict(worker_identity(worker), process_group=os.getpgrp())


def load_engine(args):
    import torch
    import torch_mlu  # noqa: F401
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from ..phase15.stage_c3 import plugin
    from ..phase15.stage_c3.model import StageC3Model
    from ..phase15.stage_c3.moe import TunableDecoder
    from ..phase15.deepseek_infer import verify_workers

    if not torch.mlu.is_available() or torch.mlu.device_count() != 8:
        raise RuntimeError("Native chat requires exactly eight usable MLU devices")
    if any(torch.mlu.get_device_properties(i).name != "MLU590-M9" for i in range(8)):
        raise RuntimeError("Native chat requires MLU590-M9 devices")
    if os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise RuntimeError("Chat requires the frozen synchronous V1 engine mode")
    os.environ.update(NEURX_C2_VARIANT="candidate", NEURX_C3_EXPERIMENT="fused,batched",
                      NEURX_C2_PROFILE="0", NEURX_C2_CAPTURE_LOGITS="0", NEURX_C3_PROFILE="0")
    os.environ.setdefault("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "30")
    plugin.register()
    llm = LLM(model=args.model_dir, trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=8,
              distributed_executor_backend="mp", hf_overrides={"architectures": [plugin.ARCH]},
              enforce_eager=True, max_model_len=args.max_model_len, block_size=16, max_num_seqs=1,
              max_num_batched_tokens=128, num_gpu_blocks_override=16, gpu_memory_utilization=0.45,
              enable_prefix_caching=False, disable_log_stats=True, disable_custom_all_reduce=True)
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens,
                            output_kind=RequestOutputKind.DELTA, ignore_eos=False,
                            skip_special_tokens=True)
    def verify(rows):
        verify_workers(rows, StageC3Model, TunableDecoder)
        if any(row["process_group"] == os.getpgrp() for row in rows):
            raise RuntimeError("A worker remains in the chat terminal process group")

    return llm, params, chat_worker_identity, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default="/results", type=Path)
    parser.add_argument("--max-model-len", default=128, type=int)
    parser.add_argument("--max-tokens", default=32, type=int)
    parser.add_argument("--temperature", default=.7, type=float)
    parser.add_argument("--history-policy", choices=("truncate", "reject"), default="truncate")
    parser.add_argument("--system", default="")
    args = parser.parse_args()
    if not 0 < args.max_tokens < args.max_model_len <= 128 or not 0 <= args.temperature <= 2:
        parser.error("Accepted C.3 requires 0 < MAX_TOKENS < MAX_MODEL_LEN <= 128 and TEMPERATURE in [0, 2]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filtered_messages_log = args.output_dir / "filtered_notifications.log"
    with (args.output_dir / "chat_smoke_test.log").open("w", encoding="utf-8", buffering=1) as transcript:
        def emit(text, end="\n"):
            import re
            # Suppress internal context-management notifications from interactive console
            # Pattern: "Context limit: removed N oldest complete turn(s) from this prompt."
            context_limit_pattern = re.compile(r"Context limit: removed \d+ oldest complete turn\(s\) from this prompt\.")
            if context_limit_pattern.search(text):
                # Log the filtered message for diagnostic purposes
                with filtered_messages_log.open("a", encoding="utf-8") as f:
                    f.write(f"[filtered] {text}\n")
                    f.flush()
                # Do not emit to interactive console
                return
            print(text, end=end, flush=True)
            transcript.write(text + end)
            transcript.flush()

        def read(prompt):
            result = input(prompt)
            transcript.write(prompt + result + "\n")
            return result

        llm = session = None
        status = "error"
        summary = dict(backend="MLU", variant="stage_c3", experiment="fused,batched",
                       engine_loads=0, initialization_ms=None)
        start = perf_counter()
        workers = []
        original_handlers = {}
        try:
            emit("NeurX DeepSeek-V2-Lite Chat\nBackend: MLU-Operator | 8 × MLU590-M9 | BF16 TP8 CNCL | Stage C.3")
            emit("Loading model and initializing eight MLU workers...")
            llm, sampling, identity, verify = load_engine(args)
            summary["engine_loads"] = 1
            core = llm.llm_engine.engine_core
            workers = [handle.proc for handle in core.engine_core.model_executor.workers]
            rows = sorted(llm.collective_rpc(identity, timeout=60), key=lambda row: row["rank"])
            write_csv(args.output_dir / "workers.csv", rows)
            verify(rows)
            tokenizer = llm.get_tokenizer()
            if not tokenizer.chat_template:
                raise RuntimeError("Checkpoint tokenizer is missing its chat template")
            summary["initialization_ms"] = (perf_counter() - start) * 1000
            session = Session(llm.llm_engine, tokenizer, sampling, args, emit=emit, read=read)
            for signum in (signal.SIGINT, signal.SIGTERM):
                original_handlers[signum] = signal.signal(signum, session.interrupt)
            emit("Model ready. Streaming engine deltas. Type /help for commands.")
            session.run()
            if session.recovery_failed:
                raise RuntimeError("Chat stopped after engine recovery failed")
            status = "interrupted" if session.signal else "ok"
        except KeyboardInterrupt:
            status = "interrupted"
        except Exception as exc:
            summary["error"] = repr(exc)
            emit(f"Chat failed: {exc}")
            raise
        finally:
            try:
                if llm is not None:
                    emit("Closing model...")
                    llm.llm_engine.engine_core.shutdown(timeout=120)
                    lifecycle = [dict(pid=p.pid, exit_code=p.exitcode, alive=p.is_alive()) for p in workers]
                    write_csv(args.output_dir / "worker_lifecycle.csv", lifecycle)
                    if len(lifecycle) != 8 or any(r["exit_code"] != 0 or r["alive"] for r in lifecycle):
                        raise RuntimeError(f"Worker shutdown failed: {lifecycle}")
                    emit("All workers exited successfully.")
            except Exception as exc:
                status = "error"
                summary["shutdown_error"] = repr(exc)
                raise
            finally:
                requests = session.rows if session else []
                write_csv(args.output_dir / "interactive_requests.csv", requests)
                write_csv(args.output_dir / "performance.csv", [r for r in requests if r["status"] == "ok"])
                summary.update(status=status, requests=len(requests),
                               completed_requests=sum(r["status"] == "ok" for r in requests),
                               signal=session.signal if session else None,
                               generated_tokens=sum(r["generated_tokens"] for r in requests))
                (args.output_dir / "session_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                for signum, handler in original_handlers.items():
                    signal.signal(signum, handler)
                emit("Chat session closed.")


if __name__ == "__main__":
    main()
