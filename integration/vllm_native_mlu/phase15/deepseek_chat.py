"""Interactive Stage C.3 TP8 chat entrypoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from statistics import mean
from time import perf_counter


BANNER = """================================================
 NeurX DeepSeek-V2-Lite Interactive Chat
================================================
Backend: MLU-Operator
Hardware: 8 x MLU590-M9
Tensor Parallel: 8
Precision: BF16
Variant: Stage C.3
Attention: Output-axis MLA
MoE: Fused Gate/Up + Expert Batching
Communication: CNCL
"""

HELP = """Commands:
  /help          Show help
  /stats         Show the latest request stats
  /reset         Clear conversation history
  exit|quit|/exit  Exit chat
"""


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, int((len(ordered) - 1) * 0.95))
    return ordered[idx]


def abort_request(engine, request_id: str) -> None:
    if hasattr(engine, "abort_request"):
        engine.abort_request(request_id)
        return
    if hasattr(engine, "abort_requests"):
        engine.abort_requests([request_id])
        return


def apply_template(tokenizer, messages: list[dict]) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    text = []
    for message in messages:
        if message["role"] == "user":
            text.append(f"User: {message['content']}")
        elif message["role"] == "assistant":
            text.append(f"Assistant: {message['content']}")
    text.append("Assistant:")
    return tokenizer.encode("\n".join(text))


def prepare_prompt(tokenizer, history: list[dict], max_model_len: int, max_tokens: int) -> tuple[list[dict], list[int], int]:
    trimmed = list(history)
    dropped = 0
    while True:
        prompt_ids = apply_template(tokenizer, trimmed)
        if len(prompt_ids) + max_tokens <= max_model_len:
            return trimmed, prompt_ids, dropped
        if len(trimmed) <= 1:
            raise RuntimeError(
                f"Prompt too long: prompt_tokens={len(prompt_ids)}, max_model_len={max_model_len}, max_tokens={max_tokens}"
            )
        # Drop the oldest user+assistant messages.
        trimmed = trimmed[2:]
        dropped += 1


def stream_generate(engine, prompt_ids: list[int], max_tokens: int, temperature: float) -> tuple[str, list[int], dict, bool]:
    from vllm import SamplingParams

    request_id = f"chat-{uuid.uuid4().hex}"
    params = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens, ignore_eos=False)
    engine.add_request(request_id, {"prompt_token_ids": prompt_ids}, params)

    text = ""
    tokens: list[int] = []
    token_timestamps: list[float] = []
    interrupted = False
    start = perf_counter()
    while engine.has_unfinished_requests():
        try:
            outputs = engine.step()
        except KeyboardInterrupt:
            interrupted = True
            abort_request(engine, request_id)
            break
        now = perf_counter()
        for result in outputs:
            output = result.outputs[0]
            current = list(output.token_ids)
            if len(current) <= len(tokens):
                continue
            generated_now = len(current) - len(tokens)
            tokens = current
            full_text = output.text
            delta = full_text[len(text) :]
            text = full_text
            if delta:
                print(delta, end="", flush=True)
            token_timestamps.extend([now] * generated_now)
    print("", flush=True)

    intervals = [(b - a) * 1000.0 for a, b in zip(token_timestamps, token_timestamps[1:])]
    elapsed_ms = (perf_counter() - start) * 1000.0
    ttft_ms = (token_timestamps[0] - start) * 1000.0 if token_timestamps else None
    mean_tpot_ms = mean(intervals) if intervals else None
    p95_tpot_ms = p95(intervals)
    decode_tps = (1000.0 / mean_tpot_ms) if mean_tpot_ms and mean_tpot_ms > 0 else None
    metrics = {
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(tokens),
        "ttft_ms": ttft_ms,
        "mean_tpot_ms": mean_tpot_ms,
        "p95_tpot_ms": p95_tpot_ms,
        "decode_tokens_per_s": decode_tps,
        "end_to_end_latency_ms": elapsed_ms,
    }
    return text, tokens, metrics, interrupted


def print_stats(stats: dict | None) -> None:
    if not stats:
        print("No completed request stats yet.", flush=True)
        return
    print(
        "Stats: "
        f"prompt_tokens={stats['prompt_tokens']} "
        f"generated_tokens={stats['generated_tokens']} "
        f"ttft_ms={stats['ttft_ms'] if stats['ttft_ms'] is not None else 'N/A'} "
        f"mean_tpot_ms={stats['mean_tpot_ms'] if stats['mean_tpot_ms'] is not None else 'N/A'} "
        f"p95_tpot_ms={stats['p95_tpot_ms'] if stats['p95_tpot_ms'] is not None else 'N/A'} "
        f"decode_tps={stats['decode_tokens_per_s'] if stats['decode_tokens_per_s'] is not None else 'N/A'} "
        f"latency_ms={stats['end_to_end_latency_ms']}",
        flush=True,
    )


def run_mock(output_dir: Path, max_tokens: int) -> None:
    history: list[dict] = []
    rows: list[dict] = []
    perf_rows: list[dict] = []
    token_parity_rows: list[dict] = []
    last_stats: dict | None = None
    print(BANNER, flush=True)
    print("Model loading...", flush=True)
    print("Model ready (mock mode).", flush=True)
    turn = 0
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break
        if user_input in {"exit", "quit", "/exit"}:
            break
        if user_input == "/help":
            print(HELP, flush=True)
            continue
        if user_input == "/reset":
            history.clear()
            print("Conversation history cleared.", flush=True)
            continue
        if user_input == "/stats":
            print_stats(last_stats)
            continue
        if not user_input:
            continue

        turn += 1
        start = perf_counter()
        reply = f"Mock DeepSeek: {user_input[: max_tokens]}"
        print("DeepSeek: ", end="", flush=True)
        print(reply, flush=True)
        elapsed = (perf_counter() - start) * 1000.0
        last_stats = {
            "prompt_tokens": len(user_input),
            "generated_tokens": len(reply),
            "ttft_ms": 0.1,
            "mean_tpot_ms": 0.1,
            "p95_tpot_ms": 0.1,
            "decode_tokens_per_s": 10000.0,
            "end_to_end_latency_ms": elapsed,
        }
        history.extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": reply}])
        rows.append(
            {
                "turn": turn,
                "prompt_sha256": text_hash(user_input),
                "response_sha256": text_hash(reply),
                "prompt_chars": len(user_input),
                "response_chars": len(reply),
                "prompt_tokens": last_stats["prompt_tokens"],
                "generated_tokens": last_stats["generated_tokens"],
                "interrupted": False,
                "history_messages": len(history),
            }
        )
        perf_rows.append({"turn": turn, **last_stats})
        token_parity_rows.append({"turn": turn, "status": "NOT_APPLICABLE", "notes": "mock mode"})

    write_csv(output_dir / "interactive_sessions.csv", rows, list(rows[0].keys()) if rows else [
        "turn", "prompt_sha256", "response_sha256", "prompt_chars", "response_chars",
        "prompt_tokens", "generated_tokens", "interrupted", "history_messages"
    ])
    write_csv(output_dir / "performance.csv", perf_rows, list(perf_rows[0].keys()) if perf_rows else [
        "turn", "prompt_tokens", "generated_tokens", "ttft_ms", "mean_tpot_ms",
        "p95_tpot_ms", "decode_tokens_per_s", "end_to_end_latency_ms"
    ])
    write_csv(output_dir / "token_parity.csv", token_parity_rows, list(token_parity_rows[0].keys()) if token_parity_rows else [
        "turn", "status", "notes"
    ])
    write_csv(output_dir / "worker_lifecycle.csv", [{"phase": "mock", "pid": os.getpid(), "exit_code": 0, "alive": False}], [
        "phase", "pid", "exit_code", "alive"
    ])
    (output_dir / "final_report.md").write_text("# DeepSeek Chat Report\n\nMode: mock\n\nOverall: PASS\n")
    print("Chat session closed.", flush=True)


def run_real(args: argparse.Namespace, output_dir: Path) -> int:
    import torch
    import torch_mlu  # noqa: F401
    from vllm import LLM

    from .deepseek_infer import verify_workers, worker_identity
    from .stage_c3 import plugin as stage_c3
    from .stage_c3.model import StageC3Model
    from .stage_c3.moe import TunableDecoder

    if args.variant != "stage_c3":
        raise ValueError(f"deepseek-chat only supports stage_c3, got {args.variant}")
    if args.experiment != "fused,batched":
        raise ValueError("deepseek-chat requires experiment=fused,batched")
    if args.tp != 8 or args.dtype != "bfloat16":
        raise ValueError("deepseek-chat requires TP=8 and BF16")
    if not torch.mlu.is_available() or torch.mlu.device_count() != 8:
        raise RuntimeError("Expected exactly 8 available MLU devices")

    os.environ.update(
        NEURX_C2_VARIANT="candidate",
        NEURX_C3_EXPERIMENT=args.experiment,
        NEURX_C2_PROFILE="0",
        NEURX_C2_CAPTURE_LOGITS="0",
        NEURX_C3_PROFILE="0",
    )
    os.environ.setdefault("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "30")

    stage_c3.register()
    history: list[dict] = []
    session_rows: list[dict] = []
    perf_rows: list[dict] = []
    token_parity_rows: list[dict] = []
    worker_lifecycle_rows: list[dict] = []
    last_stats: dict | None = None
    verdict = "PASS"
    error_message = ""

    print(BANNER, flush=True)
    print("Model loading...", flush=True)
    llm = LLM(
        model=args.model_dir,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        distributed_executor_backend="mp",
        hf_overrides={"architectures": [stage_c3.ARCH]},
        enforce_eager=True,
        max_model_len=args.max_model_len,
        block_size=16,
        max_num_seqs=1,
        max_num_batched_tokens=min(2048, args.max_model_len),
        num_gpu_blocks_override=16,
        gpu_memory_utilization=0.45,
        enable_prefix_caching=False,
        disable_log_stats=True,
        disable_custom_all_reduce=True,
    )
    print("Model ready.", flush=True)
    engine = llm.llm_engine
    core = engine.engine_core
    workers = [handle.proc for handle in core.engine_core.model_executor.workers]

    try:
        rows = sorted(llm.collective_rpc(worker_identity, timeout=60), key=lambda row: row["rank"])
        verify_workers(rows, StageC3Model, TunableDecoder)
        for row in rows:
            worker_lifecycle_rows.append(
                {
                    "phase": "initialized",
                    "rank": row["rank"],
                    "pid": row["pid"],
                    "exit_code": "",
                    "alive": True,
                    "backend": row["backend"],
                    "device": row["device"],
                    "model_class": row["model_class"],
                }
            )

        tokenizer = llm.get_tokenizer()
        turn = 0
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nType 'exit' to close chat.", flush=True)
                continue

            if user_input in {"exit", "quit", "/exit"}:
                break
            if user_input == "/help":
                print(HELP, flush=True)
                continue
            if user_input == "/reset":
                history.clear()
                print("Conversation history cleared.", flush=True)
                continue
            if user_input == "/stats":
                print_stats(last_stats)
                continue
            if not user_input:
                continue

            turn += 1
            history.append({"role": "user", "content": user_input})
            history, prompt_ids, dropped = prepare_prompt(tokenizer, history, args.max_model_len, args.max_tokens)
            if dropped:
                print(f"[context] Dropped {dropped} oldest turns to fit max_model_len.", flush=True)

            print("DeepSeek: ", end="", flush=True)
            response_text, response_tokens, metrics, interrupted = stream_generate(
                engine, prompt_ids, args.max_tokens, args.temperature
            )
            if interrupted:
                print("[generation interrupted]", flush=True)
            if response_text:
                history.append({"role": "assistant", "content": response_text})

            last_stats = metrics
            session_rows.append(
                {
                    "turn": turn,
                    "prompt_sha256": text_hash(user_input),
                    "response_sha256": text_hash(response_text),
                    "prompt_chars": len(user_input),
                    "response_chars": len(response_text),
                    "prompt_tokens": metrics["prompt_tokens"],
                    "generated_tokens": metrics["generated_tokens"],
                    "interrupted": interrupted,
                    "history_messages": len(history),
                }
            )
            perf_rows.append({"turn": turn, **metrics})
            token_parity_rows.append(
                {
                    "turn": turn,
                    "status": "NOT_APPLICABLE",
                    "notes": "Interactive free-form prompts do not have frozen token references",
                }
            )
    except Exception as exc:
        verdict = "FAIL"
        error_message = str(exc)
        raise
    finally:
        core.shutdown(timeout=120)
        for rank, proc in enumerate(workers):
            worker_lifecycle_rows.append(
                {
                    "phase": "shutdown",
                    "rank": rank,
                    "pid": proc.pid,
                    "exit_code": proc.exitcode,
                    "alive": proc.is_alive(),
                    "backend": "cncl",
                    "device": f"mlu:{rank}",
                    "model_class": "",
                }
            )
        if any(proc.is_alive() or proc.exitcode != 0 for proc in workers):
            verdict = "FAIL"
            if not error_message:
                error_message = "Worker lifecycle failure during shutdown"

        write_csv(
            output_dir / "interactive_sessions.csv",
            session_rows,
            [
                "turn",
                "prompt_sha256",
                "response_sha256",
                "prompt_chars",
                "response_chars",
                "prompt_tokens",
                "generated_tokens",
                "interrupted",
                "history_messages",
            ],
        )
        write_csv(
            output_dir / "performance.csv",
            perf_rows,
            [
                "turn",
                "prompt_tokens",
                "generated_tokens",
                "ttft_ms",
                "mean_tpot_ms",
                "p95_tpot_ms",
                "decode_tokens_per_s",
                "end_to_end_latency_ms",
            ],
        )
        write_csv(output_dir / "worker_lifecycle.csv", worker_lifecycle_rows, [
            "phase", "rank", "pid", "exit_code", "alive", "backend", "device", "model_class"
        ])
        write_csv(output_dir / "token_parity.csv", token_parity_rows, ["turn", "status", "notes"])
        report = [
            "# DeepSeek Chat Final Report",
            "",
            f"- Variant: {args.variant}",
            f"- Experiment: {args.experiment}",
            f"- TP: {args.tp}",
            f"- Dtype: {args.dtype}",
            f"- Turns completed: {len(session_rows)}",
            f"- Worker shutdown clean: {'yes' if verdict == 'PASS' else 'no'}",
            f"- Error: {error_message or 'none'}",
            "",
            f"Overall: {verdict}",
            "",
        ]
        (output_dir / "final_report.md").write_text("\n".join(report))
    print("Chat session closed.", flush=True)
    print("All workers exited successfully." if verdict == "PASS" else "Worker shutdown had issues.", flush=True)
    return 0 if verdict == "PASS" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--variant", default="stage_c3")
    parser.add_argument("--experiment", default="fused,batched")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-dir", default="/results")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.mock:
        run_mock(out, args.max_tokens)
        return 0
    return run_real(args, out)


if __name__ == "__main__":
    sys.exit(main())
