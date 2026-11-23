"""CPU-only session tests. Stub responses are never hardware evidence."""
from contextlib import redirect_stdout
from io import StringIO
import tempfile
from pathlib import Path
import json
import signal
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, call, patch

import deepseek_chat as chat


class Tokenizer:
    chat_template = "test template"

    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_dict"] is False
        self.messages.append([m.copy() for m in messages])
        return list(range(2 + sum(len(m["content"]) for m in messages)))


class Engine:
    def __init__(self):
        self.prompts, self.aborted = [], []
        self.pending = None
        self.fail = False

    def add_request(self, request, prompt, _params):
        self.prompts.append(prompt)
        self.pending = request
        self.index = 0

    def has_unfinished_requests(self):
        return self.pending is not None

    def step(self):
        if self.fail:
            self.fail = False
            raise ValueError("injected")
        request = self.pending
        text = ["你", "好"][self.index]
        self.index += 1
        if self.index == 2:
            self.pending = None
        return [NS(request_id=request, finished=self.index == 2, outputs=[NS(text=text, token_ids=[1], finish_reason="stop" if self.index == 2 else None)])]

    def abort_request(self, request_ids):
        self.aborted.extend(request_ids)
        self.pending = None


class ChatTests(unittest.TestCase):
    def make_session(self, lines=(), budget=8, limit=48):
        stream = iter(lines)
        engine, tokenizer, emit = Engine(), Tokenizer(), Mock()
        session = chat.Session(engine, tokenizer, None, NS(system="system", max_model_len=limit,
                               max_tokens=budget, history_policy="truncate"), emit, lambda _: next(stream))
        return session, engine, tokenizer, emit

    def test_history_commands_unicode_and_empty(self):
        session, engine, tokenizer, emit = self.make_session(
            ["", "/help", "你好", "English", "/stats", "/reset", "再见", "/quit"])
        session.run()
        self.assertEqual(len(engine.prompts), 3)
        self.assertEqual([m["role"] for m in tokenizer.messages[1]], ["system", "user", "assistant", "user"])
        self.assertEqual(len(tokenizer.messages[2]), 2)
        self.assertEqual(session.history[-1]["content"], "你好")
        emit.assert_any_call("你", end="")
        emit.assert_any_call("好", end="")
        emit.assert_any_call("Requests: 2 (2 complete)\nGenerated tokens: 4")
        emit.assert_any_call("Conversation history cleared.")

    def test_normal_chat_and_trimming_are_silent(self):
        budget, limit = 32, 128
        for dropped in (0, 1, 2):
            with self.subTest(dropped=dropped):
                session, engine, tokenizer, emit = self.make_session(
                    ["next", "/exit"], budget=budget, limit=limit)
                retained = [{"role": "user", "content": "u" * (50 if dropped else 1)},
                            {"role": "assistant", "content": "a" * (34 if dropped else 1)}]
                for i in range(dropped):
                    session.history.extend([{"role": "user", "content": f"old {i}"},
                                            {"role": "assistant", "content": f"reply {i}"}])
                session.history.extend(retained)
                expected = [{"role": "system", "content": "system"}, *retained,
                            {"role": "user", "content": "next"}]

                session.run()

                self.assertEqual(session.rows[0]["dropped_turns"], dropped)
                self.assertEqual(len(tokenizer.messages), dropped + 1)
                self.assertEqual(tokenizer.messages[-1], expected)
                self.assertEqual(session.history, [
                    *expected, {"role": "assistant", "content": "你好"},
                ])
                self.assertEqual(len(engine.prompts), 1)
                prompt_tokens = len(engine.prompts[0]["prompt_token_ids"])
                self.assertEqual(prompt_tokens, session.rows[0]["prompt_tokens"])
                self.assertLessEqual(prompt_tokens + budget, limit)
                if dropped:
                    self.assertEqual(prompt_tokens + budget, limit)
                self.assertEqual(emit.mock_calls, [
                    call("DeepSeek: ", end=""), call("你", end=""), call("好", end=""), call(""),
                ])

    def test_oversized_input_and_reject_policy_keep_history(self):
        session, engine, _, emit = self.make_session(limit=24)
        session.turn("first")
        history = session.history.copy()
        session.turn("x" * 100)
        self.assertEqual(session.rows[-1]["status"], "overflow")
        self.assertEqual(session.history, history)
        emit.assert_any_call("Prompt is 108 tokens plus 8 reserved output tokens. "
                             "Context limit is 24. Use /reset or shorten the prompt.")
        session.args.history_policy = "reject"
        session.turn("again")
        self.assertEqual(session.rows[-1]["status"], "overflow")
        self.assertEqual(session.history, history)
        self.assertEqual(len(engine.prompts), 1)

    def test_active_interrupt_and_error_recovery(self):
        session, engine, _, _ = self.make_session()
        engine.fail = True
        session.turn("fail")
        session.turn("ok")
        self.assertEqual([r["status"] for r in session.rows], ["error", "ok"])
        engine.step = lambda: (session.interrupt(signal.SIGINT, None) or [])
        session.turn("cancel")
        self.assertEqual(session.rows[-1]["status"], "cancelled")
        self.assertEqual(engine.aborted, ["chat-1", "chat-3"])

    def test_startup_failure_and_worker_shutdown(self):
        for fail in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                worker = NS(pid=12, exitcode=0, is_alive=lambda: False)
                core = NS(shutdown=Mock(), engine_core=NS(model_executor=NS(
                    workers=[NS(proc=worker) for _ in range(8)])))
                engine = Engine()
                engine.engine_core = core
                llm = NS(llm_engine=engine, collective_rpc=lambda *a, **k: [],
                         get_tokenizer=Tokenizer)
                with patch.object(sys, "argv", ["chat", "--model-dir", "mock", "--output-dir", tmp]), \
                     patch("builtins.input", return_value="/exit"), patch("builtins.print"), \
                     patch.object(chat, "load_engine", side_effect=RuntimeError("init") if fail else None,
                                  return_value=(llm, None, None, lambda _: None)) as load:
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "init"):
                            chat.main()
                    else:
                        chat.main()
                        core.shutdown.assert_called_once_with(timeout=120)
                        self.assertTrue((Path(tmp) / "worker_lifecycle.csv").exists())
                    load.assert_called_once()
                summary = json.loads((Path(tmp) / "session_summary.json").read_text())
                self.assertEqual(summary["status"], "error" if fail else "ok")

    def test_context_limit_notification_filtering(self):
        """Exercise the production terminal and transcript, including split notices."""
        notice = "Context limit: removed 12 oldest complete turn(s) from this prompt."
        expected = "DeepSeek: 你好 Context limit is 128. Context\n"

        def run(session):
            session.emit("Context limit: removed 1 oldest complete turn(s) from this prompt.")
            session.emit("DeepSeek: ", end="")
            session.emit("你" + notice[:16], end="")
            session.emit(notice[16:23], end="")
            session.emit(notice[23:] + "好 ", end="")
            session.emit("Context limit is 128. ", end="")
            session.emit("Context", end="")
            session.emit("")

        with tempfile.TemporaryDirectory() as tmp:
            worker = NS(pid=12, exitcode=0, is_alive=lambda: False)
            core = NS(shutdown=Mock(), engine_core=NS(model_executor=NS(
                workers=[NS(proc=worker) for _ in range(8)])))
            llm = NS(llm_engine=NS(engine_core=core), collective_rpc=lambda *a, **k: [],
                     get_tokenizer=Tokenizer)
            stdout = StringIO()
            with patch.object(sys, "argv", ["chat", "--model-dir", "mock", "--output-dir", tmp]), \
                 patch.object(chat, "load_engine", return_value=(llm, None, None, lambda _: None)), \
                 patch.object(chat.Session, "run", run), redirect_stdout(stdout):
                chat.main()

            for output in (stdout.getvalue(), (Path(tmp) / "chat_smoke_test.log").read_text()):
                self.assertIn(expected, output)
                self.assertNotIn("Context limit: removed", output)
            filtered = (Path(tmp) / "filtered_notifications.log").read_text()
            self.assertIn(notice, filtered)
            self.assertIn("removed 1 oldest", filtered)


if __name__ == "__main__":
    unittest.main()
