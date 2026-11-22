"""Streaming and history regression cases for Phase 16.3."""
import signal
from types import SimpleNamespace as NS
import unittest

import test_chat
from deepseek_chat import RoleBoundary


class BoundaryTests(unittest.TestCase):
    def test_partition_invariance(self):
        cases = (
            ("3", "3"),
            ("OK.User: I am a student.", "OK."),
            ("Hello.Assistant: Another answer.", "Hello."),
            ("6. User: 3*3", "6. "),
            ("9\nUser: 4*4", "9\n"),
            ("The User: field identifies the requester.", "The User: field identifies the requester."),
            ('The label is "User:".', 'The label is "User:".'),
            ("Compute for the User", "Compute for the User"),
            ("Assistant: Hello", ""),
        )
        for text, expected in cases:
            for split in range(len(text) + 1):
                with self.subTest(text=text, split=split):
                    filter_ = RoleBoundary()
                    chunks = [text[:split], "", *text[split:]]
                    self.assertEqual("".join(map(filter_.feed, chunks)) + filter_.finish(), expected)

    def test_flushes_ordinary_text_without_chunk_delay(self):
        filter_ = RoleBoundary()
        self.assertEqual(filter_.feed("OK."), "OK.")
        self.assertEqual(filter_.feed("User"), "")
        self.assertEqual(filter_.feed(" is a word"), "User is a word")

    def session_for(self, chunks, *, reason="stop", interrupt=False, complete=True):
        session, engine, tokenizer, emit = test_chat.ChatTests().make_session()
        fragments = iter(chunks)
        total = len(chunks)
        step = 0

        def generate():
            nonlocal step
            step += 1
            text = next(fragments)
            if interrupt and step == total:
                session.interrupt(signal.SIGINT, None)
            final = step == total
            request = engine.pending
            if final:
                engine.pending = None
            return [NS(request_id=request, finished=final and complete, outputs=[
                NS(text=text, token_ids=[1] if text else [], finish_reason=reason if final else None)
            ])]

        engine.step = generate
        return session, engine, tokenizer, emit

    def test_boundary_aborts_and_only_clean_history_is_reused(self):
        session, engine, tokenizer, emit = self.session_for(["OK.User", ": I am", " a student."])
        session.turn("hello")
        self.assertEqual(session.history[-1]["content"], "OK.")
        self.assertEqual(engine.aborted, ["chat-1"])
        emit.assert_any_call("OK.", end="")
        self.assertFalse(any("User" in str(c) for c in emit.call_args_list))
        next_session, _, _, _ = self.session_for(["3", "", ""], reason="stop")
        next_session.history = session.history
        next_session.tokenizer = tokenizer
        next_session.turn("1+2")
        self.assertEqual(tokenizer.messages[-1][-2]["content"], "OK.")
        self.assertEqual(next_session.history[-1]["content"], "3")

    def test_eos_empty_deltas_and_normal_completion(self):
        session, _, _, _ = self.session_for(["", "The User: field", ""])
        session.turn("hello")
        self.assertEqual(session.rows[-1]["status"], "ok")
        self.assertEqual(session.history[-1]["content"], "The User: field")

    def test_no_history_on_length_or_unexpected_end(self):
        for reason, complete, expected in [("length", True, "incomplete"), ("stop", False, "error")]:
            session, _, _, emit = self.session_for(["OK.", "Assis"], reason=reason, complete=complete)
            session.turn("hello")
            self.assertEqual(session.rows[-1]["status"], expected)
            self.assertEqual(len(session.history), 1)
            self.assertFalse(any("Assis" in str(c) for c in emit.call_args_list))

    def test_cancel_discards_pending_prefix(self):
        session, engine, _, emit = self.session_for(["OK.Us", "er"], interrupt=True)
        session.turn("hello")
        self.assertEqual(session.rows[-1]["status"], "cancelled")
        self.assertEqual(engine.aborted, ["chat-1"])
        self.assertFalse(any("Us" in str(c) for c in emit.call_args_list))
        self.assertEqual(len(session.history), 1)


if __name__ == "__main__":
    unittest.main()
