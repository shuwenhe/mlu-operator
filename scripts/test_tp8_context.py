"""CPU regression for native TP8 position and page validation."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import unittest

root = Path(__file__).resolve().parents[1] / "integration/vllm_native_mlu"
name = "integration.vllm_native_mlu.phase12_3.scheduled_attention"
spec = importlib.util.spec_from_file_location(name, root / "phase12_3/scheduled_attention.py")
schedule = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {name: schedule, "torch": SimpleNamespace(int32="int32"),
                              "integration.vllm_native_mlu.phase12.paged_kv": SimpleNamespace(PagedKV=object)}):
    spec.loader.exec_module(schedule)


class Tensor(list):
    device = "cpu"
    dtype = "int32"

    @property
    def ndim(self):
        return 2 if self and isinstance(self[0], list) else 1

    @property
    def shape(self):
        return (len(self), len(self[0])) if self.ndim == 2 else (len(self),)

    def __getitem__(self, index):
        value = super().__getitem__(index)
        return Tensor(value) if isinstance(index, slice) else value

    def tolist(self):
        return list(self)

    def numel(self):
        return len(self)


def span(position, limit, query=1, wrong_slot=False):
    length = position + 1
    pages = list(reversed(range(11)))
    positions = range(length - query, length)
    slots = [pages[p // 16] * 16 + p % 16 for p in positions]
    slots[-1] += wrong_slot
    metadata = SimpleNamespace(query_start_loc=Tensor([0, query]), seq_lens=Tensor([length]),
                               block_table_tensor=Tensor([pages]), slot_mapping=Tensor(slots))
    return schedule.validate_batch(metadata, [0], Tensor(positions), 16, 16, max_length=limit)[0]


class ContextTests(unittest.TestCase):
    def test_prefill_and_page_crossings(self):
        self.assertEqual(span(127, 160, query=128).blocks, 8)
        for position, pages in ((127, 8), (128, 9), (159, 10), (160, 11)):
            with self.subTest(position=position):
                self.assertEqual(span(position, 161).blocks, pages)

    def test_limit_is_exclusive_and_wrong_slot_rejected(self):
        with self.assertRaisesRegex(ValueError, "context length"):
            span(160, 160)
        with self.assertRaisesRegex(ValueError, "context length"):
            span(128, 128)
        with self.assertRaisesRegex(ValueError, "physical slots"):
            span(128, 160, wrong_slot=True)

    def test_prefill_chunk_limit_remains_128(self):
        with self.assertRaisesRegex(ValueError, "context length"):
            span(128, 160, query=129)


if __name__ == "__main__":
    unittest.main()
