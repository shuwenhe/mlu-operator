import unittest

from ...phase13_3_2.mapping import RULES
from .contracts import HEADS, TP, expert_owner, layout, spans


class TP8Contracts(unittest.TestCase):
    def test_checkpoint_partition(self):
        for name, (shape, axis) in RULES.items():
            width = shape[axis] // TP if axis is not None else shape[0]
            self.assertEqual(
                [spans(name, rank) for rank in range(TP)],
                [(axis, rank * width if axis is not None else 0, width) for rank in range(TP)],
            )
        self.assertEqual(HEADS * TP, 16)
        for rank in (-1, TP):
            with self.assertRaises(ValueError):
                spans("q_proj.weight", rank)

    def test_cache_layout(self):
        self.assertEqual(layout(3, 16), (48, 18432, 30720))
        for blocks, page in ((0, 16), (3, 32)):
            with self.assertRaises(ValueError):
                layout(blocks, page)

    def test_expert_ownership(self):
        self.assertEqual([expert_owner(i) for i in range(64)], [i for i in range(8) for _ in range(8)])
        for expert in (-1, 64):
            with self.assertRaises(ValueError):
                expert_owner(expert)
