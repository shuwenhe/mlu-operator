import unittest
from contracts import layout, spans


class Contracts(unittest.TestCase):
    def test_cache(self):
        self.assertEqual(layout(2, 16), (32, 49152, 81920))
        for args in ((0, 16), (2, 32)):
            with self.assertRaises(ValueError):
                layout(*args)

    def test_ownership(self):
        self.assertEqual(spans("q_proj.weight", 1), (0, 1536, 1536))
        self.assertEqual(spans("o_proj.weight", 1), (1, 1024, 1024))
        self.assertEqual(spans("kv_b_proj.weight", 0), (0, 0, 2048))
        self.assertEqual(spans("kv_a_proj_with_mqa.weight", 0), (None, 0, 576))

if __name__ == "__main__":
    unittest.main()
