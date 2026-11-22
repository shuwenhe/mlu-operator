import unittest
from mapping import shard


class MappingTest(unittest.TestCase):
    def test_eighth_rank_offsets(self):
        self.assertEqual(shard("q_proj.weight", 7, 8), (0, 2688, 384))
        self.assertEqual(shard("kv_b_proj.weight", 7, 8), (0, 3584, 512))
        self.assertEqual(shard("o_proj.weight", 7, 8), (1, 1792, 256))

    def test_replicas(self):
        self.assertEqual(shard("kv_a_proj_with_mqa.weight", 5, 8), (None, 0, 576))
        self.assertEqual(shard("kv_a_layernorm.weight", 5, 8), (None, 0, 512))

    def test_invalid_group(self):
        with self.assertRaises(ValueError):
            shard("q_proj.weight", 2, 2)
        with self.assertRaises(ValueError):
            shard("q_proj.weight", 0, 3)


if __name__ == "__main__":
    unittest.main()
