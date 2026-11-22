import unittest

from shards import column_spans, row_span


class ShardTests(unittest.TestCase):
    def test_packed_columns(self):
        self.assertEqual(column_spans((16, 24), 2, 4), ((8, 4), (28, 6)))

    def test_row(self):
        self.assertEqual(row_span(512, 7, 8), (448, 64))

    def test_invalid_sections_and_rank(self):
        for call in (
            lambda: column_spans((12, 4), 0, 8),
            lambda: row_span(513, 0, 8),
            lambda: row_span(512, 8, 8),
        ):
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
