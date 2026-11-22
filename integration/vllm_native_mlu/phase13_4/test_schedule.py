import unittest
from schedule import Pages, digest


class ScheduleTest(unittest.TestCase):
    def test_boundary_and_reuse(self):
        pages = Pages(3)
        first = pages.extend("first", 16)
        self.assertEqual(first[-1], 15)
        self.assertEqual(pages.extend("first", 1), [16])
        other = pages.extend("other", 1)
        self.assertEqual(other, [32])
        pages.release("first")
        self.assertEqual(pages.extend("reuse", 1), [0])

    def test_cannot_mutate_on_capacity_failure(self):
        pages = Pages(1)
        with self.assertRaises(ValueError):
            pages.extend("overflow", 17)
        self.assertEqual(pages.extend("valid", 1), [0])

    def test_digest_respects_order(self):
        self.assertNotEqual(digest([1, 2]), digest([2, 1]))


if __name__ == "__main__":
    unittest.main()
