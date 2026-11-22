"""Host regression for IEEE FP32-to-BF16 rounding and lane continuation."""
import struct
import unittest


def f32_bits(number):
    return struct.unpack("<I", struct.pack("<f", number))[0]


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bf16_bits(bits):
    return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16


def accumulate(values, initial=0.0):
    for value in values:
        initial = f32(f32_bits(initial + value))
    return initial


class Rounding(unittest.TestCase):
    def test_historical_midpoint(self):
        actual, expected = map(f32_bits, (0.0543212890625, 0.0543212965130806))
        self.assertEqual(expected, actual + 2)
        self.assertEqual((bf16_bits(actual), bf16_bits(expected)), (0x3d5e, 0x3d5f))

    def test_adjacent_midpoints(self):
        for high in (0x0000, 0x0001, 0x007f, 0x3d5e, 0x3f81, 0x6a40,
                     0x8000, 0x8001, 0x807f, 0xbd5e, 0xbf81, 0xea40):
            for offset, expected in ((-1, high), (0, high+(high&1)), (1, high+1)):
                source = (high << 16) + 0x8000 + offset
                result = bf16_bits(source)
                print(f"fp32=0x{source:08x} bf16=0x{result:04x}")
                self.assertEqual(result, expected)

    def test_cancellation_and_partitions(self):
        values = [2.0**24, 1, -2.0**24, 2.0**-24]
        full = accumulate(values)
        for split in range(1, len(values)):
            carried = accumulate(values[split:], accumulate(values[:split]))
            reset = accumulate((accumulate(values[:split]), accumulate(values[split:])))
            print(f"split={split} full={f32_bits(full):08x} carried={f32_bits(carried):08x} "
                  f"reset={f32_bits(reset):08x}")
            self.assertEqual(f32_bits(full), f32_bits(carried))
        self.assertNotEqual(full, accumulate((accumulate(values[:1]), accumulate(values[1:]))))


if __name__ == "__main__":
    unittest.main()
