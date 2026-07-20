import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from openbonsai.formats import GGUFFile, SafeTensorFile, encode_q1_g128, encode_q2_g128


def gguf_string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


class FormatTests(unittest.TestCase):
    def test_q1_decode_and_scalar_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 2, 2))
            for key, value in (("general.alignment", 32), ("general.file_type", 40)):
                header += gguf_string(key) + struct.pack("<II", 4, value)
            header += gguf_string("q") + struct.pack("<I", 1) + struct.pack("<QIQ", 128, 41, 0)
            header += gguf_string("norm") + struct.pack("<I", 1) + struct.pack("<QIQ", 2, 0, 32)
            header += b"\0" * ((-len(header)) % 32)
            scale = np.asarray([0.5], dtype="<f2").tobytes()
            signs = bytes([0b01010101] * 16)
            payload = scale + signs + b"\0" * 14 + np.asarray([2.0, 3.0], dtype="<f4").tobytes()
            path.write_bytes(header + payload)

            reader = GGUFFile(path)
            [(scales, codes)] = list(reader.iter_q1_groups("q"))
            self.assertAlmostEqual(float(scales[0]), 0.5)
            np.testing.assert_array_equal(codes[0, :8], [1, -1, 1, -1, 1, -1, 1, -1])
            np.testing.assert_allclose(reader.array("norm"), [2.0, 3.0])

    def test_legacy_q2_g128_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny-q2.gguf"
            header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
            header += gguf_string("general.alignment") + struct.pack("<II", 4, 32)
            header += gguf_string("q") + struct.pack("<I", 1) + struct.pack("<QIQ", 128, 42, 0)
            header += b"\0" * ((-len(header)) % 32)
            # Four two-bit states in little-endian slot order: 0,1,2,3.
            # The release maps them to -1,0,+1 and an unused +2 sentinel.
            payload = np.asarray([0.25], dtype="<f2").tobytes() + bytes([0b11100100] * 32)
            path.write_bytes(header + payload)

            reader = GGUFFile(path, q2_group_size=128)
            [(scales, codes)] = list(reader.iter_q2_groups("q"))
            self.assertAlmostEqual(float(scales[0]), 0.25)
            np.testing.assert_array_equal(codes[0, :8], [-1, 0, 1, 2, -1, 0, 1, 2])

    def test_payload_encoder_validation_and_sizes(self):
        q1 = np.tile(np.asarray([-1, 1], dtype=np.int8), 64).reshape(1, 128)
        q2 = np.tile(np.asarray([-1, 0, 1, 0], dtype=np.int8), 32).reshape(1, 128)
        q1_payload = encode_q1_g128(q1, [0.5])
        q2_payload = encode_q2_g128(q2, [0.25])
        self.assertEqual(len(q1_payload), 18)
        self.assertEqual(len(q2_payload), 34)
        self.assertEqual(q1_payload[2], 0b10101010)
        self.assertEqual(q2_payload[2], 0b01100100)
        with self.assertRaises(ValueError):
            encode_q1_g128(np.zeros((1, 128), dtype=np.int8), [1.0])
        with self.assertRaises(ValueError):
            bad = q2.copy()
            bad[0, 0] = 2
            encode_q2_g128(bad, [1.0])

    def test_safetensors_bf16(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.safetensors"
            values = np.asarray([1.0, -2.5, 0.25], dtype=np.float32)
            bf16 = (values.view(np.uint32) >> 16).astype("<u2").tobytes()
            header = json.dumps({
                "__metadata__": {"format": "pt"},
                "x": {"dtype": "BF16", "shape": [3], "data_offsets": [0, len(bf16)]},
            }, separators=(",", ":")).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + bf16)
            reader = SafeTensorFile(path)
            np.testing.assert_allclose(reader.float_chunk("x", 0, 3), values)


if __name__ == "__main__":
    unittest.main()
