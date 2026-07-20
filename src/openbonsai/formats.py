"""Minimal, dependency-light readers for the artifacts used by OpenBonsai.

This is intentionally not a general GGUF implementation.  It parses the GGUF
container, exposes tensor byte ranges, and decodes PrismML's Q1_0 g128 and
legacy Q2_0 g128 layouts.  It also memory-maps safetensors, including BF16,
without requiring PyTorch or safetensors.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import struct
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np


GGUF_VALUE_TYPES: dict[int, str] = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}


def encode_q1_g128(codes: np.ndarray, scales: np.ndarray) -> bytes:
    """Encode conformance blocks for Q1_0 group-128 payloads.

    This writes tensor payload blocks, not a complete GGUF container. It is
    useful for hard-export round trips and kernel conformance vectors.
    """
    z = np.asarray(codes)
    s = np.asarray(scales, dtype="<f2").reshape(-1)
    if z.ndim != 2 or z.shape[1] != 128 or len(z) != len(s):
        raise ValueError("codes must be [groups,128] with one scale per group")
    if not np.all((z == -1) | (z == 1)):
        raise ValueError("Q1 codes must belong to {-1,+1}")
    bits = (z > 0).astype(np.uint8)
    packed = np.packbits(bits, axis=1, bitorder="little")
    blocks = np.empty((len(z), 18), dtype=np.uint8)
    blocks[:, :2] = s.view(np.uint8).reshape(-1, 2)
    blocks[:, 2:] = packed
    return blocks.tobytes()


def encode_q2_g128(codes: np.ndarray, scales: np.ndarray) -> bytes:
    """Encode PrismML-compatible legacy Q2_0 group-128 payload blocks."""
    z = np.asarray(codes)
    s = np.asarray(scales, dtype="<f2").reshape(-1)
    if z.ndim != 2 or z.shape[1] != 128 or len(z) != len(s):
        raise ValueError("codes must be [groups,128] with one scale per group")
    if not np.all((z == -1) | (z == 0) | (z == 1)):
        raise ValueError("Q2 ternary codes must belong to {-1,0,+1}")
    unsigned = (z.astype(np.uint8) + 1).reshape(len(z), 32, 4)
    packed = (
        unsigned[:, :, 0]
        | (unsigned[:, :, 1] << 2)
        | (unsigned[:, :, 2] << 4)
        | (unsigned[:, :, 3] << 6)
    )
    blocks = np.empty((len(z), 34), dtype=np.uint8)
    blocks[:, :2] = s.view(np.uint8).reshape(-1, 2)
    blocks[:, 2:] = packed
    return blocks.tobytes()


def _read_exact(f: BinaryIO, n: int) -> bytes:
    value = f.read(n)
    if len(value) != n:
        raise EOFError(f"expected {n} bytes, got {len(value)}")
    return value


def _unpack(f: BinaryIO, fmt: str) -> Any:
    return struct.unpack(fmt, _read_exact(f, struct.calcsize(fmt)))[0]


def _read_string(f: BinaryIO) -> str:
    n = _unpack(f, "<Q")
    return _read_exact(f, n).decode("utf-8")


def _read_value(f: BinaryIO, value_type: int) -> Any:
    if value_type in GGUF_VALUE_TYPES:
        return _unpack(f, GGUF_VALUE_TYPES[value_type])
    if value_type == 8:
        return _read_string(f)
    if value_type == 9:
        element_type = _unpack(f, "<I")
        n = _unpack(f, "<Q")
        return [_read_value(f, element_type) for _ in range(n)]
    raise ValueError(f"unsupported GGUF metadata type {value_type}")


@dataclass(frozen=True)
class GGUFTensor:
    name: str
    dimensions: tuple[int, ...]
    tensor_type: int
    relative_offset: int
    absolute_offset: int
    byte_length: int

    @property
    def logical_shape(self) -> tuple[int, ...]:
        """Return conventional row-major shape (GGUF dimensions are reversed)."""
        return tuple(reversed(self.dimensions))

    @property
    def element_count(self) -> int:
        return math.prod(self.dimensions)


class GGUFFile:
    """Read GGUF metadata and memory-map tensor payloads."""

    def __init__(self, path: str | Path, *, q2_group_size: int = 128):
        self.path = Path(path)
        self.q2_group_size = q2_group_size
        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, GGUFTensor] = {}
        self.version = 0
        self.data_offset = 0
        self._parse()

    def _parse(self) -> None:
        with self.path.open("rb") as f:
            if _read_exact(f, 4) != b"GGUF":
                raise ValueError(f"{self.path} is not a GGUF file")
            self.version = _unpack(f, "<I")
            if self.version not in {2, 3}:
                raise ValueError(f"unsupported GGUF version {self.version}")
            tensor_count = _unpack(f, "<Q")
            kv_count = _unpack(f, "<Q")
            for _ in range(kv_count):
                key = _read_string(f)
                value_type = _unpack(f, "<I")
                self.metadata[key] = _read_value(f, value_type)

            infos: list[tuple[str, tuple[int, ...], int, int]] = []
            for _ in range(tensor_count):
                name = _read_string(f)
                n_dims = _unpack(f, "<I")
                dims = tuple(_unpack(f, "<Q") for _ in range(n_dims))
                tensor_type = _unpack(f, "<I")
                relative_offset = _unpack(f, "<Q")
                infos.append((name, dims, tensor_type, relative_offset))

            alignment = int(self.metadata.get("general.alignment", 32))
            self.data_offset = ((f.tell() + alignment - 1) // alignment) * alignment

        file_size = self.path.stat().st_size
        for index, (name, dims, tensor_type, rel) in enumerate(infos):
            next_rel = infos[index + 1][3] if index + 1 < len(infos) else file_size - self.data_offset
            expected = self._expected_bytes(dims, tensor_type)
            available = next_rel - rel
            if expected > available:
                raise ValueError(
                    f"tensor {name} needs {expected} bytes but only {available} are available"
                )
            self.tensors[name] = GGUFTensor(
                name=name,
                dimensions=dims,
                tensor_type=tensor_type,
                relative_offset=rel,
                absolute_offset=self.data_offset + rel,
                byte_length=expected,
            )

    def _expected_bytes(self, dims: tuple[int, ...], tensor_type: int) -> int:
        n = math.prod(dims)
        scalar_sizes = {0: 4, 1: 2, 16: 1, 24: 8}
        if tensor_type in scalar_sizes:
            return n * scalar_sizes[tensor_type]
        if tensor_type == 41:  # Q1_0 g128: fp16 scale + 128 sign bits
            if n % 128:
                raise ValueError("Q1_0 tensor is not divisible by 128")
            return (n // 128) * 18
        if tensor_type == 42:  # Q2_0; Prism release is legacy g128
            group = self.q2_group_size
            if n % group:
                raise ValueError(f"Q2_0 tensor is not divisible by group size {group}")
            return (n // group) * (2 + group // 4)
        raise ValueError(f"unsupported GGML tensor type {tensor_type}")

    def raw(self, name: str) -> np.memmap:
        tensor = self.tensors[name]
        return np.memmap(
            self.path,
            dtype=np.uint8,
            mode="r",
            offset=tensor.absolute_offset,
            shape=(tensor.byte_length,),
        )

    def array(self, name: str) -> np.ndarray:
        tensor = self.tensors[name]
        shape = tensor.logical_shape
        dtype = {0: "<f4", 1: "<f2", 16: "<i1", 24: "<i8"}.get(tensor.tensor_type)
        if dtype is None:
            raise TypeError(f"tensor {name} is quantized; use iter_q1_groups/iter_q2_groups")
        return np.memmap(
            self.path,
            dtype=dtype,
            mode="r",
            offset=tensor.absolute_offset,
            shape=shape,
        )

    def iter_q1_groups(self, name: str, chunk_groups: int = 65_536) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        tensor = self.tensors[name]
        if tensor.tensor_type != 41:
            raise TypeError(f"{name} is not Q1_0")
        raw = self.raw(name).reshape(-1, 18)
        for start in range(0, len(raw), chunk_groups):
            block = np.asarray(raw[start : start + chunk_groups])
            scales = block[:, :2].copy().view("<f2").astype(np.float32).reshape(-1)
            signs = np.unpackbits(block[:, 2:], axis=1, bitorder="little").astype(np.int8)
            yield scales, signs * 2 - 1

    def iter_q2_groups(self, name: str, chunk_groups: int = 65_536) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        tensor = self.tensors[name]
        if tensor.tensor_type != 42:
            raise TypeError(f"{name} is not Q2_0")
        group = self.q2_group_size
        block_bytes = 2 + group // 4
        raw = self.raw(name).reshape(-1, block_bytes)
        shifts = (np.arange(4, dtype=np.uint8) * 2)[None, None, :]
        for start in range(0, len(raw), chunk_groups):
            block = np.asarray(raw[start : start + chunk_groups])
            scales = block[:, :2].copy().view("<f2").astype(np.float32).reshape(-1)
            packed = block[:, 2:, None]
            codes = ((packed >> shifts) & 3).reshape(len(block), group).astype(np.int8) - 1
            yield scales, codes


@dataclass(frozen=True)
class SafeTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    absolute_offset: int
    byte_length: int


class SafeTensorFile:
    """Memory-map a single safetensors shard."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, SafeTensor] = {}
        with self.path.open("rb") as f:
            header_length = _unpack(f, "<Q")
            header = json.loads(_read_exact(f, header_length))
        data_offset = 8 + header_length
        self.metadata = header.pop("__metadata__", {})
        sizes = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1}
        for name, info in header.items():
            start, end = info["data_offsets"]
            dtype = info["dtype"]
            expected = math.prod(info["shape"]) * sizes[dtype]
            if end - start != expected:
                raise ValueError(f"invalid byte length for {name}")
            self.tensors[name] = SafeTensor(
                name=name,
                dtype=dtype,
                shape=tuple(info["shape"]),
                absolute_offset=data_offset + start,
                byte_length=end - start,
            )

    def raw(self, name: str) -> np.memmap:
        tensor = self.tensors[name]
        dtype = {"BF16": "<u2", "F16": "<f2", "F32": "<f4", "I8": "<i1", "U8": "<u1"}[tensor.dtype]
        return np.memmap(
            self.path,
            dtype=dtype,
            mode="r",
            offset=tensor.absolute_offset,
            shape=tensor.shape,
        )

    def float_chunk(self, name: str, start: int, stop: int) -> np.ndarray:
        tensor = self.tensors[name]
        flat = self.raw(name).reshape(-1)[start:stop]
        if tensor.dtype == "BF16":
            bits = np.asarray(flat, dtype=np.uint16).astype(np.uint32) << 16
            return bits.view(np.float32)
        return np.asarray(flat, dtype=np.float32)

    def iter_float_groups(self, name: str, group_size: int = 128, chunk_groups: int = 65_536) -> Iterator[np.ndarray]:
        tensor = self.tensors[name]
        n = math.prod(tensor.shape)
        if n % group_size:
            raise ValueError(f"{name} is not divisible by group size {group_size}")
        group_count = n // group_size
        for first_group in range(0, group_count, chunk_groups):
            last_group = min(first_group + chunk_groups, group_count)
            yield self.float_chunk(
                name,
                first_group * group_size,
                last_group * group_size,
            ).reshape(-1, group_size)


def qwen3_gguf_to_hf(name: str) -> str:
    """Map the Qwen3 tensor names used by GGUF back to Hugging Face names."""
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output_norm.weight":
        return "model.norm.weight"
    if not name.startswith("blk."):
        raise KeyError(name)
    _, layer, suffix = name.split(".", 2)
    mapping = {
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
    }
    return f"model.layers.{layer}.{mapping[suffix]}"
