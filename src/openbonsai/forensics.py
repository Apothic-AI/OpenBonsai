"""Compare released Bonsai tensors with their full-precision Qwen ancestor."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from .formats import GGUFFile, SafeTensorFile, qwen3_gguf_to_hf
from .quantize import ternary_rtn


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _pearson(n: int, sx: float, sy: float, sxx: float, syy: float, sxy: float) -> float:
    numerator = n * sxy - sx * sy
    denominator = math.sqrt(max(n * sxx - sx * sx, 0.0) * max(n * syy - sy * sy, 0.0))
    return _safe_div(numerator, denominator)


def _sample(values: np.ndarray, target: int = 512) -> list[float]:
    flat = np.asarray(values).reshape(-1)
    if not len(flat):
        return []
    stride = max(1, len(flat) // target)
    return [float(x) for x in flat[::stride][:target]]


@dataclass
class Running:
    values: int = 0
    groups: int = 0
    base_ss: float = 0.0
    q1_error_ss: float = 0.0
    q2_error_ss: float = 0.0
    binary_rtn_error_ss: float = 0.0
    ternary_rtn_error_ss: float = 0.0
    q1_sign_matches: int = 0
    q2_nonzero: int = 0
    q2_zero: int = 0
    q2_invalid_plus_two: int = 0
    q2_base_sign_matches: int = 0
    q1_q2_sign_matches: int = 0
    q1_positive: int = 0
    base_positive: int = 0
    scale_n: int = 0
    q1_scale_sum: float = 0.0
    q2_scale_sum: float = 0.0
    base_absmean_sum: float = 0.0
    q1_scale_ss: float = 0.0
    q2_scale_ss: float = 0.0
    base_absmean_ss: float = 0.0
    q1_scale_base_cross: float = 0.0
    q2_scale_base_cross: float = 0.0
    q1_scale_ratios: list[float] = field(default_factory=list)
    q2_scale_ratios: list[float] = field(default_factory=list)

    def add(self, other: "Running") -> None:
        for key in (
            "values", "groups", "base_ss", "q1_error_ss", "q2_error_ss",
            "binary_rtn_error_ss", "ternary_rtn_error_ss", "q1_sign_matches",
            "q2_nonzero", "q2_zero", "q2_invalid_plus_two",
            "q2_base_sign_matches", "q1_q2_sign_matches", "q1_positive",
            "base_positive", "scale_n", "q1_scale_sum", "q2_scale_sum",
            "base_absmean_sum", "q1_scale_ss", "q2_scale_ss",
            "base_absmean_ss", "q1_scale_base_cross", "q2_scale_base_cross",
        ):
            setattr(self, key, getattr(self, key) + getattr(other, key))
        self.q1_scale_ratios.extend(other.q1_scale_ratios)
        self.q2_scale_ratios.extend(other.q2_scale_ratios)

    def summary(self) -> dict[str, Any]:
        root = math.sqrt(self.base_ss) if self.base_ss else float("nan")
        q1_ratios = np.asarray(self.q1_scale_ratios)
        q2_ratios = np.asarray(self.q2_scale_ratios)
        return {
            "values": self.values,
            "groups": self.groups,
            "q1_relative_weight_error": _safe_div(math.sqrt(self.q1_error_ss), root),
            "q2_relative_weight_error": _safe_div(math.sqrt(self.q2_error_ss), root),
            "binary_rtn_relative_weight_error": _safe_div(math.sqrt(self.binary_rtn_error_ss), root),
            "ternary_rtn_relative_weight_error": _safe_div(math.sqrt(self.ternary_rtn_error_ss), root),
            "q1_base_sign_agreement": _safe_div(self.q1_sign_matches, self.values),
            "q2_base_sign_agreement_nonzero": _safe_div(self.q2_base_sign_matches, self.q2_nonzero),
            "q1_q2_sign_agreement_nonzero": _safe_div(self.q1_q2_sign_matches, self.q2_nonzero),
            "q1_positive_fraction": _safe_div(self.q1_positive, self.values),
            "base_positive_fraction": _safe_div(self.base_positive, self.values),
            "q2_zero_fraction": _safe_div(self.q2_zero, self.values),
            "q2_invalid_plus_two": self.q2_invalid_plus_two,
            "q1_scale_vs_base_absmean_pearson": _pearson(
                self.scale_n, self.q1_scale_sum, self.base_absmean_sum,
                self.q1_scale_ss, self.base_absmean_ss, self.q1_scale_base_cross,
            ),
            "q2_scale_vs_base_absmean_pearson": _pearson(
                self.scale_n, self.q2_scale_sum, self.base_absmean_sum,
                self.q2_scale_ss, self.base_absmean_ss, self.q2_scale_base_cross,
            ),
            "q1_scale_over_base_absmean_median": float(np.median(q1_ratios)) if len(q1_ratios) else float("nan"),
            "q2_scale_over_base_absmean_median": float(np.median(q2_ratios)) if len(q2_ratios) else float("nan"),
        }


def _tensor_family(name: str) -> str:
    return re.sub(r"^blk\.\d+\.", "", name)


def _norm_metrics(
    name: str,
    hf_name: str,
    base: SafeTensorFile,
    binary: GGUFFile,
    ternary: GGUFFile,
) -> dict[str, Any]:
    reference = base.float_chunk(hf_name, 0, math.prod(base.tensors[hf_name].shape))
    one = np.asarray(binary.array(name), dtype=np.float32).reshape(-1)
    two = np.asarray(ternary.array(name), dtype=np.float32).reshape(-1)
    reference_norm = float(np.linalg.norm(reference))
    return {
        "name": name,
        "values": len(reference),
        "binary_relative_change": _safe_div(float(np.linalg.norm(one - reference)), reference_norm),
        "ternary_relative_change": _safe_div(float(np.linalg.norm(two - reference)), reference_norm),
        "binary_ternary_relative_difference": _safe_div(float(np.linalg.norm(one - two)), reference_norm),
        "binary_exact_base_fraction": float(np.mean(one == reference)),
        "ternary_exact_base_fraction": float(np.mean(two == reference)),
    }


def compare_tensor(
    name: str,
    hf_name: str,
    base: SafeTensorFile,
    binary: GGUFFile,
    ternary: GGUFFile,
    chunk_groups: int,
) -> Running:
    stats = Running()
    one_iter = binary.iter_q1_groups(name, chunk_groups)
    two_iter = ternary.iter_q2_groups(name, chunk_groups)
    base_iter = base.iter_float_groups(hf_name, 128, chunk_groups)
    for (q1_scales, q1_signs), (q2_scales, q2_codes), weights in zip(one_iter, two_iter, base_iter, strict=True):
        if weights.shape != q1_signs.shape or weights.shape != q2_codes.shape:
            raise ValueError(f"shape mismatch while comparing {name}")
        q1 = q1_signs.astype(np.float32) * q1_scales[:, None]
        q2 = q2_codes.astype(np.float32) * q2_scales[:, None]
        base_signs = np.where(weights >= 0, 1, -1).astype(np.int8)
        base_scales = np.mean(np.abs(weights), axis=1, dtype=np.float32)
        binary_rtn = base_signs.astype(np.float32) * base_scales[:, None]
        ternary_baseline, _, _ = ternary_rtn(weights.reshape(-1), 128, iterations=5)
        ternary_baseline = ternary_baseline.reshape(weights.shape)
        nonzero = q2_codes != 0

        stats.values += weights.size
        stats.groups += len(weights)
        stats.base_ss += float(np.sum(weights.astype(np.float64) ** 2))
        stats.q1_error_ss += float(np.sum((q1.astype(np.float64) - weights) ** 2))
        stats.q2_error_ss += float(np.sum((q2.astype(np.float64) - weights) ** 2))
        stats.binary_rtn_error_ss += float(np.sum((binary_rtn.astype(np.float64) - weights) ** 2))
        stats.ternary_rtn_error_ss += float(np.sum((ternary_baseline.astype(np.float64) - weights) ** 2))
        stats.q1_sign_matches += int(np.count_nonzero(q1_signs == base_signs))
        stats.q1_positive += int(np.count_nonzero(q1_signs > 0))
        stats.base_positive += int(np.count_nonzero(base_signs > 0))
        stats.q2_nonzero += int(np.count_nonzero(nonzero))
        stats.q2_zero += int(np.count_nonzero(q2_codes == 0))
        stats.q2_invalid_plus_two += int(np.count_nonzero(q2_codes == 2))
        stats.q2_base_sign_matches += int(np.count_nonzero(q2_codes[nonzero] == base_signs[nonzero]))
        stats.q1_q2_sign_matches += int(np.count_nonzero(q1_signs[nonzero] == q2_codes[nonzero]))

        n = len(base_scales)
        stats.scale_n += n
        stats.q1_scale_sum += float(np.sum(q1_scales))
        stats.q2_scale_sum += float(np.sum(q2_scales))
        stats.base_absmean_sum += float(np.sum(base_scales))
        stats.q1_scale_ss += float(np.sum(q1_scales.astype(np.float64) ** 2))
        stats.q2_scale_ss += float(np.sum(q2_scales.astype(np.float64) ** 2))
        stats.base_absmean_ss += float(np.sum(base_scales.astype(np.float64) ** 2))
        stats.q1_scale_base_cross += float(np.sum(q1_scales.astype(np.float64) * base_scales))
        stats.q2_scale_base_cross += float(np.sum(q2_scales.astype(np.float64) * base_scales))
        denominator = np.maximum(base_scales, np.finfo(np.float32).eps)
        stats.q1_scale_ratios.extend(_sample(q1_scales / denominator))
        stats.q2_scale_ratios.extend(_sample(q2_scales / denominator))
    return stats


def run_forensics(
    binary_path: Path,
    ternary_path: Path,
    base_path: Path,
    *,
    chunk_groups: int = 8_192,
) -> dict[str, Any]:
    binary = GGUFFile(binary_path)
    ternary = GGUFFile(ternary_path, q2_group_size=128)
    base = SafeTensorFile(base_path)
    global_stats = Running()
    family_stats: dict[str, Running] = defaultdict(Running)
    layer_stats: dict[int, Running] = defaultdict(Running)
    tensors: dict[str, Any] = {}
    norms: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}

    common = sorted(set(binary.tensors) & set(ternary.tensors))
    for name in common:
        try:
            hf_name = qwen3_gguf_to_hf(name)
        except KeyError:
            skipped[name] = "no Qwen3 name mapping"
            continue
        if hf_name not in base.tensors:
            skipped[name] = f"{hf_name} absent from base shard"
            continue
        one = binary.tensors[name]
        two = ternary.tensors[name]
        if one.tensor_type == 41 and two.tensor_type == 42:
            base_count = math.prod(base.tensors[hf_name].shape)
            if one.element_count != base_count or two.element_count != base_count:
                skipped[name] = (
                    f"element-count mismatch release={one.element_count}/{two.element_count}, "
                    f"base={base_count} (expected for vocabulary-trimmed embeddings)"
                )
                continue
            running = compare_tensor(name, hf_name, base, binary, ternary, chunk_groups)
            tensors[name] = running.summary()
            global_stats.add(running)
            family_stats[_tensor_family(name)].add(running)
            match = re.match(r"^blk\.(\d+)\.", name)
            if match:
                layer_stats[int(match.group(1))].add(running)
        elif one.tensor_type in {0, 1} and two.tensor_type in {0, 1}:
            if math.prod(base.tensors[hf_name].shape) != one.element_count:
                skipped[name] = "non-quantized tensor shape mismatch"
                continue
            norms.append(_norm_metrics(name, hf_name, base, binary, ternary))
        else:
            skipped[name] = f"unhandled type pair {one.tensor_type}/{two.tensor_type}"

    return {
        "schema_version": 2,
        "inputs": {
            "binary": str(binary_path),
            "ternary": str(ternary_path),
            "base": str(base_path),
            "binary_file_type": binary.metadata.get("general.file_type"),
            "ternary_file_type": ternary.metadata.get("general.file_type"),
        },
        "container_observations": {
            "binary_tensor_count": len(binary.tensors),
            "ternary_tensor_count": len(ternary.tensors),
            "binary_vocab_rows": binary.tensors.get("token_embd.weight").logical_shape[0],
            "ternary_vocab_rows": ternary.tensors.get("token_embd.weight").logical_shape[0],
            "base_vocab_rows": base.tensors.get("model.embed_tokens.weight").shape[0],
        },
        "global": global_stats.summary(),
        "by_tensor_family": {key: value.summary() for key, value in sorted(family_stats.items())},
        "by_layer": {str(key): value.summary() for key, value in sorted(layer_stats.items())},
        "tensors": tensors,
        "non_quantized_tensors": norms,
        "skipped": skipped,
    }


def render_markdown(result: dict[str, Any]) -> str:
    g = result["global"]
    c = result["container_observations"]
    lines = [
        "# OpenBonsai 1.7B weight-forensics report",
        "",
        "This report compares PrismML's released binary and ternary 1.7B GGUF tensors "
        "with the BF16 Qwen3-1.7B ancestor. It is generated by `openbonsai-forensics`.",
        "",
        "## Main result",
        "",
        f"Across {g['values']:,} comparable matrix weights, binary Bonsai preserves only "
        f"{g['q1_base_sign_agreement']:.1%} of the base signs. Ternary Bonsai preserves "
        f"{g['q2_base_sign_agreement_nonzero']:.1%} of base signs among nonzero codes and "
        f"uses zero for {g['q2_zero_fraction']:.1%} of weights. The binary and ternary "
        f"releases agree on {g['q1_q2_sign_agreement_nonzero']:.1%} of nonzero signs.",
        "",
        f"Binary Bonsai's relative Euclidean weight error is {g['q1_relative_weight_error']:.3f}, "
        f"versus {g['binary_rtn_relative_weight_error']:.3f} for naïve groupwise binary RTN. "
        f"Ternary Bonsai's is {g['q2_relative_weight_error']:.3f}, versus "
        f"{g['ternary_rtn_relative_weight_error']:.3f} for a naïve alternating ternary MSE baseline. "
        "The released weights are therefore *farther* from Qwen in Euclidean weight space than "
        "simple rounding, despite retaining much more functional capability than ordinary extreme "
        "PTQ. This is strong evidence for task/function-space optimization after discretization.",
        "",
        "## Aggregate measurements",
        "",
        "| Measurement | Value |",
        "|---|---:|",
        f"| Comparable matrix weights | {g['values']:,} |",
        f"| Binary/base sign agreement | {g['q1_base_sign_agreement']:.2%} |",
        f"| Ternary/base sign agreement (nonzero) | {g['q2_base_sign_agreement_nonzero']:.2%} |",
        f"| Binary/ternary sign agreement (nonzero) | {g['q1_q2_sign_agreement_nonzero']:.2%} |",
        f"| Ternary zero fraction | {g['q2_zero_fraction']:.2%} |",
        f"| Invalid ternary `+2` codes | {g['q2_invalid_plus_two']:,} |",
        f"| Binary actual / base mean-absolute scale (sampled median) | {g['q1_scale_over_base_absmean_median']:.3f} |",
        f"| Ternary actual / base mean-absolute scale (sampled median) | {g['q2_scale_over_base_absmean_median']:.3f} |",
        f"| Binary scale/base mean-absolute correlation | {g['q1_scale_vs_base_absmean_pearson']:.3f} |",
        f"| Ternary scale/base mean-absolute correlation | {g['q2_scale_vs_base_absmean_pearson']:.3f} |",
        "",
        "## Tensor-family detail",
        "",
        "| Family | Weights | Q1/base signs | Q2/base signs | Q2 zero | Q1 error | Q1 RTN error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, values in result["by_tensor_family"].items():
        lines.append(
            f"| `{family}` | {values['values']:,} | {values['q1_base_sign_agreement']:.1%} | "
            f"{values['q2_base_sign_agreement_nonzero']:.1%} | {values['q2_zero_fraction']:.1%} | "
            f"{values['q1_relative_weight_error']:.3f} | {values['binary_rtn_relative_weight_error']:.3f} |"
        )
    lines += [
        "",
        "## Depth profile",
        "",
        "| Layer | Weights | Q1/base signs | Q2/base signs | Q1/Q2 signs | Q2 zero | Q1 error | Q2 error |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, values in result["by_layer"].items():
        lines.append(
            f"| {layer} | {values['values']:,} | {values['q1_base_sign_agreement']:.1%} | "
            f"{values['q2_base_sign_agreement_nonzero']:.1%} | "
            f"{values['q1_q2_sign_agreement_nonzero']:.1%} | {values['q2_zero_fraction']:.1%} | "
            f"{values['q1_relative_weight_error']:.3f} | {values['q2_relative_weight_error']:.3f} |"
        )
    lines += [
        "",
        "## Non-quantized parameter movement",
        "",
        "| Tensor | Binary relative change | Ternary relative change | Binary/ternary difference |",
        "|---|---:|---:|---:|",
    ]
    for item in result["non_quantized_tensors"]:
        lines.append(
            f"| `{item['name']}` | {item['binary_relative_change']:.4f} | "
            f"{item['ternary_relative_change']:.4f} | {item['binary_ternary_relative_difference']:.4f} |"
        )
    lines += [
        "",
        "## Container finding",
        "",
        f"The released embedding has {c['binary_vocab_rows']:,} rows, while the original Qwen "
        f"checkpoint has {c['base_vocab_rows']:,}: a reduction of "
        f"{c['base_vocab_rows'] - c['binary_vocab_rows']:,} vocabulary rows. The embedding is "
        "excluded from base-weight aggregates because the row mapping has not yet been proven.",
        "",
        "## Interpretation boundaries",
        "",
        "These measurements establish that the release is a learned/transformed discrete model; "
        "they do not uniquely identify the loss, optimizer, training corpus, or curriculum. Weight "
        "movement can be produced by several families of QAT and distillation. The ranked method "
        "hypotheses and discriminating experiments are in `docs/02-reverse-engineering-hypotheses.md`.",
        "",
    ]
    return "\n".join(lines)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="Bonsai Q1_0 g128 GGUF")
    parser.add_argument("--ternary", required=True, type=Path, help="Ternary Bonsai Q2_0 g128 GGUF")
    parser.add_argument("--base", required=True, type=Path, help="Qwen3 BF16 safetensors shard containing the transformer")
    parser.add_argument("--json", type=Path, help="write machine-readable results")
    parser.add_argument("--markdown", type=Path, help="write a human-readable report")
    parser.add_argument("--chunk-groups", type=int, default=8_192)
    args = parser.parse_args(argv)
    result = run_forensics(args.binary, args.ternary, args.base, chunk_groups=args.chunk_groups)
    payload = json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload)
    else:
        print(payload)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(result))


if __name__ == "__main__":
    main()
