"""PyTorch QAT primitives for the OpenBonsai training hypotheses.

PyTorch is an optional dependency so the forensic tools remain lightweight.
The modules keep full-precision shadow weights during training and execute a
progressive group-wise binary or ternary representation in the forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator, Literal

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only without optional deps
    raise ImportError("install OpenBonsai with the `training` extra to use QAT") from exc


Mode = Literal["binary", "ternary"]


@dataclass
class QATConfig:
    mode: Mode = "ternary"
    group_size: int = 128
    initial_beta: float = 1.0
    final_beta: float = 20.0
    activation_noise_std: float = 0.0
    quantize_embeddings: bool = True
    quantize_lm_head: bool = True


def _reshape_groups(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Size]:
    if weight.shape[-1] % group_size:
        raise ValueError(f"last dimension {weight.shape[-1]} is not divisible by {group_size}")
    original = weight.shape
    return weight.reshape(-1, group_size), original


class ProgressiveDiscreteWeight(nn.Module):
    """Progressively moves a shadow weight to a hard binary/ternary forward."""

    def __init__(self, rows: int, groups_per_row: int, config: QATConfig):
        super().__init__()
        self.config = config
        self.progress = 0.0
        self.beta = config.initial_beta
        # This second scale is deliberately separate from the analytic MSE scale.
        # It tests the dual-scale compensation hypothesis without inference overhead.
        self.log_compensation = nn.Parameter(torch.zeros(rows, groups_per_row, 1))

    def set_progress(self, progress: float) -> None:
        self.progress = min(max(float(progress), 0.0), 1.0)
        ratio = self.config.final_beta / max(self.config.initial_beta, 1e-8)
        self.beta = self.config.initial_beta * ratio ** self.progress

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        grouped, original = _reshape_groups(weight, self.config.group_size)
        analytic = grouped.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        compensation = self.log_compensation.reshape(-1, 1).exp()
        scale = analytic * compensation
        normalized = grouped / scale
        if self.config.mode == "binary":
            hard_codes = torch.where(normalized >= 0, 1.0, -1.0)
            soft_codes = torch.tanh(self.beta * normalized)
        else:
            hard_codes = torch.where(
                normalized > 0.5,
                1.0,
                torch.where(normalized < -0.5, -1.0, 0.0),
            )
            # Difference of two sigmoids: smooth {-1,0,+1} thresholds.
            soft_codes = torch.sigmoid(self.beta * (normalized - 0.5)) + torch.sigmoid(
                self.beta * (normalized + 0.5)
            ) - 1.0
        # Hard forward, smooth backward at progress=1. Earlier stages blend with W.
        discrete_ste = hard_codes.detach() - soft_codes.detach() + soft_codes
        transformed = discrete_ste * scale
        return ((1.0 - self.progress) * grouped + self.progress * transformed).reshape(original)


class QuantizedLinear(nn.Module):
    def __init__(self, source: nn.Linear, config: QATConfig):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight = source.weight
        self.bias = source.bias
        self.discretizer = ProgressiveDiscreteWeight(
            source.out_features,
            source.in_features // config.group_size,
            config,
        )
        self.noise_std = config.activation_noise_std

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.discretizer(self.weight)
        output = F.linear(inputs, weight, self.bias)
        if self.training and self.noise_std:
            output = output + torch.randn_like(output) * self.noise_std * output.detach().std().clamp_min(1e-8)
        return output


class QuantizedEmbedding(nn.Module):
    def __init__(self, source: nn.Embedding, config: QATConfig):
        super().__init__()
        self.num_embeddings = source.num_embeddings
        self.embedding_dim = source.embedding_dim
        self.padding_idx = source.padding_idx
        self.weight = source.weight
        self.discretizer = ProgressiveDiscreteWeight(
            source.num_embeddings,
            source.embedding_dim // config.group_size,
            config,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.embedding(inputs, self.discretizer(self.weight), padding_idx=self.padding_idx)


def convert_model(model: nn.Module, config: QATConfig) -> nn.Module:
    """Replace Linear/Embedding children while preserving shadow parameters.

    Shared modules are replaced once and then reused, which matters for tied
    embedding/lm-head weights. Call the model's `tie_weights()` again after
    conversion when its Transformers implementation supplies that method.
    """
    replacements: dict[int, nn.Module] = {}

    def walk(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            replacement: nn.Module | None = None
            if id(child) in replacements:
                replacement = replacements[id(child)]
            elif isinstance(child, nn.Linear):
                is_head = qualified.endswith("lm_head")
                if config.quantize_lm_head or not is_head:
                    replacement = QuantizedLinear(child, config)
            elif isinstance(child, nn.Embedding) and config.quantize_embeddings:
                replacement = QuantizedEmbedding(child, config)
            if replacement is not None:
                replacements[id(child)] = replacement
                setattr(parent, name, replacement)
            else:
                walk(child, qualified)

    walk(model)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


def discretizers(model: nn.Module) -> Iterator[ProgressiveDiscreteWeight]:
    for module in model.modules():
        if isinstance(module, ProgressiveDiscreteWeight):
            yield module


def set_progress(model: nn.Module, progress: float) -> None:
    for module in discretizers(model):
        module.set_progress(progress)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 2.0,
    ce_weight: float = 0.25,
    kl_weight: float = 0.75,
) -> tuple[torch.Tensor, dict[str, float]]:
    vocab = student_logits.shape[-1]
    ce = F.cross_entropy(
        student_logits[:, :-1].reshape(-1, vocab),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    t = temperature
    # Position i predicts label i+1. Padding and ignored labels must not enter
    # either CE or teacher KL, or length/padding changes become an objective.
    student_logp = F.log_softmax(student_logits[:, :-1] / t, dim=-1)
    teacher_p = F.softmax(teacher_logits[:, :-1].detach() / t, dim=-1)
    token_kl = F.kl_div(student_logp, teacher_p, reduction="none").sum(dim=-1)
    valid = labels[:, 1:] != -100
    kl = (token_kl * valid).sum() / valid.sum().clamp_min(1)
    kl = kl * t * t
    loss = ce_weight * ce + kl_weight * kl
    return loss, {"ce": float(ce.detach()), "kl": float(kl.detach())}


def hidden_alignment_loss(
    student_states: tuple[torch.Tensor, ...],
    teacher_states: tuple[torch.Tensor, ...],
    taps: int = 5,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cosine hidden-state matching at evenly spaced depth taps."""
    count = min(len(student_states), len(teacher_states))
    indices = torch.linspace(0, count - 1, taps).round().long().unique().tolist()
    losses = []
    for index in indices:
        student = F.layer_norm(student_states[index], student_states[index].shape[-1:])
        teacher = F.layer_norm(teacher_states[index].detach(), teacher_states[index].shape[-1:])
        distance = 1.0 - F.cosine_similarity(student, teacher, dim=-1)
        if attention_mask is None:
            losses.append(distance.mean())
        else:
            mask = attention_mask.to(distance.dtype)
            losses.append((distance * mask).sum() / mask.sum().clamp_min(1))
    return torch.stack(losses).mean()
