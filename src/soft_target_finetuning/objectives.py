from __future__ import annotations

import torch
import torch.nn.functional as F


def acquisition_ratio(base_gold_probs: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """SFT-gradient-weighted mean acquisition weight at a sequence floor."""
    numerator = torch.clamp(tau - base_gold_probs, min=0).sum()
    denominator = torch.clamp(1 - base_gold_probs, min=1e-12).sum()
    return numerator / denominator


@torch.no_grad()
def solve_sequence_floor(
    base_gold_probs: torch.Tensor,
    gradient_budget: float,
    iterations: int = 32,
) -> torch.Tensor:
    """Solve R_x(tau_x) = R* by scalar bisection."""
    if not 0 <= gradient_budget <= 1:
        raise ValueError("gradient_budget must be in [0, 1]")
    low = torch.zeros((), dtype=torch.float64, device=base_gold_probs.device)
    high = torch.ones((), dtype=torch.float64, device=base_gold_probs.device)
    probs = base_gold_probs.double()
    for _ in range(iterations):
        mid = (low + high) / 2
        if acquisition_ratio(probs, mid) < gradient_budget:
            low = mid
        else:
            high = mid
    return ((low + high) / 2).to(base_gold_probs.dtype)


def soft_target_loss(
    student_logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    gradient_budgets: torch.Tensor,
) -> torch.Tensor:
    """Exact SoFT cross entropy with one sequence-level budget per example.

    ``labels`` uses -100 outside supervised assistant tokens. The fixed Base
    logits must be detached (or produced under ``torch.no_grad()``).
    """
    shift_student = student_logits[:, :-1].float()
    shift_base = base_logits[:, :-1].float().detach()
    shift_labels = labels[:, 1:]
    losses: list[torch.Tensor] = []

    for batch_idx in range(shift_labels.size(0)):
        mask = shift_labels[batch_idx].ne(-100)
        gold = shift_labels[batch_idx, mask]
        if gold.numel() == 0:
            continue
        student_log_probs = F.log_softmax(shift_student[batch_idx, mask], dim=-1)
        base_log_probs = F.log_softmax(shift_base[batch_idx, mask], dim=-1)
        base_probs = base_log_probs.exp()
        base_gold = base_probs.gather(-1, gold[:, None]).squeeze(-1)
        tau = solve_sequence_floor(base_gold, float(gradient_budgets[batch_idx]))
        weights = torch.where(
            base_gold < tau,
            (tau - base_gold) / (1 - base_gold).clamp_min(1e-12),
            torch.zeros_like(base_gold),
        )
        demonstration_ce = -student_log_probs.gather(-1, gold[:, None]).squeeze(-1)
        preservation_ce = -(base_probs * student_log_probs).sum(dim=-1)
        losses.append((weights * demonstration_ce + (1 - weights) * preservation_ce).mean())

    if not losses:
        raise ValueError("batch contains no supervised assistant tokens")
    return torch.stack(losses).mean()

