import torch
import torch.nn.functional as F

from soft_target_finetuning.objectives import acquisition_ratio, solve_sequence_floor, soft_target_loss


def test_budget_endpoints_and_monotonicity():
    probs = torch.tensor([0.1, 0.4, 0.8])
    taus = [solve_sequence_floor(probs, budget) for budget in (0.0, 0.25, 0.75, 1.0)]
    assert all(left <= right for left, right in zip(taus, taus[1:]))
    assert torch.isclose(acquisition_ratio(probs, taus[1]), torch.tensor(0.25), atol=1e-5)
    assert torch.isclose(acquisition_ratio(probs, taus[2]), torch.tensor(0.75), atol=1e-5)


def test_soft_endpoints_match_preservation_and_sft():
    torch.manual_seed(0)
    student = torch.randn(1, 5, 11)
    base = torch.randn(1, 5, 11)
    labels = torch.tensor([[-100, 2, 4, 6, 8]])
    shifted_student = F.log_softmax(student[:, :-1], dim=-1)[0]
    shifted_base = F.softmax(base[:, :-1], dim=-1)[0]
    gold = labels[:, 1:][0]
    expected_sft = -shifted_student.gather(-1, gold[:, None]).mean()
    expected_preservation = -(shifted_base * shifted_student).sum(-1).mean()
    assert torch.allclose(
        soft_target_loss(student, base, labels, torch.tensor([1.0])), expected_sft, atol=1e-5
    )
    assert torch.allclose(
        soft_target_loss(student, base, labels, torch.tensor([0.0])),
        expected_preservation,
        atol=1e-5,
    )

