import torch

from soft_target_finetuning.objectives import acquisition_ratio, solve_sequence_floor


def test_budget_endpoints_and_monotonicity():
    probs = torch.tensor([0.1, 0.4, 0.8])
    taus = [solve_sequence_floor(probs, budget) for budget in (0.0, 0.25, 0.75, 1.0)]
    assert all(left <= right for left, right in zip(taus, taus[1:]))
    assert torch.isclose(acquisition_ratio(probs, taus[1]), torch.tensor(0.25), atol=1e-5)
    assert torch.isclose(acquisition_ratio(probs, taus[2]), torch.tensor(0.75), atol=1e-5)

