# SoFT: Soft-Target Fine-Tuning

Minimal training code for **SoFT: Soft Targets for Generalizable LLM Fine-Tuning**.

SoFT replaces the one-hot SFT target with the smallest change to a fixed Base-model
distribution that raises the demonstrated token to a sequence-specific probability
floor. A gradient budget `R*` determines the floor by

```text
sum_t max(tau_x - p_t, 0) / sum_t (1 - p_t) = R*.
```

For `p_t < tau_x`, the target can be written as
`q_t = a_t delta_gold + (1-a_t) pi_0`, where
`a_t = (tau_x-p_t)/(1-p_t)`. This repository implements the equivalent weighted
cross entropy exactly, using a frozen Base model as the reference.

## Install

```bash
pip install -e .
```

## Train

The dataset must contain either a `messages` column or `prompt` and `completion`
message lists, plus a `domain` or `task` column.

```bash
# Standard SFT
accelerate launch -m soft_target_finetuning.train \
  --objective sft --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset Egbertjing/SoFT-Mixed-Domain-Distillation \
  --output-dir outputs/sft --bf16

# SoFT with the released 1.5B reasoning budgets
accelerate launch -m soft_target_finetuning.train \
  --objective soft --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset Egbertjing/SoFT-Mixed-Domain-Distillation \
  --budgets configs/reasoning-1.5b.json --output-dir outputs/soft --bf16
```

Released budget files are provided separately for the 1.5B reasoning, 7B
reasoning, and 8B agentic settings.

The implementation intentionally contains only the SFT and fixed-reference SoFT
objectives. Experiment orchestration, internal paths, credentials, evaluation
outputs, and unrelated baselines are excluded.

## Released artifacts

- `Egbertjing/SoFT-Qwen2.5-1.5B-Reasoning`
- `Egbertjing/SoFT-Qwen2.5-7B-Reasoning`
- `Egbertjing/SoFT-Qwen3-8B-Agentic`
- `Egbertjing/SoFT-Mixed-Domain-Distillation`

## Citation

```bibtex
@article{jing2026soft,
  title={SoFT: Soft Targets for Generalizable LLM Fine-Tuning},
  author={Jing, Huihao and Hu, Wenbin and Chen, Shaojin and Shi, Haochen and Xie, Zhongwei and Zhang, Guijia and Liu, Yuxuan and Huang, Haoyu and Li, Haoran and Song, Yangqiu},
  journal={arXiv preprint arXiv:2609.32493},
  year={2026}
}
```
