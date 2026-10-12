<div align="center">

# SoFT: Soft Targets for Generalizable LLM Fine-Tuning

<p>
  <a href="https://openreview.net/profile?id=~Huihao_JING1">Huihao Jing</a><sup>1</sup> ·
  <a href="https://openreview.net/profile?id=~Wenbin_Hu3">Wenbin Hu</a><sup>1</sup> ·
  <a href="https://openreview.net/profile?id=~Shaojin_Chen1">Shaojin Chen</a><sup>1</sup> ·
  <a href="https://openreview.net/profile?id=~Haochen_Shi4">Haochen Shi</a><sup>1</sup> ·
  <a href="https://openreview.net/profile?id=~Zhongwei_Xie2">Zhongwei Xie</a><sup>1</sup><br>
  <a href="https://openreview.net/profile?id=~Guijia_Zhang1">Guijia Zhang</a><sup>2</sup> ·
  <a href="https://openreview.net/profile?id=~Yuxuan_Liu44">Yuxuan Liu</a><sup>1</sup> ·
  <a href="https://openreview.net/profile?id=~Haoyu_Huang2">Haoyu Huang</a><sup>1</sup> ·
  <a href="https://openreview.net/profile?id=~Haoran_Li3">Haoran Li</a><sup>1</sup> ·
  <a href="https://openreview.net/profile?id=~Yangqiu_Song1">Yangqiu Song</a><sup>1</sup>
</p>

<p><sup>1</sup>HKUST &nbsp;&nbsp; <sup>2</sup>HKUST(GZ)</p>

<p><a href="https://arxiv.org/abs/2609.32493"><strong>📄 Open-source paper</strong></a></p>

</div>

---

## Overview

SoFT replaces the one-hot SFT target with the smallest change to a fixed Base-model
distribution that raises the demonstrated token to a sequence-specific probability
floor. A gradient budget $R^\star$ determines the floor through

$$
\frac{\sum_t \max(\tau_x-p_t,0)}{\sum_t(1-p_t)}=R^\star.
$$

For $p_t<\tau_x$, the target can be written as
$q_t=a_t\delta_{\mathrm{gold}}+(1-a_t)\pi_0$, where
$a_t=(\tau_x-p_t)/(1-p_t)$. This repository implements the equivalent weighted
cross entropy exactly using a frozen Base model as the reference.

## Installation

```bash
pip install -e .
```

## Training

The dataset must contain either a `messages` column or `prompt` and `completion`
message lists, plus a `domain` or `task` column.

```bash
# Standard SFT
accelerate launch -m soft_target_finetuning.train \
  --objective sft --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset Egbertjing/SoFT-Mixed-Domain-Distillation \
  --track reasoning \
  --output-dir outputs/sft --bf16

# SoFT with the released 1.5B reasoning budgets
accelerate launch -m soft_target_finetuning.train \
  --objective soft --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset Egbertjing/SoFT-Mixed-Domain-Distillation \
  --track reasoning \
  --budgets configs/reasoning-1.5b.json --output-dir outputs/soft --bf16
```

Released budget files are provided separately for the 1.5B reasoning, 7B
reasoning, and 8B agentic settings.

The implementation intentionally contains only the SFT and fixed-reference SoFT
objectives. Experiment orchestration, internal paths, credentials, evaluation
outputs, and unrelated baselines are excluded.

## Related Artifacts

| Artifact | Description |
|---|---|
| [SoFT-Qwen2.5-1.5B-Reasoning](https://huggingface.co/Egbertjing/SoFT-Qwen2.5-1.5B-Reasoning) | 1.5B reasoning checkpoint |
| [SoFT-Qwen2.5-7B-Reasoning](https://huggingface.co/Egbertjing/SoFT-Qwen2.5-7B-Reasoning) | 7B reasoning checkpoint |
| [SoFT-Qwen3-8B-Agentic](https://huggingface.co/Egbertjing/SoFT-Qwen3-8B-Agentic) | 8B agentic checkpoint |
| [SoFT-Mixed-Domain-Distillation](https://huggingface.co/datasets/Egbertjing/SoFT-Mixed-Domain-Distillation) | Unified training, validation, and test data |

## Citation

```bibtex
@article{jing2026soft,
  title={SoFT: Soft Targets for Generalizable LLM Fine-Tuning},
  author={Jing, Huihao and Hu, Wenbin and Chen, Shaojin and Shi, Haochen and Xie, Zhongwei and Zhang, Guijia and Liu, Yuxuan and Huang, Haoyu and Li, Haoran and Song, Yangqiu},
  journal={arXiv preprint arXiv:2609.32493},
  year={2026}
}
```
