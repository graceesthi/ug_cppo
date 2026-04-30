# UG-CPPO: Uncertainty-Gated LLM Infusion for Risk-Sensitive Trading Agents

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS%202026-FinAI%20Contest-red.svg)](https://openreview.net/group?id=NeurIPS.cc/2026/Conference)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset%20%26%20Models-yellow)](https://huggingface.co/graceesthi/ug-cppo-finai-2025)

> **TL;DR** — UG-CPPO extends FinRL-DeepSeek by gating LLM trading signals through epistemic uncertainty estimation. We query the LLM with N=5 semantically diverse prompts per article, compute σ across responses, and suppress signals when σ exceeds threshold τ. The mechanism is borrowed from Monte Carlo Dropout in medical imaging AI.

> **Author** — Grace-Esther Dong · Aivancity Paris-Cachan
> **Paper** — [`UG_CPPO_paper.pdf`](./paper/UG_CPPO_paper.pdf) (NeurIPS 2024 format, 8 pages)
> **Submission** — FinAI Contest 2025, Task 1, IEEE CSCloud 2025

---

## 🎯 Key Results (v11, 500k steps, real OpenAI gpt-4o-mini signals)

| Model | Cumul. Return | Rachev Ratio | Max Drawdown | Outperf. Bear | Mean \|a\| |
|:------|:-------------:|:------------:|:------------:|:-------------:|:--------:|
| QQQ Benchmark | 173.26% | — | — | — | — |
| PPO | 93.21% | 0.9433 | −23.73% | 51.0% | 0.881 |
| CPPO | 54.30% | 0.9242 | **−18.35%** | 49.8% | 0.726 |
| **UG-CPPO** (ours) | **103.73%** | 0.9396 | −29.52% | **51.0%** | 0.893 |

**Validated hypotheses**: H1 (gate rate 34.2% in target [0.30, 0.40]) ✓ · H2 (UG-CPPO > PPO cumret) ✓ · H3 (UG-CPPO Rachev > CPPO) ✓ · H4 (UG-CPPO MDD < CPPO) ✗ — discussed in paper Section 7.

---

## 🧠 The Idea in 3 Lines

1. **Problem**: FinRL-DeepSeek (Benhenda 2025) shows LLM signal infusion *degrades* RL trading performance because the agent treats every LLM output as deterministic.
2. **Insight from medical imaging**: Monte Carlo Dropout in our prior breast cancer detection work [(IABM 2026)](https://hal.science/hal-05561689v2) shows that epistemic uncertainty must gate downstream decisions.
3. **Mechanism**: We transfer this paradigm — query LLM with N=5 diverse prompts, gate the infusion when std(scores) > τ.

```
Sf = 1 + α · δ · c(σ) · m(μ)            ← action modifier
c(σ) = max(0, 1 - σ/σ_max)               ← confidence gate
gate fires when c(σ) < τ → Sf = 1.0      ← signal suppressed
```

---

##  Quick Start

### Install

```bash
git clone https://github.com/graceesthi/ug_cppo.git
cd ug_cppo
pip install -r requirements.txt
```

### Reproduce the paper results

```bash
# Set your OpenAI API key (gpt-4o-mini, ~$3-5 total cost)
echo "OPENAI_API_KEY=sk-..." > .env

# Or use the pre-computed signals from HuggingFace (free, instant)
huggingface-cli download graceesthi/ug-cppo-finai-2025-signals \
    ug_signals.parquet --local-dir data/

# Run the full notebook (~3 hours on Apple M-series CPU)
jupyter notebook notebooks/UG_CPPO_v11.ipynb
```

### Run individual stages

```bash
# Pre-compute LLM signals from FNSPID (~2-3h, ~$3-5 OpenAI)
python scripts/precompute_signals.py --provider openai

# Train each agent (≈30 min/agent on M1/M2 CPU)
python scripts/train.py --mode ppo     --timesteps 500000
python scripts/train.py --mode cppo    --timesteps 500000
python scripts/train.py --mode ug_cppo --timesteps 500000

# Evaluate on 2019-2023 trading period
python scripts/evaluate.py
```

---

##  Repository Structure

```
ug_cppo/
├── src/
│   ├── uncertainty_llm.py    # Prompt ensemble + uncertainty gate (★ core)
│   ├── ug_cppo_env.py         # FinRL env extended with Sf/Rf injection
│   ├── cvar_ppo.py            # Minimal CVaRPPO (loss patched via callback)
│   ├── data_pipeline.py       # FNSPID + OHLCV loading, signal precompute
│   └── evaluation.py          # 4 contest metrics + calibration stats
├── scripts/
│   ├── precompute_signals.py  # LLM ensemble queries with checkpoint
│   ├── train.py               # Training entry point with chunked checkpointing
│   ├── evaluate.py            # Evaluation on trade period
│   └── ablation.py            # Hyperparameter ablation runner
├── configs/
│   └── default.yaml           # All hyperparameters in one place
├── notebooks/
│   └── UG_CPPO_v11.ipynb      # End-to-end notebook (final paper version)
├── paper/
│   ├── main.tex               # NeurIPS LaTeX source
│   ├── references.bib         # All references (no hallucinations)
│   └── UG_CPPO_paper.pdf      # Compiled paper
└── requirements.txt
```

---

##  Reproducibility

### Hardware used
- **Compute**: Apple M-series (CPU only, no GPU)
- **Time**: ~3-4 hours total (500k steps × 3 agents)
- **Cost**: ~$3.40 OpenAI gpt-4o-mini for 28,502 article signals (×9 prompts each)

### Hyperparameters (paper Section 5.2)
- Training: lr=1e-3, n_steps=1024, batch_size=128, γ=0.99, GAE λ=0.95
- UG-CPPO specific: α=0.05, τ=0.40, CVaR α=0.05, CVaR λ=0.10
- Environment: hmax=1000, transaction_cost=0.001, reward_scaling=1e-2

### Random seeds
All experiments use seed=42. Multi-seed robustness analysis is listed as future work in paper Section 8.

### Pre-computed artifacts on HuggingFace

- **Signals**: [graceesthi/ug-cppo-finai-2025-signals](https://huggingface.co/datasets/graceesthi/ug-cppo-finai-2025-signals)
  *55,360 LLM uncertainty signals (μ, σ, c(σ)) for 28,502 (ticker, date) pairs*
- **Models**: [graceesthi/ug-cppo-finai-2025](https://huggingface.co/graceesthi/ug-cppo-finai-2025)
  *Trained PPO, CPPO, UG-CPPO checkpoints + final models*

---

##  What the Numbers Mean

The Rachev ratio is the contest's primary risk metric: it measures **expected tail gain over expected tail loss** at the 5% confidence level. Higher = better gain/loss asymmetry.

UG-CPPO achieves:
- **Higher cumulative return** than PPO (103.73% vs 93.21%) → the gate doesn't sacrifice growth
- **Higher Rachev ratio** than CPPO (0.9396 vs 0.9242) → better risk-adjusted asymmetry
- **Tied outperformance in bear markets** (51.0%) → robust to 2022 downturn
- **Higher max drawdown** than CPPO (−29.52% vs −18.35%) → trades more aggressively when confident, accepted as a tradeoff

---

##  Citation

```bibtex
@inproceedings{dong2026ugcppo,
  title={UG-CPPO: Uncertainty-Gated LLM Infusion for Risk-Sensitive 
         Reinforcement Learning Trading Agents},
  author={Dong, Grace-Esther},
  booktitle={NeurIPS 2026 — FinAI Contest 2025, Task 1},
  year={2026}
}
```

### Related work by the author (transfer source)

```bibtex
@inproceedings{dong2026mammography,
  title={Improving Diagnostic Confidence in Breast Cancer Detection 
         through Spatial Attention and Uncertainty Modeling},
  author={Dong, Grace-Esther and Manfouho, Léana a Yemene and 
          Mounzeo, Gil-Allen M and Bagnenda, Johyce D and 
          Kar, Anuradha and Ammar, Doreid},
  booktitle={IABM 2026: 4ème Colloque Français d'Intelligence 
             Artificielle en Imagerie Biomédicale},
  address={Lyon, France},
  month=mar,
  year={2026},
  url={https://hal.science/hal-05561689v2}
}
```

### Baseline

```bibtex
@misc{benhenda2025,
  title={FinRL-DeepSeek: LLM-Infused Risk-Sensitive Reinforcement Learning 
         for Trading Agents},
  author={Mostapha Benhenda},
  year={2025},
  eprint={2502.07393},
  archivePrefix={arXiv}
}
```

---

##  Contributing

This is a research repository for a contest submission, not actively maintained for production use. Issues and discussion are welcome. For follow-up work, please cite the paper.

---

##  License

MIT License — see [LICENSE](./LICENSE) for details.

---

##  Acknowledgments

- **Mostapha Benhenda** — FinRL-DeepSeek baseline
- **AI4Mammography team** — for the prior medical imaging research that grounds this work conceptually

---

*Last updated: 2026-04-30 · v11 final · ready for OpenReview submission*
