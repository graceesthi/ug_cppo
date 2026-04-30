# UG-CPPO: Uncertainty-Gated LLM Infusion for Risk-Sensitive Trading Agents

**FinAI Contest 2025 — Task 1 — IEEE CSCloud 2025**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/ug_cppo/blob/main/notebooks/UG_CPPO_FinAI_Contest.ipynb)

---

## 🎯 Core Idea

FinRL-DeepSeek shows that naive LLM infusion degrades RL trading performance (Table 2, Benhenda 2025). We identify the root cause: **the system treats every LLM output as equally reliable**. A score of "4" from a clear news article and a score of "4" from an ambiguous, contradictory one receive identical treatment.

**UG-CPPO** fixes this by borrowing *epistemic uncertainty estimation* from medical imaging AI (Monte Carlo Dropout). We query the LLM N=5 times with semantically diverse prompts, compute σ across responses, and gate the infusion proportionally to confidence:

```
Sf = 1 + α · δ · c(σ) · m(μ)          # action modifier
c(σ) = max(0, 1 - σ/σ_max)            # confidence gate
Gate = 0  when  c(σ) < τ               # kill noisy signals
```

---

## 🗂️ Project Structure

```
ug_cppo/
├── src/
│   ├── uncertainty_llm.py   # ★ Core: prompt ensemble + uncertainty gate
│   ├── ug_cppo_env.py       # Modified FinRL env with Sf/Rf injection
│   ├── cvar_ppo.py          # CVaR-PPO agent with uncertainty-gated loss
│   ├── data_pipeline.py     # FNSPID + OHLCV loading & signal precompute
│   └── evaluation.py        # All 4 contest metrics + calibration stats
├── scripts/
│   ├── train.py             # Main training script
│   ├── precompute_signals.py# Offline LLM signal precomputation
│   └── ablation.py          # Full factorial ablation study
├── configs/
│   └── default.yaml         # All hyperparameters
├── notebooks/
│   └── UG_CPPO_FinAI_Contest.ipynb  # Full Colab notebook
└── results/                 # Models, logs, reports
```

---

## 🚀 Quick Start

### Option 1: Colab (recommended)
Click the badge above. Set `USE_MOCK_LLM = True` for a cost-free demo.

### Option 2: Local

```bash
git clone https://github.com/YOUR_USERNAME/ug_cppo
cd ug_cppo
pip install -r requirements.txt

# Step 1: Pre-compute LLM signals (once, ~hours with real API)
export DEEPSEEK_API_KEY=sk-xxx
python scripts/precompute_signals.py

# Or use mock for testing:
python scripts/precompute_signals.py --mock --n-tickers 10

# Step 2: Train
python scripts/train.py --mode ug_cppo --seed 42
python scripts/train.py --mode cppo    --seed 42   # baseline
python scripts/train.py --mode ppo     --seed 42   # baseline

# Step 3: Ablation study
python scripts/ablation.py --timesteps 500000
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Cumulative Return | Total return over 2019–2023 trading period |
| Rachev Ratio | ETG(5%) / ETL(5%) — extreme upside vs. extreme downside |
| Max Drawdown | Largest peak-to-trough drop |
| Outperf. Frequency | % of days outperforming Nasdaq-100, especially bear markets |

---

## 🔬 Architecture

```
Price Data  ──► PPO Policy ──► Raw Action aₜ ──────────────────────┐
                                                                     │
News Data   ──► LLM Ensemble ──► Uncertainty (μ,σ) ──► Sf (gated) ─► aₜ_mod
                (N=5 prompts)       ↓                                │
                              Gate c(σ)                              ▼
                                    └──► Rf (gated) ──► CVaR-PPO Loss
```

---

## 📈 Expected Results

| Model | Cumul. Return | Rachev Ratio | Max Drawdown | Outperf. Freq. (Bear) |
|-------|:---:|:---:|:---:|:---:|
| PPO (baseline) | ~1.8× | 1.064 | ~−25% | Moderate |
| CPPO-DeepSeek 10% | ~1.5× | 0.982 | ~−30% | Low |
| **UG-CPPO (ours)** | **≥ 2.0×** | **> 1.10** | **< −15%** | **High** |

---

## 🤗 HuggingFace

Pre-computed signals and trained agents: `huggingface.co/YOUR_USERNAME/ug-cppo-finai-2025`

---

## 📄 Citation

```bibtex
@misc{ugcppo2025,
  title={Uncertainty-Gated LLM Infusion for Risk-Sensitive Reinforcement Learning Trading Agents},
  author={Grace Esther},
  year={2025},
  note={FinAI Contest 2025, Task 1, IEEE CSCloud 2025}
}
```

---

## 🔑 Key Dependencies

- `finrl` — RL trading environment
- `stable-baselines3` — PPO implementation
- `openai` — DeepSeek API (OpenAI-compatible)
- `yfinance` — OHLCV data download
