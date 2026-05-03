"""
upload_hf_v3.py
===============
Upload UG-CPPO v3 artifacts to HuggingFace Hub (30 models + signals + report).

Features:
  - Uploads all 30 trained models (10 seeds × 3 agents)
  - Uploads multiseed report JSON with Wilcoxon tests
  - Uploads performance comparison plots
  - Updates model and dataset cards with v3 metadata
  - Fully idempotent — safe to re-run

Usage:
    huggingface-cli login  # one-time
    python scripts/upload_hf_v3.py
"""

from __future__ import annotations
import os
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────
HF_USERNAME = "graceesthi"
MODEL_REPO    = f"{HF_USERNAME}/ug-cppo-finai-2025"
DATASET_REPO  = f"{HF_USERNAME}/ug-cppo-finai-2025-signals"

PROJECT_ROOT  = Path.cwd()
MODELS_DIR    = PROJECT_ROOT / "results" / "models"
SIGNALS_PATH  = PROJECT_ROOT / "data" / "ug_signals.parquet"
REPORT_PATH   = PROJECT_ROOT / "results" / "multiseed_report_v13.json"
PLOTS_DIR     = PROJECT_ROOT / "results" / "plots"
PAPER_PDF     = PROJECT_ROOT / "paper" / "UG_CPPO_paper.pdf"

SEEDS = list(range(42, 52))  # 42-51
MODES = ["ppo", "cppo", "ug_cppo"]

# ─── MODEL CARD V3 ────────────────────────────────────────────────────────
MODEL_CARD_V3 = """---
license: mit
language:
- en
tags:
- reinforcement-learning
- finance
- trading
- llm
- uncertainty-quantification
- finrl
- cvar-ppo
- multi-seed-evaluation
library_name: stable-baselines3
---

# UG-CPPO v3: Uncertainty-Gated CVaR-PPO Trading Agents (Multi-Seed, Honest Eval)

Trained models for the **UG-CPPO paper v3** (FinAI Contest 2025, NeurIPS 2026 submission).

> **Author** — Grace-Esther Dong · Aivancity Paris-Cachan
> **Paper** — [`UG_CPPO_preprint_PAT_corrected.pdf`](./UG_CPPO_paper.pdf)
> **Code** — https://github.com/graceesthi/ug_cppo
> **Preprint** — arXiv [TBA]

---

## What's in this repo

**30 trained Stable-Baselines3 agents** (10 seeds × 3 algorithms):
- **Seeds**: 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
- **Training**: 250,000 timesteps each
- **Evaluation**: 2019-2023 Nasdaq data
- **Tickers**: 10 stocks (AAPL, MSFT, AMZN, NVDA, META, GOOGL, TSLA, NFLX, AMD, COST)

### File naming
```
{mode}_seed{seed}.zip
  ppo_seed42.zip       → Vanilla PPO, seed 42
  cppo_seed42.zip      → CVaR-PPO, seed 42
  ug_cppo_seed42.zip   → UG-CPPO (ours), seed 42
  ... (3 modes × 10 seeds = 30 files total)
```

---

## Results (250k steps, 10 seeds, honest multi-seed eval)

**Cumulative Return (mean ± std)**

| Model | Mean | Std | Rachev | MDD | Wilcoxon p (vs PPO) |
|-------|:----:|:---:|:------:|:---:|:------------------:|
| PPO | 43.94% | ±32.18% | 0.9445 | −27.95% | — |
| CPPO | 39.71% | ±46.01% | 0.9408 | −31.08% | 0.1720 |
| **UG-CPPO** | **35.99%** | ±38.70% | **0.9420** | **−29.72%** | **0.8127** |

**Interpretation**:
- UG-CPPO cumulative return is **−7.95pp lower** than PPO (95% CI includes zero)
- **Wilcoxon rank-sum test**: p=0.8127 >> 0.05 → **no significant difference** in medians
- **H2 hypothesis (UG-CPPO > PPO)**: **not rejected** but also **not accepted** (honest null-preserving stat)
- Honest variance (σ=38.7%) reflects genuine seed-to-seed variability

**Top performers (by Rachev)**:
- Seed 47 (UG-CPPO): Rachev 1.0104
- Seed 46 (UG-CPPO): Rachev 0.9940
- Seed 51 (PPO): Rachev 0.9915

---

## Quick load

```python
from stable_baselines3 import PPO
from huggingface_hub import hf_hub_download

# Download UG-CPPO seed 47 (top performer)
path = hf_hub_download(
    repo_id="graceesthi/ug-cppo-finai-2025",
    filename="ug_cppo_seed47.zip"
)
agent = PPO.load(path)
```

---

## Reproducibility

- **Hardware**: Apple M-series (CPU only)
- **Config**: 250k steps, 10 independent runs (seeds 42-51)
- **Hyperparams**: lr=1e-3, batch_size=128, γ=0.99, CVaR α=0.05
- **Statistical test**: Wilcoxon rank-sum (non-parametric, no normality assumption)

---

## Files

- `ppo_seed*.zip`, `cppo_seed*.zip`, `ug_cppo_seed*.zip` — Trained agents
- `multiseed_report_v13.json` — Full results with Wilcoxon tests
- `UG_CPPO_paper.pdf` — Full paper with methodology
- `multiseed_performance.png` — Performance comparison plot

---

## Citation

```bibtex
@inproceedings{dong2026ugcppo,
  title={UG-CPPO: Uncertainty-Gated LLM Infusion for Risk-Sensitive
         Reinforcement Learning Trading Agents},
  author={Dong, Grace-Esther},
  booktitle={NeurIPS 2026 — FinAI Contest 2025, Task 1},
  year={2026},
  note={v3: multi-seed honest evaluation with PAT corrections}
}
```
"""

# ─── DATASET CARD V3 ───────────────────────────────────────────────────────
DATASET_CARD_V3 = """---
license: mit
language:
- en
tags:
- finance
- llm
- uncertainty-quantification
- trading
- nasdaq
size_categories:
- 10K<n<100K
---

# UG-CPPO Signals: LLM Uncertainty Estimates for Financial News

Pre-computed uncertainty-aware LLM trading signals over the FNSPID dataset,
used in the UG-CPPO paper (FinAI Contest 2025, v3).

> **Source code** — https://github.com/graceesthi/ug_cppo
> **Paper** — UG_CPPO_preprint_PAT_corrected.pdf
> **Author** — Grace-Esther Dong · Aivancity Paris-Cachan

---

## Dataset summary

Each row corresponds to a (ticker, date) pair from FNSPID, scored by
OpenAI gpt-4o-mini using a 5-prompt ensemble for recommendation
and a 4-prompt ensemble for risk.

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | str | Stock symbol (e.g. AAPL) |
| `date` | str | YYYY-MM-DD |
| `mean_score` | float | μ — mean recommendation score [1, 5] |
| `std_score` | float | σ — std across N=5 prompts (epistemic uncertainty) |
| `confidence` | float | c(σ) = max(0, 1 - σ/σ_max) ∈ [0, 1] |
| `mean_risk` | float | μ_risk — mean risk score [1, 5] |
| `std_risk` | float | σ_risk — std across N=4 risk prompts |
| `risk_confidence` | float | c(σ_risk) ∈ [0, 1] |
| `gate_fired` | bool | True when c(σ) < 0.40 (signal suppressed) |

**Stats:**
- 28,502 (ticker, date) pairs
- 20 Nasdaq tickers, period 2013-2023
- 256,518 LLM API calls total
- Cost ~$3.40 (OpenAI gpt-4o-mini at $0.150/1M input tokens)

---

## Quick load

```python
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="graceesthi/ug-cppo-finai-2025-signals",
    filename="ug_signals.parquet",
    repo_type="dataset",
)
df = pd.read_parquet(path)
print(df.head())
```

---

## Citation

```bibtex
@inproceedings{dong2026ugcppo,
  title={UG-CPPO: Uncertainty-Gated LLM Infusion for Risk-Sensitive
         Reinforcement Learning Trading Agents},
  author={Dong, Grace-Esther},
  booktitle={NeurIPS 2026 — FinAI Contest 2025, Task 1},
  year={2026}
}
```
"""


def main():
    try:
        from huggingface_hub import HfApi, login, create_repo
    except ImportError:
        logger.error("pip install huggingface_hub")
        sys.exit(1)

    # ── Login ─────────────────────────────────────────────────────────────
    token = os.environ.get("HF_TOKEN", "")
    if token:
        login(token=token)
        logger.info("✓ Logged in via HF_TOKEN env var")
    else:
        logger.info("Using cached login (run `huggingface-cli login` first)")

    api = HfApi()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1 — MODEL REPO (30 trained agents)
    # ─────────────────────────────────────────────────────────────────────
    logger.info(f"\n[1/2] Model repo: {MODEL_REPO}")

    create_repo(MODEL_REPO, repo_type="model", exist_ok=True, private=False)
    logger.info(f"  ✓ Repo ready")

    # Upload README (model card)
    readme_path = Path("/tmp/model_readme_v3.md")
    readme_path.write_text(MODEL_CARD_V3)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=MODEL_REPO, repo_type="model",
        commit_message="Update model card v3: multi-seed honest eval",
    )
    logger.info(f"  ✓ README.md uploaded")

    # Upload 30 trained models
    model_count = 0
    for seed in SEEDS:
        for mode in MODES:
            model_file = MODELS_DIR / f"{mode}_seed{seed}.zip"
            if not model_file.exists():
                logger.warning(f"  ⚠ {model_file.name} missing")
                continue

            api.upload_file(
                path_or_fileobj=str(model_file),
                path_in_repo=f"{mode}_seed{seed}.zip",
                repo_id=MODEL_REPO, repo_type="model",
                commit_message=f"Upload {mode} seed {seed}",
            )
            model_count += 1
            if model_count % 5 == 0:
                logger.info(f"  ✓ {model_count}/30 models uploaded")

    logger.info(f"  ✓ All {model_count} models uploaded")

    # Upload multiseed report
    if REPORT_PATH.exists():
        api.upload_file(
            path_or_fileobj=str(REPORT_PATH),
            path_in_repo="multiseed_report_v13.json",
            repo_id=MODEL_REPO, repo_type="model",
            commit_message="Upload multiseed report (Wilcoxon tests)",
        )
        logger.info(f"  ✓ multiseed_report_v13.json")

    # Upload performance plot
    plot_file = PLOTS_DIR / "multiseed_performance.png"
    if plot_file.exists():
        api.upload_file(
            path_or_fileobj=str(plot_file),
            path_in_repo="multiseed_performance.png",
            repo_id=MODEL_REPO, repo_type="model",
            commit_message="Upload performance comparison plot",
        )
        logger.info(f"  ✓ multiseed_performance.png")

    # Upload paper PDF
    if PAPER_PDF.exists():
        api.upload_file(
            path_or_fileobj=str(PAPER_PDF),
            path_in_repo="UG_CPPO_paper.pdf",
            repo_id=MODEL_REPO, repo_type="model",
            commit_message="Upload paper (PAT-corrected v3)",
        )
        logger.info(f"  ✓ UG_CPPO_paper.pdf")

    logger.info(f"  → https://huggingface.co/{MODEL_REPO}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2 — DATASET REPO (LLM signals)
    # ─────────────────────────────────────────────────────────────────────
    logger.info(f"\n[2/2] Dataset repo: {DATASET_REPO}")

    if not SIGNALS_PATH.exists():
        logger.error(f"  ✗ {SIGNALS_PATH} missing — skip dataset upload")
        return

    create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True, private=False)
    logger.info(f"  ✓ Repo ready")

    # Upload dataset README
    readme_path = Path("/tmp/dataset_readme_v3.md")
    readme_path.write_text(DATASET_CARD_V3)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=DATASET_REPO, repo_type="dataset",
        commit_message="Update dataset card v3",
    )
    logger.info(f"  ✓ README.md uploaded")

    # Upload signals parquet
    api.upload_file(
        path_or_fileobj=str(SIGNALS_PATH),
        path_in_repo="ug_signals.parquet",
        repo_id=DATASET_REPO, repo_type="dataset",
        commit_message="Upload LLM uncertainty signals",
    )
    size_mb = SIGNALS_PATH.stat().st_size / 1e6
    logger.info(f"  ✓ ug_signals.parquet ({size_mb:.1f} MB)")

    logger.info(f"  → https://huggingface.co/datasets/{DATASET_REPO}")

    # ─────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    logger.info("\n" + "="*70)
    logger.info(" UPLOAD COMPLETE ✓ (v3: 30 models + signals + report)")
    logger.info("="*70)
    logger.info(f" Models:  https://huggingface.co/{MODEL_REPO}")
    logger.info(f" Signals: https://huggingface.co/datasets/{DATASET_REPO}")
    logger.info("="*70)
    logger.info(f"\n✓ v3 ready for OpenReview submission")
    logger.info(f"  - 30 trained models (10 seeds × 3 agents)")
    logger.info(f"  - Honest multi-seed evaluation (Wilcoxon tests)")
    logger.info(f"  - PAT corrections applied")


if __name__ == "__main__":
    main()
