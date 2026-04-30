"""
upload_hf.py
============
Upload UG-CPPO artifacts to HuggingFace Hub.

Creates 2 repos:
  1. graceesthi/ug-cppo-finai-2025          (model — trained agents)
  2. graceesthi/ug-cppo-finai-2025-signals  (dataset — LLM uncertainty signals)

Usage:
    # 1. Login first (one-time)
    huggingface-cli login
    # OR set HF_TOKEN env var

    # 2. Run from project root
    cd ~/Downloads/ug_cppo
    python scripts/upload_hf.py

The script is IDEMPOTENT — re-running it just updates files that changed.
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
PAPER_PDF     = PROJECT_ROOT / "paper" / "UG_CPPO_paper.pdf"

# ─── MODEL CARD ──────────────────────────────────────────────────────────
MODEL_CARD = """---
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
library_name: stable-baselines3
---

# UG-CPPO: Uncertainty-Gated CVaR-PPO Trading Agents

Trained models for the **UG-CPPO paper** (FinAI Contest 2025, NeurIPS 2026 submission).

> **Author** — Grace-Esther Dong · Aivancity Paris-Cachan
> **Paper** — [`UG_CPPO_paper.pdf`](./UG_CPPO_paper.pdf)
> **Code** — https://github.com/graceesthi/ug_cppo

---

## What's in this repo

3 trained Stable-Baselines3 agents trained on Nasdaq stocks 2013-2018,
evaluated on 2019-2023:

| File | Algorithm | Description |
|------|-----------|-------------|
| `ppo_seed42.zip` | Vanilla PPO | Baseline — no LLM signals |
| `cppo_seed42.zip` | CVaR-PPO | Risk-sensitive, no uncertainty gating |
| `ug_cppo_seed42.zip` | **UG-CPPO** (ours) | Uncertainty-gated CVaR-PPO |

All agents trained with:
- 500,000 timesteps
- Seed = 42
- 10 Nasdaq tickers (AAPL, MSFT, AMZN, NVDA, META, GOOGL, TSLA, NFLX, AMD, COST)
- LLM signals from OpenAI gpt-4o-mini via FNSPID news data

---

## Performance (2019-2023 trade period)

| Model | Cumul. Return | Rachev Ratio | Max Drawdown | Outperf. Bear |
|-------|:-:|:-:|:-:|:-:|
| QQQ Benchmark | 173.26% | — | — | — |
| PPO | 93.21% | 0.9433 | -23.73% | 51.0% |
| CPPO | 54.30% | 0.9242 | **-18.35%** | 49.8% |
| **UG-CPPO** | **103.73%** | 0.9396 | -29.52% | **51.0%** |

UG-CPPO achieves +10.5pp cumulative return over PPO and +49.4pp over CPPO,
while validating 3 of 4 pre-registered hypotheses (paper Section 6).

---

## Quick load

```python
from stable_baselines3 import PPO
from huggingface_hub import hf_hub_download

# Download UG-CPPO agent
path = hf_hub_download(
    repo_id="graceesthi/ug-cppo-finai-2025",
    filename="ug_cppo_seed42.zip"
)
agent = PPO.load(path)
```

For the full reproduction pipeline, see https://github.com/graceesthi/ug_cppo

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

# ─── DATASET CARD ────────────────────────────────────────────────────────
DATASET_CARD = """---
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
used in the UG-CPPO paper (FinAI Contest 2025).

> **Source code** — https://github.com/graceesthi/ug_cppo
> **Paper** — [arXiv link TBA]
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
- Computed at $0.150/1M input tokens (gpt-4o-mini), total cost ~$3.40

---

## Calibration insight

LLM uncertainty σ tracks bear-market regimes:

| Year | Mean σ | Regime |
|------|:-:|--------|
| 2019 | 0.525 | Bull |
| 2020 | 0.581 | COVID crash |
| 2021 | 0.534 | Recovery |
| **2022** | **0.614** | **Bear (Ukraine, Fed)** |
| 2023 | 0.553 | Recovery |

σ in 2022 is 17% higher than 2019, validating that prompt-ensemble σ is a
genuine market-regime indicator (paper Section 6.3).

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
    # STEP 1 — MODEL REPO (trained agents)
    # ─────────────────────────────────────────────────────────────────────
    logger.info(f"\n[1/2] Model repo: {MODEL_REPO}")

    create_repo(MODEL_REPO, repo_type="model", exist_ok=True, private=False)
    logger.info(f"  ✓ Repo ready")

    # Upload README (model card)
    readme_path = Path("/tmp/model_readme.md")
    readme_path.write_text(MODEL_CARD)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=MODEL_REPO, repo_type="model",
        commit_message="Update model card",
    )
    logger.info(f"  ✓ README.md uploaded")

    # Upload trained models
    for mode in ["ppo", "cppo", "ug_cppo"]:
        model_file = MODELS_DIR / f"{mode}_seed42.zip"
        if not model_file.exists():
            logger.warning(f"  ⚠ {model_file} missing — skip")
            continue
        api.upload_file(
            path_or_fileobj=str(model_file),
            path_in_repo=f"{mode}_seed42.zip",
            repo_id=MODEL_REPO, repo_type="model",
            commit_message=f"Upload {mode} agent",
        )
        size_mb = model_file.stat().st_size / 1e6
        logger.info(f"  ✓ {mode}_seed42.zip ({size_mb:.1f} MB)")

    # Upload paper PDF if available
    if PAPER_PDF.exists():
        api.upload_file(
            path_or_fileobj=str(PAPER_PDF),
            path_in_repo="UG_CPPO_paper.pdf",
            repo_id=MODEL_REPO, repo_type="model",
            commit_message="Upload paper PDF",
        )
        logger.info(f"  ✓ UG_CPPO_paper.pdf")

    # Upload final report JSON if available
    final_report = PROJECT_ROOT / "results" / "final_report.json"
    if final_report.exists():
        api.upload_file(
            path_or_fileobj=str(final_report),
            path_in_repo="final_report.json",
            repo_id=MODEL_REPO, repo_type="model",
        )
        logger.info(f"  ✓ final_report.json")

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
    readme_path = Path("/tmp/dataset_readme.md")
    readme_path.write_text(DATASET_CARD)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=DATASET_REPO, repo_type="dataset",
        commit_message="Update dataset card",
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
    logger.info("\n" + "="*60)
    logger.info(" UPLOAD COMPLETE ✓")
    logger.info("="*60)
    logger.info(f" Model:   https://huggingface.co/{MODEL_REPO}")
    logger.info(f" Signals: https://huggingface.co/datasets/{DATASET_REPO}")
    logger.info("="*60)
    logger.info(" Next: vérifier les 2 URLs dans le navigateur, puis")
    logger.info("       les liens fonctionnent dans ton README.md GitHub")


if __name__ == "__main__":
    main()
