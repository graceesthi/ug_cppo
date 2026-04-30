"""
ablation.py
===========
Full factorial ablation study for UG-CPPO.

Varies:
  - N (ensemble size): [1, 3, 5]
  - tau (gate threshold): [0.0, 0.2, 0.3, 0.5]
  - alpha (infusion strength): [0.01, 0.05, 0.10]
  - uq_on_rf_only: [True, False]

Each configuration is trained for 500k steps (abbreviated).
Results saved to results/ablation/ablation_results.csv

Usage:
    python scripts/ablation.py --mock --n-tickers 5
    python scripts/ablation.py  # full run (needs API key + data)
"""

from __future__ import annotations
import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ABLATION_GRID = {
    "n_prompts":         [1, 3, 5],
    "threshold_tau":     [0.0, 0.2, 0.3, 0.5],
    "infusion_strength": [0.01, 0.05, 0.10],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    default="configs/default.yaml")
    p.add_argument("--mock",      action="store_true")
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--output",    default="results/ablation")
    return p.parse_args()


def run_single_config(
    cfg,
    n_prompts:         int,
    threshold_tau:     float,
    infusion_strength: float,
    total_steps:       int,
    seed:              int,
    use_mock:          bool = False,
) -> dict:
    """Run one ablation configuration, return evaluation report."""
    from src.data_pipeline import load_ohlcv, load_signals, split_data, add_technical_indicators
    from src.ug_cppo_env import UGCPPOTradingEnv
    from src.cvar_ppo import build_agent
    from src.evaluation import evaluate, load_benchmark
    from src.uncertainty_llm import LLMConfig

    TECH = ["macd", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]

    # Data (reuse cached)
    df_raw = load_ohlcv(cache_path=cfg["data"]["ohlcv_path"])
    df     = add_technical_indicators(df_raw)
    tickers = sorted(df["tic"].unique().tolist())
    _, trade_df = split_data(df,
        train_start=cfg["data"]["train_start"],
        train_end=cfg["data"]["train_end"],
        trade_start=cfg["data"]["trade_start"],
        trade_end=cfg["data"]["trade_end"],
    )

    # Load signals (pre-computed with full N=5 prompts)
    signal_df = None
    if Path(cfg["data"]["signals_path"]).exists():
        signal_df = load_signals(cfg["data"]["signals_path"])

    # Build env with this config
    env_kwargs = dict(
        tickers=tickers, signal_df=signal_df,
        initial_amount=cfg["env"]["initial_amount"],
        hmax=cfg["env"]["hmax"],
        transaction_cost=cfg["env"]["transaction_cost"],
        reward_scaling=cfg["env"]["reward_scaling"],
        alpha=infusion_strength,
        threshold_tau=threshold_tau,
        tech_indicator_list=TECH,
        mode="ug_cppo",
    )
    from src.data_pipeline import split_data as sd
    df_raw2 = load_ohlcv(cache_path=cfg["data"]["ohlcv_path"])
    df2     = add_technical_indicators(df_raw2)
    train_df2, _ = sd(df2,
        train_start=cfg["data"]["train_start"],
        train_end=cfg["data"]["train_end"],
        trade_start=cfg["data"]["trade_start"],
        trade_end=cfg["data"]["trade_end"],
    )

    train_env = UGCPPOTradingEnv(df=train_df2, **env_kwargs)
    agent = build_agent(
        env=train_env, mode="ug_cppo",
        learning_rate=cfg["agent"]["learning_rate"],
        n_steps=min(cfg["agent"]["n_steps"], 512),  # smaller for ablation speed
        batch_size=cfg["agent"]["batch_size"],
        n_epochs=cfg["agent"]["n_epochs"],
        gamma=cfg["agent"]["gamma"],
        gae_lambda=cfg["agent"]["gae_lambda"],
        clip_range=cfg["agent"]["clip_range"],
        cvar_alpha=cfg["agent"]["cvar_alpha"],
        cvar_lambda=cfg["agent"]["cvar_lambda"],
        seed=seed, verbose=0,
    )
    agent.learn(total_timesteps=total_steps, progress_bar=False)

    # Evaluate on trade period
    trade_env = UGCPPOTradingEnv(df=trade_df, **env_kwargs)
    obs, _ = trade_env.reset()
    done   = False
    while not done:
        act, _ = agent.predict(obs, deterministic=True)
        obs, _, done, _, _ = trade_env.step(act)

    bv, bd = load_benchmark(
        start=cfg["data"]["trade_start"],
        end=cfg["data"]["trade_end"],
    )
    report = evaluate(
        portfolio_values=trade_env.portfolio_value_history,
        benchmark_values=bv, dates=bd,
        model_name=f"N={n_prompts}_tau={threshold_tau}_alpha={infusion_strength}",
    )
    report.update({
        "n_prompts": n_prompts,
        "threshold_tau": threshold_tau,
        "infusion_strength": infusion_strength,
        "seed": seed,
    })
    return report


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = list(itertools.product(
        ABLATION_GRID["n_prompts"],
        ABLATION_GRID["threshold_tau"],
        ABLATION_GRID["infusion_strength"],
    ))
    total = len(configs)
    logger.info(f"Running {total} ablation configurations × seed {args.seed}")

    results = []
    for i, (n, tau, alpha) in enumerate(configs):
        label = f"[{i+1}/{total}] N={n} tau={tau} alpha={alpha}"
        logger.info(f"Running {label}")
        try:
            report = run_single_config(
                cfg=cfg,
                n_prompts=n,
                threshold_tau=tau,
                infusion_strength=alpha,
                total_steps=args.timesteps,
                seed=args.seed,
                use_mock=args.mock,
            )
            results.append(report)
            # Save checkpoint after each run
            df_results = _reports_to_df(results)
            df_results.to_csv(output_dir / "ablation_results.csv", index=False)
            logger.info(f"  → Cumret: {report.get('cumulative_return',0):.2%}  "
                        f"Rachev: {report.get('rachev_ratio',0):.4f}  "
                        f"MDD: {report.get('max_drawdown',0):.2%}")
        except Exception as e:
            logger.error(f"  FAILED: {e}")

    df_final = _reports_to_df(results)
    df_final.to_csv(output_dir / "ablation_results.csv", index=False)
    logger.info(f"\nAblation complete. Results: {output_dir / 'ablation_results.csv'}")
    print("\nTop 5 configs by Rachev Ratio:")
    print(df_final.nlargest(5, "rachev_ratio")[
        ["n_prompts", "threshold_tau", "infusion_strength",
         "cumulative_return", "rachev_ratio", "max_drawdown"]
    ].to_string(index=False))


def _reports_to_df(reports: list) -> pd.DataFrame:
    rows = []
    for r in reports:
        opf = r.get("outperformance_frequency", {})
        rows.append({
            "model":               r.get("model", ""),
            "n_prompts":           r.get("n_prompts"),
            "threshold_tau":       r.get("threshold_tau"),
            "infusion_strength":   r.get("infusion_strength"),
            "cumulative_return":   r.get("cumulative_return"),
            "rachev_ratio":        r.get("rachev_ratio"),
            "max_drawdown":        r.get("max_drawdown"),
            "cvar":                r.get("cvar"),
            "information_ratio":   r.get("information_ratio"),
            "outperf_overall":     opf.get("overall"),
            "outperf_bear":        opf.get("bear"),
            "seed":                r.get("seed"),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
