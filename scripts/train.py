"""
train.py
========
Main training script for UG-CPPO.

Usage:
    python scripts/train.py --mode ug_cppo --seed 42
    python scripts/train.py --mode ppo     --seed 42   # baseline
    python scripts/train.py --mode cppo    --seed 42   # CVaR only, no UQ

All 3 modes use identical hyperparameters for fair comparison.
Results saved to results/models/ and results/logs/.
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import load_ohlcv, load_signals, split_data, add_technical_indicators
from src.ug_cppo_env import UGCPPOTradingEnv
from src.cvar_ppo import build_agent
from src.evaluation import evaluate, print_report, load_benchmark, compare_models
from src.uncertainty_llm import LLMConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TECH_INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30",
                   "close_30_sma", "close_60_sma"]


def parse_args():
    p = argparse.ArgumentParser(description="Train UG-CPPO trading agent")
    p.add_argument("--config",    default="configs/default.yaml")
    p.add_argument("--mode",      default="ug_cppo",
                   choices=["ug_cppo", "cppo", "ppo"])
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--timesteps", type=int, default=None,
                   help="Override config total_timesteps")
    p.add_argument("--signals",   default=None,
                   help="Path to pre-computed signals parquet")
    p.add_argument("--no-cache",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    total_steps = args.timesteps or cfg["training"]["total_timesteps"]
    signals_path = args.signals or cfg["data"]["signals_path"]

    logger.info(f"Mode: {args.mode.upper()} | Seed: {args.seed} | Steps: {total_steps:,}")

    # ── Load data ────────────────────────────────────────────────────────
    logger.info("Loading OHLCV data...")
    df_raw = load_ohlcv(
        start=cfg["data"]["train_start"],
        end=cfg["data"]["trade_end"],
        cache_path=cfg["data"]["ohlcv_path"],
    )
    df = add_technical_indicators(df_raw)
    tickers = sorted(df["tic"].unique().tolist())
    logger.info(f"Loaded {len(tickers)} tickers, {len(df)} rows")

    train_df, trade_df = split_data(
        df,
        train_start=cfg["data"]["train_start"],
        train_end=cfg["data"]["train_end"],
        trade_start=cfg["data"]["trade_start"],
        trade_end=cfg["data"]["trade_end"],
    )

    # ── Load signals ─────────────────────────────────────────────────────
    signal_df = None
    if args.mode in ("ug_cppo", "cppo") and Path(signals_path).exists():
        logger.info(f"Loading pre-computed signals from {signals_path}")
        signal_df = load_signals(signals_path)
        logger.info(f"Loaded {len(signal_df)} signals")
    elif args.mode in ("ug_cppo", "cppo"):
        logger.warning(f"Signals not found at {signals_path}. Run scripts/precompute_signals.py first.")
        logger.warning("Falling back to mode=ppo (no LLM infusion)")
        args.mode = "ppo"

    # ── Build training environment ────────────────────────────────────────
    env_kwargs = dict(
        tickers=tickers,
        signal_df=signal_df,
        initial_amount=cfg["env"]["initial_amount"],
        hmax=cfg["env"]["hmax"],
        transaction_cost=cfg["env"]["transaction_cost"],
        reward_scaling=cfg["env"]["reward_scaling"],
        alpha=cfg["agent"]["infusion_strength"],
        threshold_tau=cfg["uncertainty"]["threshold_tau"],
        tech_indicator_list=TECH_INDICATORS,
        mode=args.mode,
    )
    train_env = UGCPPOTradingEnv(df=train_df, **env_kwargs)

    # ── Build agent ───────────────────────────────────────────────────────
    log_dir = Path(cfg["training"]["tensorboard_dir"]) / args.mode / f"seed_{args.seed}"
    model_dir = Path(cfg["training"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    agent = build_agent(
        env=train_env,
        mode=args.mode,
        learning_rate=cfg["agent"]["learning_rate"],
        n_steps=cfg["agent"]["n_steps"],
        batch_size=cfg["agent"]["batch_size"],
        n_epochs=cfg["agent"]["n_epochs"],
        gamma=cfg["agent"]["gamma"],
        gae_lambda=cfg["agent"]["gae_lambda"],
        clip_range=cfg["agent"]["clip_range"],
        ent_coef=cfg["agent"]["ent_coef"],
        vf_coef=cfg["agent"]["vf_coef"],
        cvar_alpha=cfg["agent"]["cvar_alpha"],
        cvar_lambda=cfg["agent"]["cvar_lambda"],
        cvar_beta=cfg["agent"]["cvar_beta"],
        tensorboard_log=str(log_dir),
        seed=args.seed,
        verbose=1,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info(f"Training {args.mode.upper()} for {total_steps:,} steps...")
    agent.learn(
        total_timesteps=total_steps,
        progress_bar=True,
    )

    # ── Save model ────────────────────────────────────────────────────────
    model_path = model_dir / f"{args.mode}_seed{args.seed}"
    agent.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # ── Evaluate on trade period ──────────────────────────────────────────
    logger.info("Evaluating on trade period 2019–2023...")
    trade_env = UGCPPOTradingEnv(df=trade_df, **env_kwargs)
    obs, _ = trade_env.reset()
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, done, _, _ = trade_env.step(action)

    portfolio_values = trade_env.portfolio_value_history
    benchmark_values, bench_dates = load_benchmark(
        start=cfg["data"]["trade_start"],
        end=cfg["data"]["trade_end"],
    )
    report = evaluate(
        portfolio_values=portfolio_values,
        benchmark_values=benchmark_values,
        dates=bench_dates,
        model_name=f"{args.mode}_seed{args.seed}",
    )
    print_report(report)

    # Save report
    import json
    rpt_path = model_dir / f"{args.mode}_seed{args.seed}_report.json"
    with open(rpt_path, "w") as f:
        # Flatten outperformance_frequency for JSON serialisation
        r = dict(report)
        if isinstance(r.get("outperformance_frequency"), dict):
            for k, v in r["outperformance_frequency"].items():
                r[f"outperf_{k}"] = v
            del r["outperformance_frequency"]
        json.dump(r, f, indent=2)
    logger.info(f"Report saved to {rpt_path}")


if __name__ == "__main__":
    main()
