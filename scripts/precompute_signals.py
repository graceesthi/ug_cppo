"""
precompute_signals.py
=====================
Pre-compute all LLM uncertainty signals from FNSPID.

Run ONCE before training. Results cached to data/ug_signals.parquet.
Supports resuming from checkpoint.

Usage:
    # With real DeepSeek API:
    DEEPSEEK_API_KEY=sk-xxx python scripts/precompute_signals.py

    # With mock LLM (for testing):
    python scripts/precompute_signals.py --mock

    # For a subset of tickers only:
    python scripts/precompute_signals.py --n-tickers 10 --mock
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import load_fnspid, precompute_signals
from src.uncertainty_llm import LLMConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--mock",       action="store_true",
                   help="Use mock LLM (no API key needed, for testing)")
    p.add_argument("--n-tickers",  type=int, default=None,
                   help="Limit to N tickers (for quick tests)")
    p.add_argument("--n-prompts",  type=int, default=5)
    p.add_argument("--tau",        type=float, default=0.30)
    p.add_argument("--output",     default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # LLM config
    llm_config = LLMConfig(
        provider="mock" if args.mock else cfg["llm"]["provider"],
        model=cfg["llm"]["model"],
        api_base=cfg["llm"].get("api_base", "https://api.deepseek.com"),
        max_retries=cfg["llm"]["max_retries"],
        retry_delay=cfg["llm"]["retry_delay"],
    )

    # Load FNSPID
    logger.info("Loading FNSPID news data...")
    fnspid_df = load_fnspid(
        fnspid_path=cfg["data"]["fnspid_path"],
        start=cfg["data"]["train_start"],
        end=cfg["data"]["trade_end"],
        cache_path=cfg["data"]["fnspid_path"].replace(".parquet", "_filtered.parquet"),
    )

    if args.n_tickers:
        top_tickers = fnspid_df["ticker"].value_counts().head(args.n_tickers).index.tolist()
        fnspid_df   = fnspid_df[fnspid_df["ticker"].isin(top_tickers)]
        logger.info(f"Subset to {args.n_tickers} tickers: {top_tickers[:5]}...")

    logger.info(f"Computing signals for {len(fnspid_df):,} (ticker, date) pairs")
    logger.info(f"LLM: {llm_config.provider} / {llm_config.model}")
    logger.info(f"N prompts: {args.n_prompts}  |  Tau: {args.tau}")
    logger.info(f"Estimated API calls: {len(fnspid_df) * (args.n_prompts + 4):,}")

    output = args.output or cfg["data"]["signals_path"]
    precompute_signals(
        fnspid_df=fnspid_df,
        llm_config=llm_config,
        n_prompts=args.n_prompts,
        threshold_tau=args.tau,
        output_path=output,
        resume=True,
        batch_size=100,
    )
    logger.info(f"Done. Signals saved to: {output}")


if __name__ == "__main__":
    main()
