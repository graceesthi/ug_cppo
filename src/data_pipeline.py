"""
data_pipeline.py
================
Data loading, preprocessing, and LLM signal pre-computation.

Workflow:
  1. load_ohlcv()       — download/load Nasdaq-100 OHLCV data
  2. load_fnspid()      — load filtered FNSPID news data
  3. precompute_signals()— batch query LLM ensemble, store results
  4. load_signals()     — fast load of pre-computed signals for training
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

NASDAQ100_TICKERS = [
    "AAPL","MSFT","AMZN","NVDA","META","GOOGL","GOOG","TSLA","AVGO","ASML",
    "COST","NFLX","AMD","AZN","CSCO","ADBE","PEP","QCOM","TMUS","TXN",
    "AMGN","INTU","INTC","CMCSA","AMAT","HON","BKNG","VRTX","LRCX","REGN",
    "MDLZ","ADI","KLAC","PANW","ISRG","GILD","SNPS","CDNS","MRVL","NXPI",
    "MU","CRWD","ABNB","MELI","FTNT","MAR","KDP","ORLY","ROST","PCAR",
    "CTAS","AEP","MNST","WDAY","DXCM","FAST","ODFL","BIIB","PAYX","IDXX",
    "EXC","XEL","FANG","CTSH","VRSK","KHC","DLTR","GEHC","ON","WBD",
    "EA","ZS","SIRI","ANSS","ILMN","ALGN","DDOG","TEAM","ZM","LCID",
]

# ─── OHLCV DATA ───────────────────────────────────────────────────────────────

def load_ohlcv(
    tickers:    List[str] = NASDAQ100_TICKERS,
    start:      str = "2013-01-01",
    end:        str = "2023-12-31",
    cache_path: Optional[str] = "data/ohlcv_nasdaq100.parquet",
) -> pd.DataFrame:
    """
    Load OHLCV data for Nasdaq-100 tickers.
    Downloads via yfinance, caches to parquet.

    Returns
    -------
    DataFrame with columns: date, tic, open, high, low, close, volume
    Sorted by (date, tic) — FinRL StockTradingEnv format.
    """
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading OHLCV from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Downloading OHLCV for {len(tickers)} tickers: {start} → {end}")
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    dfs = []
    for tic in tqdm(tickers, desc="Downloading OHLCV"):
        try:
            raw = yf.download(tic, start=start, end=end,
                              auto_adjust=True, progress=False)
            if raw.empty:
                continue
            raw = raw.reset_index()
            raw.columns = raw.columns.get_level_values(0)
            df = pd.DataFrame({
                "date":   raw["Date"].dt.strftime("%Y-%m-%d"),
                "tic":    tic,
                "open":   raw["Open"].values,
                "high":   raw["High"].values,
                "low":    raw["Low"].values,
                "close":  raw["Close"].values,
                "volume": raw["Volume"].values,
            })
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to download {tic}: {e}")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.dropna().sort_values(["date", "tic"]).reset_index(drop=True)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df_all.to_parquet(cache_path, index=False)
        logger.info(f"Saved {len(df_all)} OHLCV rows to {cache_path}")

    return df_all


# ─── FNSPID DATA ──────────────────────────────────────────────────────────────

def load_fnspid(
    fnspid_path: str,
    tickers:     List[str] = NASDAQ100_TICKERS,
    start:       str = "2013-01-01",
    end:         str = "2023-12-31",
    one_per_day: bool = True,
    cache_path:  Optional[str] = "data/fnspid_filtered.parquet",
) -> pd.DataFrame:
    """
    Load and filter FNSPID dataset.

    FNSPID schema: ticker, date, article_title, article_content, ...
    We reduce to 1 article/stock/day to match the baseline.

    Returns
    -------
    DataFrame with columns: ticker, date, news_text
    """
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading FNSPID from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Loading FNSPID from {fnspid_path}")
    df = pd.read_parquet(fnspid_path) if fnspid_path.endswith(".parquet") \
         else pd.read_csv(fnspid_path)

    # Normalise column names (FNSPID schema varies)
    col_map = {}
    for col in df.columns:
        if "ticker" in col.lower():       col_map[col] = "ticker"
        elif "date" in col.lower():       col_map[col] = "date"
        elif "content" in col.lower():    col_map[col] = "news_text"
        elif "title" in col.lower():      col_map[col] = "news_title"
    df = df.rename(columns=col_map)

    if "news_text" not in df.columns and "news_title" in df.columns:
        df["news_text"] = df["news_title"]

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[df["ticker"].isin(tickers)]
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df = df[["ticker", "date", "news_text"]].dropna()

    if one_per_day:
        # Keep first article per (ticker, date) — matches baseline
        df = df.groupby(["ticker", "date"]).first().reset_index()

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info(f"Saved {len(df)} FNSPID rows to {cache_path}")

    return df


# ─── SIGNAL PRECOMPUTATION ────────────────────────────────────────────────────

def precompute_signals(
    fnspid_df:       pd.DataFrame,
    llm_config,                         # LLMConfig
    n_prompts:       int   = 5,
    n_risk_prompts:  int   = 4,
    threshold_tau:   float = 0.30,
    alpha:           float = 0.10,
    output_path:     str   = "data/ug_signals.parquet",
    resume:          bool  = True,
    batch_size:      int   = 100,
) -> pd.DataFrame:
    """
    Pre-compute all LLM uncertainty signals offline.
    
    This is the EXPENSIVE step — called once before training.
    Approximately 5 API calls per (ticker, date) pair.
    
    With 2M records → 10M LLM calls (use batching + caching).
    For contests: use a subset of high-liquidity tickers.

    Returns
    -------
    DataFrame indexed by (ticker, date) with columns:
        mean_score, std_score, confidence,
        mean_risk, std_risk, risk_confidence,
        gate_fired
    """
    from src.uncertainty_llm import UncertaintyAwareLLM, LLMConfig

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    existing_keys = set()
    if resume and output.exists():
        existing = pd.read_parquet(output)
        existing_keys = set(zip(existing["ticker"], existing["date"]))
        logger.info(f"Resuming: {len(existing_keys)} signals already computed")

    llm = UncertaintyAwareLLM(
        config=llm_config,
        n_prompts=n_prompts,
        n_risk_prompts=n_risk_prompts,
        alpha=alpha,
        threshold_tau=threshold_tau,
    )

    rows_to_process = fnspid_df[
        ~fnspid_df.apply(lambda r: (r["ticker"], r["date"]) in existing_keys, axis=1)
    ]
    logger.info(f"Computing signals for {len(rows_to_process)} (ticker, date) pairs")

    results = []
    for i, (_, row) in enumerate(tqdm(rows_to_process.iterrows(),
                                       total=len(rows_to_process),
                                       desc="Pre-computing LLM signals")):
        sig = llm.get_signal(
            ticker=row["ticker"],
            news=row["news_text"],
            action=0.0,   # action=0 for pre-computation (Sf computed per-step later)
            date=row["date"],
        )
        results.append({
            "ticker":          sig.ticker,
            "date":            sig.date,
            "mean_score":      sig.mean_score,
            "std_score":       sig.std_score,
            "confidence":      sig.confidence,
            "mean_risk":       sig.mean_risk,
            "std_risk":        sig.std_risk,
            "risk_confidence": sig.risk_confidence,
            "gate_fired":      sig.gate_fired,
        })

        # Checkpoint every batch_size records
        if (i + 1) % batch_size == 0:
            _checkpoint(results, existing_keys, output)
            results = []

    _checkpoint(results, existing_keys, output)

    final = pd.read_parquet(output)
    logger.info(f"Total signals: {len(final)}")
    logger.info(llm.calibration_report())
    return final


def _checkpoint(new_rows: List[dict], existing_keys: set, output: Path) -> None:
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows)
    if output.exists():
        existing = pd.read_parquet(output)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    else:
        combined = new_df
    combined.to_parquet(output, index=False)


def load_signals(path: str = "data/ug_signals.parquet") -> pd.DataFrame:
    """
    Load pre-computed signals as a (ticker, date)-indexed DataFrame.
    Used by UG-CPPO env at each trading step — O(1) lookup.
    """
    df = pd.read_parquet(path)
    df = df.set_index(["ticker", "date"])
    return df


# ─── TRAIN/TRADE SPLIT ────────────────────────────────────────────────────────

def split_data(
    df:          pd.DataFrame,
    train_start: str = "2013-01-01",
    train_end:   str = "2018-12-31",
    trade_start: str = "2019-01-01",
    trade_end:   str = "2023-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split OHLCV DataFrame into train and trade sets."""
    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
    trade = df[(df["date"] >= trade_start) & (df["date"] <= trade_end)].copy()
    logger.info(f"Train: {len(train)} rows ({train['date'].min()} → {train['date'].max()})")
    logger.info(f"Trade: {len(trade)} rows ({trade['date'].min()} → {trade['date'].max()})")
    return train, trade


# ─── ADD TECHNICAL INDICATORS ────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators used by FinRL's StockTradingEnv.
    MACD, RSI, Bollinger Bands, CCI, DX.
    """
    try:
        from stockstats import StockDataFrame as Sdf
    except ImportError:
        logger.warning("stockstats not installed — skipping technical indicators")
        return df

    processed = []
    for tic, group in df.groupby("tic"):
        stock = Sdf.retype(group.copy())
        try:
            group["macd"]  = stock["macd"]
            group["rsi_30"]= stock["rsi_30"]
            group["cci_30"]= stock["cci_30"]
            group["dx_30"] = stock["dx_30"]
            group["close_30_sma"] = stock["close_30_sma"]
            group["close_60_sma"] = stock["close_60_sma"]
        except Exception:
            pass
        processed.append(group)

    result = pd.concat(processed).sort_values(["date", "tic"]).reset_index(drop=True)
    return result.fillna(0.0)
