"""
evaluation.py
=============
Contest evaluation metrics for UG-CPPO.

Metrics:
  1. Cumulative return
  2. Rachev ratio  (ETG/ETL at 5% confidence)
  3. Maximum drawdown
  4. Outperformance frequency vs. Nasdaq-100 (especially in downturns)

Plus: Information ratio, CVaR, calibration stats.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Nasdaq-100 ticker is ^NDX — we use QQQ as a proxy
BENCHMARK_TICKER = "QQQ"


# ─── CORE METRICS ─────────────────────────────────────────────────────────────

def cumulative_return(portfolio_values: List[float]) -> float:
    """Total return = (final - initial) / initial."""
    if len(portfolio_values) < 2:
        return 0.0
    return (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]


def daily_returns(portfolio_values: List[float]) -> np.ndarray:
    pv = np.array(portfolio_values, dtype=float)
    return np.diff(pv) / pv[:-1]


def rachev_ratio(
    portfolio_values: List[float],
    alpha: float = 0.05,
) -> float:
    """
    Rachev Ratio = ETG(α) / ETL(α)

    ETG(α) = Expected Tail Gain = mean of top α% returns
    ETL(α) = Expected Tail Loss = mean of bottom α% losses (absolute)

    Higher is better. Measures extreme upside potential vs. extreme downside risk.
    """
    rets = daily_returns(portfolio_values)
    if len(rets) < 20:
        return float("nan")
    threshold_gain = np.percentile(rets, (1 - alpha) * 100)
    threshold_loss = np.percentile(rets, alpha * 100)
    etg = rets[rets >= threshold_gain].mean()
    etl = abs(rets[rets <= threshold_loss].mean())
    if etl < 1e-10:
        return float("inf")
    return float(etg / etl)


def max_drawdown(portfolio_values: List[float]) -> float:
    """
    Maximum drawdown = max (peak - trough) / peak over all time windows.
    Returns negative value (e.g. -0.25 = 25% drawdown).
    """
    pv     = np.array(portfolio_values, dtype=float)
    peak   = np.maximum.accumulate(pv)
    dd     = (pv - peak) / peak
    return float(dd.min())


def cvar(portfolio_values: List[float], alpha: float = 0.05) -> float:
    """
    CVaR at confidence level α: mean of worst α% daily returns.
    Returns negative value.
    """
    rets = daily_returns(portfolio_values)
    if len(rets) == 0:
        return float("nan")
    threshold = np.percentile(rets, alpha * 100)
    tail      = rets[rets <= threshold]
    return float(tail.mean()) if len(tail) else float("nan")


def information_ratio(
    portfolio_values: List[float],
    benchmark_values: List[float],
) -> float:
    """
    IR = mean(excess_return) / std(excess_return)
    Measures risk-adjusted outperformance vs. benchmark.
    """
    pr = daily_returns(portfolio_values)
    br = daily_returns(benchmark_values)
    n  = min(len(pr), len(br))
    if n < 5:
        return float("nan")
    excess = pr[:n] - br[:n]
    std    = excess.std()
    if std < 1e-10:
        return 0.0
    return float(excess.mean() / std)


def outperformance_frequency(
    portfolio_values:   List[float],
    benchmark_values:   List[float],
    dates:              Optional[List[str]] = None,
    downturn_start:     str = "2022-01-01",
    downturn_end:       str = "2023-12-31",
) -> Dict[str, float]:
    """
    Fraction of days where agent daily return > benchmark daily return.

    Returns
    -------
    {
        "overall":       float  — full period
        "bull":          float  — before downturn_start
        "bear":          float  — during downturn
    }
    """
    pr = np.array(daily_returns(portfolio_values))
    br = np.array(daily_returns(benchmark_values))
    n  = min(len(pr), len(br))
    pr, br = pr[:n], br[:n]
    outperf = (pr > br).astype(float)

    result: Dict[str, float] = {"overall": float(outperf.mean())}

    if dates and len(dates) >= n + 1:
        d = np.array(dates[:n])
        bull_mask = d < downturn_start
        bear_mask = (d >= downturn_start) & (d <= downturn_end)
        if bull_mask.sum() > 0:
            result["bull"] = float(outperf[bull_mask].mean())
        if bear_mask.sum() > 0:
            result["bear"] = float(outperf[bear_mask].mean())

    return result


# ─── FULL EVALUATION REPORT ───────────────────────────────────────────────────

def evaluate(
    portfolio_values:  List[float],
    benchmark_values:  Optional[List[float]] = None,
    dates:             Optional[List[str]]   = None,
    model_name:        str = "model",
    alpha:             float = 0.05,
) -> Dict:
    """
    Full evaluation report for the contest.

    Returns
    -------
    Dict with all 4 contest metrics + extras.
    """
    pv = portfolio_values

    report: Dict = {
        "model":             model_name,
        # Contest metrics
        "cumulative_return": cumulative_return(pv),
        "rachev_ratio":      rachev_ratio(pv, alpha),
        "max_drawdown":      max_drawdown(pv),
        # Extra
        "cvar":              cvar(pv, alpha),
        "final_value":       pv[-1] if pv else float("nan"),
        "n_days":            len(pv) - 1,
    }

    if benchmark_values:
        bv = benchmark_values
        report["information_ratio"]        = information_ratio(pv, bv)
        report["benchmark_cumret"]         = cumulative_return(bv)
        report["outperformance_frequency"] = outperformance_frequency(
            pv, bv, dates
        )

    return report


def print_report(report: Dict) -> None:
    """Pretty-print evaluation report."""
    print(f"\n{'='*60}")
    print(f"  {report.get('model', 'Model').upper()} — EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"  Cumulative Return      : {report.get('cumulative_return', 0):.2%}")
    print(f"  Rachev Ratio           : {report.get('rachev_ratio', 0):.4f}")
    print(f"  Max Drawdown           : {report.get('max_drawdown', 0):.2%}")
    print(f"  CVaR (5%)              : {report.get('cvar', 0):.4f}")
    if "information_ratio" in report:
        print(f"  Information Ratio      : {report['information_ratio']:.4f}")
    if "benchmark_cumret" in report:
        print(f"  Benchmark (QQQ) Return : {report['benchmark_cumret']:.2%}")
    if "outperformance_frequency" in report:
        opf = report["outperformance_frequency"]
        print(f"  Outperf. Freq. Overall : {opf.get('overall', 0):.2%}")
        if "bull" in opf:
            print(f"  Outperf. Freq. Bull    : {opf.get('bull', 0):.2%}")
        if "bear" in opf:
            print(f"  Outperf. Freq. Bear    : {opf.get('bear', 0):.2%}")
    print(f"  Final Portfolio Value  : ${report.get('final_value', 0):,.0f}")
    print(f"{'='*60}\n")


def compare_models(reports: List[Dict]) -> pd.DataFrame:
    """Build comparison DataFrame for the paper's results table."""
    rows = []
    for r in reports:
        opf = r.get("outperformance_frequency", {})
        rows.append({
            "Model":            r.get("model", "—"),
            "Cumul. Return":    f"{r.get('cumulative_return', 0):.2%}",
            "Rachev Ratio":     f"{r.get('rachev_ratio', 0):.4f}",
            "Max Drawdown":     f"{r.get('max_drawdown', 0):.2%}",
            "CVaR (5%)":        f"{r.get('cvar', 0):.4f}",
            "IR":               f"{r.get('information_ratio', 0):.4f}",
            "Outperf. Overall": f"{opf.get('overall', 0):.2%}",
            "Outperf. Bear":    f"{opf.get('bear', 0):.2%}",
        })
    return pd.DataFrame(rows)


# ─── BENCHMARK LOADER ────────────────────────────────────────────────────────

def load_benchmark(
    start: str = "2019-01-01",
    end:   str = "2023-12-31",
    ticker: str = BENCHMARK_TICKER,
    initial_value: float = 1_000_000.0,
) -> Tuple[List[float], List[str]]:
    """
    Load Nasdaq-100 benchmark (QQQ) portfolio values.
    Returns (portfolio_values, dates).
    """
    try:
        import yfinance as yf
        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False)
        prices = raw["Close"].values.flatten()
        dates  = raw.index.strftime("%Y-%m-%d").tolist()
        values = list(initial_value * prices / prices[0])
        return values, dates
    except Exception as e:
        logger.warning(f"Could not load benchmark {ticker}: {e}")
        return [initial_value], ["2019-01-01"]
