"""
uncertainty_llm.py
==================
Uncertainty-Gated LLM Signal Generation for UG-CPPO.

Providers supported:
  - "openai"    : OpenAI API (gpt-4o-mini recommended)
  - "anthropic" : Anthropic API (claude-haiku-4-5-20251001 recommended)
  - "mock"      : keyword heuristics, zero cost, for pipeline testing

Core innovation:
  Instead of a single LLM score (deterministic, no confidence info),
  we query with N semantically diverse prompts → get a distribution of scores
  → compute epistemic uncertainty σ → gate infusion proportionally to c(σ).

  c(σ) = max(0, 1 - σ/σ_max)          # confidence gate ∈ [0,1]
  Sf   = 1 + α·δ·c(σ)·m(μ)            # gated action modifier
  Rf   = 1 + (Rf_base-1)·c(σ_risk)    # gated CVaR modifier
"""

from __future__ import annotations
import os
import re
import time
import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SIGMA_MAX = float(np.std([1.0, 2.0, 3.0, 4.0, 5.0]))  # ≈ 1.4142


# ─── CONFIG ───────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """
    LLM provider configuration.

    Providers:
      "openai"    → OpenAI API  (gpt-4o-mini recommended, ~$0.15/1M tokens)
      "anthropic" → Anthropic API (claude-haiku-4-5-20251001, fast + cheap)
      "mock"      → keyword heuristic, free, for testing

    Keys loaded from env vars: OPENAI_API_KEY | ANTHROPIC_API_KEY
    Or set directly: LLMConfig(provider="openai", api_key="sk-...")
    """
    provider:    str            = "openai"
    model:       str            = "gpt-4o-mini"
    api_base:    str            = "https://api.openai.com/v1"
    api_key:     Optional[str]  = None
    max_retries: int            = 3
    retry_delay: float          = 2.0
    timeout:     int            = 30
    max_tokens:  int            = 8

    def get_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        env_vars = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
        key = os.environ.get(env_vars.get(self.provider, "LLM_API_KEY"), "")
        if not key and self.provider != "mock":
            logger.warning(
                f"No API key for provider='{self.provider}'. "
                f"Set env var: {env_vars.get(self.provider, 'LLM_API_KEY')}"
            )
        return key

    # ── Convenience constructors ──────────────────────────────────────────────
    @classmethod
    def openai(cls, model: str = "gpt-4o-mini", **kw) -> "LLMConfig":
        """OpenAI API — gpt-4o-mini is best cost/speed tradeoff."""
        return cls(provider="openai", model=model,
                   api_base="https://api.openai.com/v1", **kw)

    @classmethod
    def anthropic(cls, model: str = "claude-haiku-4-5-20251001", **kw) -> "LLMConfig":
        """Anthropic API — Haiku is fast and cost-efficient for scoring."""
        return cls(provider="anthropic", model=model, **kw)

    @classmethod
    def mock(cls) -> "LLMConfig":
        """No API calls — keyword heuristic for pipeline testing."""
        return cls(provider="mock", model="mock")


# ─── SIGNAL DATA CLASS ────────────────────────────────────────────────────────

@dataclass
class UncertainSignal:
    """A trading signal with associated epistemic uncertainty."""
    mean_score:      float
    std_score:       float
    confidence:      float           # c(σ) ∈ [0,1]
    raw_scores:      List[float]
    calibrated_sf:   float           # action modifier Sf ∈ [1-α, 1+α]
    mean_risk:       float = 3.0
    std_risk:        float = 0.0
    risk_confidence: float = 1.0
    raw_risk_scores: List[float] = field(default_factory=list)
    calibrated_rf:   float = 1.0    # CVaR modifier Rf
    gate_fired:      bool  = False
    ticker:          str   = ""
    date:            str   = ""

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker, "date": self.date,
            "mean_score": self.mean_score, "std_score": self.std_score,
            "confidence": self.confidence, "gate_fired": self.gate_fired,
            "calibrated_sf": self.calibrated_sf,
            "mean_risk": self.mean_risk, "std_risk": self.std_risk,
            "risk_confidence": self.risk_confidence,
            "calibrated_rf": self.calibrated_rf,
        }


# ─── PROMPT BANKS ─────────────────────────────────────────────────────────────

RECOMMENDATION_PROMPTS = [
    # 1 — Senior sell-side analyst
    ("You are a senior equity analyst at a tier-1 investment bank. "
     "Given the following news about {ticker}, rate the short-term stock outlook "
     "from 1 to 5 (1=very negative, 2=negative, 3=neutral, 4=positive, 5=very positive). "
     "Consider only the next 3-5 trading days. "
     "Respond with ONLY a single integer between 1 and 5.\n\nNews: {news}"),

    # 2 — Quantitative momentum trader
    ("You are a quantitative momentum trader. Assess the following news about {ticker} "
     "as a directional signal for a 5-day holding period. "
     "Score from 1 (strong sell signal) to 5 (strong buy signal), 3 = no clear signal. "
     "Respond with ONLY a single integer between 1 and 5.\n\nNews: {news}"),

    # 3 — Contrarian (forces opposite perspective before scoring)
    ("You are evaluating market sentiment for {ticker}. "
     "First, consider: what would a contrarian investor think about this news? "
     "Then give your balanced short-term outlook score from 1 to 5 "
     "(1=bearish, 3=neutral, 5=bullish). "
     "Respond with ONLY a single integer between 1 and 5.\n\nNews: {news}"),

    # 4 — Short-horizon event-driven
    ("For algorithmic trading, analyze this news about {ticker}. "
     "What is the expected 1-week price pressure? "
     "1=strong downward, 2=mild downward, 3=no effect, 4=mild upward, 5=strong upward. "
     "Respond with ONLY a single integer between 1 and 5.\n\nNews: {news}"),

    # 5 — Institutional buy-side
    ("As an institutional portfolio manager doing daily news triage for {ticker}: "
     "Rate the trading signal from 1 to 5. "
     "1=strong negative catalyst, 2=mild negative, 3=noise/non-event, "
     "4=mild positive catalyst, 5=strong positive catalyst. "
     "Respond with ONLY a single integer between 1 and 5.\n\nNews: {news}"),
]

RISK_PROMPTS = [
    # 1 — Tail-risk / VaR focus
    ("You are a tail-risk manager. Based on this news about {ticker}, "
     "estimate the downside tail risk for the next week: "
     "1=negligible, 2=low, 3=moderate (default for ambiguous news), "
     "4=elevated, 5=extreme tail risk. "
     "Respond with ONLY a single integer between 1 and 5.\n\nNews: {news}"),

    # 2 — Volatility focus
    ("As a volatility trader, assess the near-term volatility risk from this news about {ticker}. "
     "1=very low vol, 2=low vol, 3=normal vol, 4=elevated vol, 5=vol spike likely. "
     "Respond with ONLY a single integer between 1 and 5.\n\nNews: {news}"),

    # 3 — CVaR framing (aligned with agent loss function)
    ("You are computing CVaR adjustments for a risk-managed portfolio. "
     "Rate the news-implied risk for {ticker} at the 5%% confidence level: "
     "1=no CVaR impact, 2=minor, 3=moderate, 4=significant, 5=severe tail impact. "
     "Respond with ONLY a single integer between 1 and 5.\n\nNews: {news}"),

    # 4 — Systemic + idiosyncratic blend
    ("Consider systemic and firm-specific risk in this news about {ticker}. "
     "Risk score 1-5: 1=very safe, 2=mild risk, 3=neutral, "
     "4=concerning, 5=dangerous for portfolio. "
     "Respond with ONLY a single integer between 1 and 5.\n\nNews: {news}"),
]


# ─── SCORE PARSING ────────────────────────────────────────────────────────────

def parse_score(response: str, fallback: float = 3.0) -> float:
    """Robustly extract integer score 1–5 from any LLM response."""
    text = response.strip()
    if text in ("1", "2", "3", "4", "5"):
        return float(text)
    hits = re.findall(r'\b([1-5])\b', text)
    if hits:
        return float(hits[0])
    logger.debug(f"parse_score fallback for: {repr(text[:60])}")
    return fallback


# ─── UNCERTAINTY MATH ─────────────────────────────────────────────────────────

def compute_uncertainty(scores: List[float]) -> Tuple[float, float, float]:
    """
    Compute (μ, σ, confidence) from N scores.

    confidence c(σ) = max(0, 1 - σ / σ_max)
    σ_max = std([1,2,3,4,5]) ≈ 1.414  (maximum possible disagreement)
    """
    arr   = np.array(scores, dtype=float)
    mu    = float(np.mean(arr))
    sigma = float(np.std(arr))
    conf  = max(0.0, 1.0 - sigma / SIGMA_MAX)
    return mu, sigma, conf


def compute_sf(
    mu:         float,
    confidence: float,
    action:     float,
    alpha:      float = 0.10,
    tau:        float = 0.30,
) -> Tuple[float, bool]:
    """
    Uncertainty-gated action modifier.

    Sf = 1 + α · δ(μ, action) · c(σ) · m(μ)

    δ ∈ {-1, 0, +1}   directional alignment (signal vs. action direction)
    c(σ)              confidence gate — 0 if c < τ
    m(μ) = |μ-3|/2   signal magnitude ∈ [0,1]

    Returns (Sf, gate_fired)
    """
    if confidence < tau:
        return 1.0, True   # gate fired: no LLM influence

    if   mu >= 4.0 and action > 0:  delta = +1.0
    elif mu <= 2.0 and action < 0:  delta = +1.0
    elif mu >= 4.0 and action < 0:  delta = -1.0
    elif mu <= 2.0 and action > 0:  delta = -1.0
    else:                            delta =  0.0  # neutral zone

    magnitude = abs(mu - 3.0) / 2.0
    sf = 1.0 + alpha * delta * confidence * magnitude
    return float(np.clip(sf, 1.0 - alpha, 1.0 + alpha)), False


def compute_rf(
    mu_risk:    float,
    confidence: float,
    tau:        float = 0.30,
) -> float:
    """
    Uncertainty-gated CVaR risk modifier.

    Rf_gated = 1 + (Rf_base - 1) · c(σ_risk)
    When c < τ → Rf_gated = 1.0 (no CVaR distortion from uncertain signals)
    """
    BASE_RF = {5: 1.10, 4: 1.05, 3: 1.00, 2: 0.95, 1: 0.90}
    if confidence < tau:
        return 1.0
    risk_int = int(round(np.clip(mu_risk, 1.0, 5.0)))
    base     = BASE_RF[risk_int]
    return float(1.0 + (base - 1.0) * confidence)


# ─── LLM BACKENDS ─────────────────────────────────────────────────────────────

class _MockLLM:
    """
    Zero-cost mock for pipeline testing.
    Uses simple keyword heuristics to mimic LLM scoring behavior.
    """
    POS = ["beat", "record", "growth", "profit", "strong", "surge", "upgrade",
           "historic", "raised", "guidance", "outperform"]
    NEG = ["miss", "loss", "decline", "downgrade", "risk", "concern", "weak",
           "lawsuit", "probe", "fraud", "cut", "disappoints"]

    def complete(self, prompt: str) -> str:
        text  = prompt.lower()
        score = 3
        score += sum(1 for w in self.POS if w in text)
        score -= sum(1 for w in self.NEG if w in text)
        return str(max(1, min(5, score)))


class _OpenAIBackend:
    """OpenAI API backend (gpt-4o-mini or gpt-4o)."""

    def __init__(self, config: LLMConfig):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self._client = OpenAI(
            api_key=config.get_api_key(),
            base_url=config.api_base,
        )
        self.model       = config.model
        self.max_tokens  = config.max_tokens
        self.max_retries = config.max_retries
        self.retry_delay = config.retry_delay

    def complete(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (attempt + 1)
                    logger.warning(f"OpenAI retry {attempt+1}: {e} — waiting {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"OpenAI failed after {self.max_retries} attempts: {e}")
                    return "3"


class _AnthropicBackend:
    """
    Anthropic API backend (claude-haiku-4-5-20251001 recommended).
    Uses the Messages API — NOT OpenAI-compatible, separate client.
    """

    def __init__(self, config: LLMConfig):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        self._client = anthropic.Anthropic(api_key=config.get_api_key())
        self.model       = config.model
        self.max_tokens  = max(config.max_tokens, 16)  # Anthropic min is higher
        self.max_retries = config.max_retries
        self.retry_delay = config.retry_delay

    def complete(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text.strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (attempt + 1)
                    logger.warning(f"Anthropic retry {attempt+1}: {e} — waiting {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"Anthropic failed after {self.max_retries} attempts: {e}")
                    return "3"


def build_llm_backend(config: LLMConfig):
    """Factory: return the right LLM backend for the config."""
    if config.provider == "mock":
        return _MockLLM()
    if config.provider == "anthropic":
        return _AnthropicBackend(config)
    # "openai" — also works for any OpenAI-compatible endpoint (DeepSeek, etc.)
    return _OpenAIBackend(config)


# ─── MAIN CLASS ───────────────────────────────────────────────────────────────

class UncertaintyAwareLLM:
    """
    Uncertainty-gated LLM trading signal generator.

    Parameters
    ----------
    config          : LLMConfig — which LLM to use
    n_prompts       : ensemble size for recommendation (1–5)
    n_risk_prompts  : ensemble size for risk (1–4)
    alpha           : infusion strength α ∈ (0, 0.1]
    threshold_tau   : gate threshold τ — below this, signal is killed
    cache           : optional dict for (ticker, date) → UncertainSignal caching

    Quick start
    -----------
    >>> llm = UncertaintyAwareLLM(LLMConfig.openai())
    >>> sig = llm.get_signal("AAPL", "Apple beats earnings...", action=0.5)
    >>> print(sig.calibrated_sf, sig.confidence, sig.gate_fired)
    """

    def __init__(
        self,
        config:         LLMConfig      = None,
        n_prompts:      int            = 5,
        n_risk_prompts: int            = 4,
        alpha:          float          = 0.10,
        threshold_tau:  float          = 0.30,
        cache:          Optional[Dict] = None,
    ):
        if config is None:
            config = LLMConfig.mock()
        self.llm            = build_llm_backend(config)
        self.n_prompts      = min(n_prompts, len(RECOMMENDATION_PROMPTS))
        self.n_risk_prompts = min(n_risk_prompts, len(RISK_PROMPTS))
        self.alpha          = alpha
        self.tau            = threshold_tau
        self.cache          = cache if cache is not None else {}
        self._history:      List[Dict] = []

    # ── Public interface ──────────────────────────────────────────────────────

    def get_signal(
        self,
        ticker: str,
        news:   str,
        action: float          = 0.0,
        date:   Optional[str]  = None,
    ) -> UncertainSignal:
        """
        Compute uncertainty-gated signal for a (ticker, news, date) triplet.
        Caches result for the same (ticker, date, news_prefix).
        """
        key = self._cache_key(ticker, news, date)
        if key in self.cache:
            return self.cache[key]

        # Recommendation ensemble
        rec_scores = self._query(RECOMMENDATION_PROMPTS[:self.n_prompts], ticker, news)
        mu, sigma, conf = compute_uncertainty(rec_scores)

        # Risk ensemble
        risk_scores = self._query(RISK_PROMPTS[:self.n_risk_prompts], ticker, news)
        mu_r, sigma_r, conf_r = compute_uncertainty(risk_scores)

        # Gated modifiers
        sf, gate = compute_sf(mu, conf, action, self.alpha, self.tau)
        rf        = compute_rf(mu_r, conf_r, self.tau)

        sig = UncertainSignal(
            mean_score=mu,    std_score=sigma,    confidence=conf,
            raw_scores=rec_scores,  calibrated_sf=sf,
            mean_risk=mu_r,   std_risk=sigma_r,   risk_confidence=conf_r,
            raw_risk_scores=risk_scores, calibrated_rf=rf,
            gate_fired=gate,  ticker=ticker,       date=date or "",
        )
        self.cache[key] = sig
        self._history.append(sig.to_dict())
        return sig

    def get_signal_from_cache(
        self,
        ticker:    str,
        date:      str,
        action:    float,
        signal_df=None,
    ) -> UncertainSignal:
        """
        Fast path during RL training — no live LLM calls.
        Looks up pre-computed (μ, σ, confidence) from DataFrame,
        then recomputes Sf with the current action.
        """
        if signal_df is not None:
            try:
                row = signal_df.loc[(ticker, date)]
                sf, gate = compute_sf(
                    float(row["mean_score"]), float(row["confidence"]),
                    action, self.alpha, self.tau
                )
                rf = compute_rf(float(row["mean_risk"]),
                                float(row["risk_confidence"]), self.tau)
                return UncertainSignal(
                    mean_score=float(row["mean_score"]),
                    std_score=float(row["std_score"]),
                    confidence=float(row["confidence"]),
                    raw_scores=[],
                    calibrated_sf=sf,
                    mean_risk=float(row["mean_risk"]),
                    std_risk=float(row["std_risk"]),
                    risk_confidence=float(row["risk_confidence"]),
                    calibrated_rf=rf,
                    gate_fired=gate, ticker=ticker, date=date,
                )
            except (KeyError, TypeError):
                pass
        # Neutral fallback — no signal available for this (ticker, date)
        return UncertainSignal(
            mean_score=3.0, std_score=0.0, confidence=0.0, raw_scores=[],
            calibrated_sf=1.0, mean_risk=3.0, std_risk=0.0, risk_confidence=0.0,
            calibrated_rf=1.0, gate_fired=True, ticker=ticker, date=date,
        )

    def calibration_report(self) -> Dict:
        """Statistical summary of all signals generated. For the paper."""
        if not self._history:
            return {}
        confs  = [s["confidence"] for s in self._history]
        sigmas = [s["std_score"]  for s in self._history]
        gated  = sum(1 for s in self._history if s["gate_fired"])
        n      = len(self._history)
        return {
            "n_signals":        n,
            "gate_rate":        gated / n,
            "mean_confidence":  float(np.mean(confs)),
            "mean_sigma":       float(np.mean(sigmas)),
            "confidence_buckets": {
                "low  (0.00–0.33)": sum(c < 0.33 for c in confs) / n,
                "mid  (0.33–0.66)": sum(0.33 <= c < 0.66 for c in confs) / n,
                "high (0.66–1.00)": sum(c >= 0.66 for c in confs) / n,
            },
        }

    def save_cache_parquet(self, path: str) -> None:
        import pandas as pd
        df = pd.DataFrame(self._history)
        df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} signals → {path}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _query(self, prompts: List[str], ticker: str, news: str) -> List[float]:
        scores = []
        for template in prompts:
            prompt = template.format(ticker=ticker, news=news[:2000])
            try:
                raw   = self.llm.complete(prompt)
                score = parse_score(raw)
            except Exception as e:
                logger.warning(f"LLM query error [{ticker}]: {e}")
                score = 3.0
            scores.append(score)
        return scores

    @staticmethod
    def _cache_key(ticker: str, news: str, date: Optional[str]) -> str:
        content = f"{ticker}|{date or ''}|{news[:200]}"
        return hashlib.md5(content.encode()).hexdigest()
