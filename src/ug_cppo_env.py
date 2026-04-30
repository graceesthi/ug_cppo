"""
ug_cppo_env.py
==============
UG-CPPO Trading Environment.

Extends FinRL's StockTradingEnv to inject uncertainty-gated LLM signals
into both the action (Sf) and the return for CVaR loss (Rf).

Key modifications vs. baseline:
  - step(): applies Sf modifier to action BEFORE execution
  - _get_portfolio_return(): returns Rf-adjusted value for CVaR loss
  - Signals loaded from pre-computed DataFrame (O(1) lookup, no live LLM)
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


class UGCPPOTradingEnv(gym.Env):
    """
    Uncertainty-Gated CVaR-PPO Trading Environment.

    State space:
        [cash, stock_prices×N, stock_shares×N, tech_indicators×N×M]

    Action space:
        Continuous [-1, 1]^N — negative=sell, positive=buy (scaled by hmax)

    UG-CPPO Additions:
        signal_df: pre-computed (ticker, date) indexed DataFrame with
                   {mean_score, confidence, mean_risk, risk_confidence}
        On each step:
          1. Compute Sf via compute_sf(μ, c, action_i, α, τ)
          2. Apply: a_mod = Sf * a_t   (action modification)
          3. Compute Rf_portfolio = Σ_i w_i * Rf_i
          4. Store Rf in info dict for CVaR loss computation
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df:               pd.DataFrame,         # OHLCV + tech indicators, sorted by (date, tic)
        tickers:          list,
        signal_df:        Optional[pd.DataFrame] = None,  # pre-computed UQ signals
        initial_amount:   float = 1_000_000.0,
        hmax:             int   = 100,
        transaction_cost: float = 0.001,
        reward_scaling:   float = 1e-4,
        alpha:            float = 0.10,
        threshold_tau:    float = 0.30,
        tech_indicator_list: Optional[list] = None,
        mode:             str  = "ug_cppo",      # "ug_cppo" | "cppo" | "ppo"
    ):
        super().__init__()
        self.df               = df.copy()
        self.tickers          = tickers
        self.n_stocks         = len(tickers)
        self.signal_df        = signal_df
        self.initial_amount   = initial_amount
        self.hmax             = hmax
        self.transaction_cost = transaction_cost
        self.reward_scaling   = reward_scaling
        self.alpha            = alpha
        self.tau              = threshold_tau
        self.tech_indicators  = tech_indicator_list or []
        self.mode             = mode

        # Build date list
        self.dates      = sorted(df["date"].unique())
        self.date_index = 0
        self.n_dates    = len(self.dates)

        # State dimensions
        n_tech    = len(self.tech_indicators)
        state_dim = 1 + self.n_stocks + self.n_stocks + self.n_stocks * n_tech

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_stocks,), dtype=np.float32
        )

        # Portfolio state
        self.cash:   float      = initial_amount
        self.shares: np.ndarray = np.zeros(self.n_stocks)
        self.cost_basis:     float = 0.0
        self.portfolio_value_history: list = []

        # UG-CPPO tracking
        self.last_rf:          float = 1.0    # for CVaR loss
        self.gate_events:      int   = 0
        self.total_steps:      int   = 0

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.date_index  = 0
        self.cash        = self.initial_amount
        self.shares      = np.zeros(self.n_stocks)
        self.cost_basis  = 0.0
        self.portfolio_value_history = [self.initial_amount]
        self.last_rf     = 1.0
        self.gate_events = 0
        self.total_steps = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one trading step with UG-CPPO signal injection.

        Sequence:
          1. Get current prices
          2. (UG-CPPO) Compute per-stock Sf modifiers, apply to actions
          3. Execute trades
          4. Compute portfolio value and reward
          5. (UG-CPPO) Compute portfolio Rf for CVaR
          6. Advance date
        """
        current_date = self.dates[self.date_index]
        prices       = self._get_prices(current_date)
        weights      = self._get_portfolio_weights(prices)

        # ── UG-CPPO: Apply Sf to each action ─────────────────────────────
        if self.mode == "ug_cppo" and self.signal_df is not None:
            action, gate_count = self._apply_sf(action, current_date, prices)
            self.gate_events += gate_count
        self.total_steps += 1

        # ── Execute trades ────────────────────────────────────────────────
        portfolio_value_before = self._portfolio_value(prices)
        self._execute_trades(action, prices)
        portfolio_value_after  = self._portfolio_value(prices)

        # ── Reward (scaled portfolio return) ─────────────────────────────
        raw_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        reward     = float(raw_return * self.reward_scaling)

        # ── UG-CPPO: Compute portfolio Rf ─────────────────────────────────
        if self.mode in ("ug_cppo", "cppo") and self.signal_df is not None:
            self.last_rf = self._compute_portfolio_rf(current_date, weights)
        else:
            self.last_rf = 1.0

        self.portfolio_value_history.append(portfolio_value_after)
        self.date_index += 1
        done = (self.date_index >= self.n_dates - 1)

        info = {
            "date":            current_date,
            "portfolio_value": portfolio_value_after,
            "rf":              self.last_rf,             # used by CVaR loss
            "raw_return":      raw_return,
            "gate_rate":       self.gate_events / max(1, self.total_steps),
        }
        return self._get_obs(), reward, done, False, info

    # ── UG-CPPO core logic ────────────────────────────────────────────────────

    def _apply_sf(
        self,
        actions:      np.ndarray,
        date:         str,
        prices:       np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Apply uncertainty-gated Sf modifier to each stock's action.
        Returns (modified_actions, n_gates_fired)
        """
        from src.uncertainty_llm import compute_sf

        modified = actions.copy()
        gates    = 0
        for i, ticker in enumerate(self.tickers):
            try:
                row = self.signal_df.loc[(ticker, date)]
                mu   = float(row["mean_score"])
                conf = float(row["confidence"])
                sf, fired = compute_sf(mu, conf, float(actions[i]), self.alpha, self.tau)
                modified[i] = sf * actions[i]
                if fired:
                    gates += 1
            except (KeyError, TypeError):
                pass   # no signal for this (ticker, date) → neutral
        return modified, gates

    def _compute_portfolio_rf(
        self,
        date:    str,
        weights: np.ndarray,
    ) -> float:
        """
        Portfolio-level Rf: Σ_i w_i * Rf_i
        Used to scale trajectory return in CVaR-PPO loss.
        """
        from src.uncertainty_llm import compute_rf

        rf_portfolio = 0.0
        for i, ticker in enumerate(self.tickers):
            try:
                row   = self.signal_df.loc[(ticker, date)]
                mu_r  = float(row["mean_risk"])
                conf_r = float(row["risk_confidence"])
                rf_i  = compute_rf(mu_r, conf_r, self.tau)
            except (KeyError, TypeError):
                rf_i = 1.0
            rf_portfolio += weights[i] * rf_i

        # Normalise (weights may not sum to 1 if mostly cash)
        weight_sum = weights.sum()
        if weight_sum > 0:
            rf_portfolio = 1.0 + (rf_portfolio - weight_sum) / weight_sum
        return float(np.clip(rf_portfolio, 0.9, 1.1))

    # ── Trade execution ───────────────────────────────────────────────────────

    def _execute_trades(self, actions: np.ndarray, prices: np.ndarray) -> None:
        """Execute buy/sell orders based on continuous actions ∈ [-1,1]."""
        for i in range(self.n_stocks):
            if prices[i] <= 0:
                continue
            desired_shares = int(actions[i] * self.hmax)
            delta          = desired_shares - int(self.shares[i])

            if delta > 0:  # Buy
                affordable = int(self.cash / (prices[i] * (1 + self.transaction_cost)))
                delta      = min(delta, affordable)
                cost       = delta * prices[i] * (1 + self.transaction_cost)
                self.cash   -= cost
                self.shares[i] += delta

            elif delta < 0:  # Sell
                delta = max(delta, -int(self.shares[i]))
                proceeds = abs(delta) * prices[i] * (1 - self.transaction_cost)
                self.cash   += proceeds
                self.shares[i] += delta

    # ── Observations ─────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        if self.date_index >= self.n_dates:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        date   = self.dates[self.date_index]
        prices = self._get_prices(date)
        tech   = self._get_tech(date)
        state  = np.concatenate([[self.cash / self.initial_amount],
                                  prices / 1000.0,
                                  self.shares / self.hmax,
                                  tech.flatten()])
        return state.astype(np.float32)

    def _get_prices(self, date: str) -> np.ndarray:
        day = self.df[self.df["date"] == date]
        prices = []
        for tic in self.tickers:
            row = day[day["tic"] == tic]
            prices.append(float(row["close"].values[0]) if len(row) else 0.0)
        return np.array(prices, dtype=np.float64)

    def _get_tech(self, date: str) -> np.ndarray:
        if not self.tech_indicators:
            return np.array([])
        day = self.df[self.df["date"] == date]
        tech = []
        for tic in self.tickers:
            row = day[day["tic"] == tic]
            if len(row):
                tech.append([float(row[col].values[0]) for col in self.tech_indicators])
            else:
                tech.append([0.0] * len(self.tech_indicators))
        return np.array(tech, dtype=np.float64)

    def _portfolio_value(self, prices: np.ndarray) -> float:
        return float(self.cash + np.dot(self.shares, prices))

    def _get_portfolio_weights(self, prices: np.ndarray) -> np.ndarray:
        """Fractional weights of each stock position in portfolio."""
        total = self._portfolio_value(prices)
        if total <= 0:
            return np.ones(self.n_stocks) / self.n_stocks
        return (self.shares * prices) / total

    def render(self, mode="human") -> None:
        pv = self.portfolio_value_history[-1] if self.portfolio_value_history else self.initial_amount
        print(f"Date: {self.dates[min(self.date_index, self.n_dates-1)]} | "
              f"Portfolio: ${pv:,.0f} | Gate rate: {self.gate_events/max(1,self.total_steps):.1%}")
