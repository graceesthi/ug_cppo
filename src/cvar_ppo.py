"""
cvar_ppo.py — CVaR-PPO with uncertainty-gated return adjustment.
Rf injection done via RFCollectorCallback (in notebook), not here.
"""
from __future__ import annotations
import logging
from typing import Optional, Union, Type

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv

logger = logging.getLogger(__name__)


class CVaRPPO(PPO):
    """
    CVaR-PPO with Uncertainty-Gated return adjustment.

    The CVaR adjustment is performed externally by RFCollectorCallback,
    which modifies rollout_buffer.returns in _on_rollout_end BEFORE
    PPO.train() reads them. This avoids dtype conflicts and double-patching.

    Parameters
    ----------
    cvar_alpha  : CVaR confidence level (default 0.05 = worst 5%)
    cvar_lambda : Lagrange multiplier — blending weight for Rf adjustment
    cvar_beta   : auxiliary penalty (unused, kept for API compatibility)
    """
    def __init__(self, policy, env, cvar_alpha=0.05, cvar_lambda=0.10,
                 cvar_beta=0.0, **ppo_kwargs):
        super().__init__(policy=policy, env=env, **ppo_kwargs)
        self.cvar_alpha  = cvar_alpha
        self.cvar_lambda = cvar_lambda
        self.cvar_beta   = cvar_beta
        # Buffer populated by RFCollectorCallback
        self._rf_buffer: list = []


def build_agent(env, mode="ug_cppo", learning_rate=3e-4, n_steps=2048,
                batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
                cvar_alpha=0.05, cvar_lambda=0.10, cvar_beta=0.0,
                tensorboard_log=None, seed=42, verbose=1):
    """
    Factory: 'ppo' → standard PPO | 'cppo'/'ug_cppo' → CVaRPPO
    All modes use identical hyperparameters for fair comparison.
    """
    common = dict(
        policy="MlpPolicy", env=env,
        learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
        n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda,
        clip_range=clip_range, ent_coef=ent_coef, vf_coef=vf_coef,
        tensorboard_log=tensorboard_log, seed=seed, verbose=verbose,
    )
    if mode == "ppo":
        return PPO(**common)
    return CVaRPPO(cvar_alpha=cvar_alpha, cvar_lambda=cvar_lambda,
                   cvar_beta=cvar_beta, **common)
