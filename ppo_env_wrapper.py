"""
Environment wrapper for PPO agent training with ElegantRL.

This module contains the ElegantFinRLWrapper class, which must be in a separate
file (not in a Jupyter notebook) to support multiprocessing pickle serialization
required by ElegantRL's train_agent_multiprocessing() function.
"""

import gymnasium as gym
import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


class ElegantFinRLWrapper(gym.Wrapper):
    def __init__(self, df, stock_dim, hmax, initial_amount, num_stock_shares,
                 buy_cost_pct, sell_cost_pct, reward_scaling, state_space,
                 action_space, tech_indicator_list, env_name, state_dim,
                 action_dim, if_discrete, target_return, **kwargs):

        buy_cost_list = buy_cost_pct if isinstance(buy_cost_pct, list) else [0.001] * stock_dim
        sell_cost_list = sell_cost_pct if isinstance(sell_cost_pct, list) else [0.001] * stock_dim

        env = StockTradingEnv(
            df=df,
            stock_dim=stock_dim,
            hmax=hmax,
            initial_amount=initial_amount,
            num_stock_shares=num_stock_shares,
            buy_cost_pct=buy_cost_list,
            sell_cost_pct=sell_cost_list,
            reward_scaling=reward_scaling,
            state_space=state_space,
            action_space=action_space,
            tech_indicator_list=tech_indicator_list
        )
        super().__init__(env)

        self.env_name = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.if_discrete = if_discrete
        self.target_return = target_return
        self.max_step = len(df.index.unique())

        self.prev_action = np.zeros(action_dim)
                   
        self.penalty_coef = 0.05

    def reset(self):
        self.prev_action = np.zeros(self.action_dim)
        res = self.env.reset()
        if isinstance(res, tuple) and len(res) == 2:
            obs, info = res
        else:
            obs = res
            info = {}
        return np.array(obs, dtype=np.float32), info

    def step(self, action):
        res = self.env.step(action)

        # Action Penalty: Penalize frequent portfolio changes
        action_diff = np.sum(np.abs(action - self.prev_action))
        penalty = action_diff * self.penalty_coef
        self.prev_action = action.copy()

        if len(res) == 5:
            obs, reward, term, trunc, info = res
            modified_reward = float(reward) - float(penalty)
            return np.array(obs, dtype=np.float32), modified_reward, term, trunc, info
        else:
            obs, reward, done, info = res
            modified_reward = float(reward) - float(penalty)
            return np.array(obs, dtype=np.float32), modified_reward, done, False, info

print("[✓] ElegantFinRLWrapper class defined!")