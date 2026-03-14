"""
Environment wrapper for PPO agent training with ElegantRL.

This module contains:
1. FastContinuousCryptoEnv - High-frequency crypto trading with continuous actions (NumPy-based)
2. ElegantFinRLWrapper - Wrapper for StockTradingEnv compatibility

These must be in a separate file (not in a Jupyter notebook) to support multiprocessing 
pickle serialization required by ElegantRL's train_agent() function.
"""

import gymnasium as gym
import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

print("[✓] PPO Environment Wrappers imported successfully")


# ---------------------------------------------------------
# 1. 高频加密货币连续动作环境（NumPy 极速版）
# ---------------------------------------------------------
class FastContinuousCryptoEnv(gym.Env):
    """
    High-frequency crypto trading environment with continuous action space.
    Action: [-1, 1] where -1=sell, 0=hold, 1=buy
    
    Similar to FastDiscreteCryptoEnv but uses continuous actions for PPO.
    """
    def __init__(self, df, indicators, initial_amount=1000000, hmax=5, cost_pct=0.0002,
                 env_name="FastCryptoEnv", reward_scale=1000.0, **kwargs):
        super().__init__()
        self.env_name = env_name
        
        # 将 DataFrame 提前转换为 NumPy 矩阵，消除 CPU 索引瓶颈
        prices = df['close'].values.astype(np.float32)
        features = df[indicators].values.astype(np.float32)
        self.prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        self.features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        self.max_step = len(df) - 1
        
        self.initial_amount = float(initial_amount)
        self.hmax = int(hmax)
        self.cost_pct = float(cost_pct)
        self.reward_scale = float(reward_scale)
        self.price_scale = max(float(self.prices[0]), 1.0)
        self.max_possible_shares = max(
            int(self.initial_amount / (self.price_scale * (1 + self.cost_pct))),
            1,
        )
        
        # 连续动作空间：[-1, 1]，其中 -1=卖出, 0=持有, 1=买入
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_dim = 1
        
        # 状态空间：[账户余额, 当前价格, 持有份额] + [特征指标...]
        self.state_dim = 3 + len(indicators)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                                 shape=(self.state_dim,), dtype=np.float32)
        
        # 记录内部状态
        self.day = 0
        self.balance = self.initial_amount
        self.shares = 0
        self.asset_memory = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.balance = self.initial_amount
        self.shares = 0
        self.asset_memory = [self.initial_amount]
        return self._get_obs(), {}
        
    def _get_obs(self):
        # 对核心账户变量做尺度归一化，避免大数值淹没特征信号
        norm_balance = self.balance / self.initial_amount
        norm_price = self.prices[self.day] / self.price_scale
        norm_shares = self.shares / float(self.max_possible_shares)
        obs = np.concatenate(([norm_balance, norm_price, norm_shares], self.features[self.day]))
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs.astype(np.float32)
        
    def step(self, action):
        self.day += 1
        if self.day >= self.max_step:
            return self._get_obs(), 0.0, True, False, {}

        if isinstance(action, np.ndarray):
            action = float(action.squeeze())

        if not np.isfinite(action):
            action = 0.0

        action = float(np.clip(action, -1.0, 1.0))

        price = float(self.prices[self.day])
        prev_asset = float(self.asset_memory[-1])

        trade_shares = 0
        max_short = self.hmax   # 最大允许空仓

        if price > 0:

            # BUY
            if action > 0:
                max_buy = int(self.balance / (price * (1 + self.cost_pct)))
                target = int(abs(action) * self.hmax)
                trade_shares = min(max_buy, target)

            # SELL / SHORT
            elif action < 0:
                target = int(abs(action) * self.hmax)

                # 如果有持仓先卖出
                if self.shares > 0:
                    trade_shares = -min(self.shares, target)

                # 如果没有持仓允许做空
                else:
                    trade_shares = -min(max_short, target)

        trade_amount = trade_shares * price
        fee = abs(trade_amount) * self.cost_pct

        self.balance -= (trade_amount + fee)
        self.shares += trade_shares

        current_asset = float(self.balance + self.shares * price)
        safe_asset = max(current_asset, 0.0)

        reward = (safe_asset - prev_asset) / self.initial_amount

        # 无交易惩罚（防止策略坍塌）
        if trade_shares == 0:
            reward -= 1e-4

        reward = float(np.clip(reward * self.reward_scale, -5, 5))

        self.asset_memory.append(safe_asset)

        done = safe_asset <= 0.0 or self.day >= self.max_step

        return self._get_obs(), reward, done, False, {}

# ---------------------------------------------------------
# 2. StockTradingEnv 包装器
# ---------------------------------------------------------
class ElegantFinRLWrapper(gym.Wrapper):
    def __init__(self, df, stock_dim, hmax, initial_amount, num_stock_shares,
                 buy_cost_pct, sell_cost_pct, reward_scaling, state_space,
                 action_space, tech_indicator_list, env_name, state_dim,
                 action_dim, if_discrete, target_return, print_verbosity=1000, **kwargs):

        # Coerce accidental non-scalar values (e.g., pandas Series) to scalar cash value.
        if isinstance(initial_amount, (list, tuple, np.ndarray)):
            initial_amount = float(np.asarray(initial_amount).reshape(-1)[-1])
        elif hasattr(initial_amount, "iloc"):
            initial_amount = float(initial_amount.iloc[-1])
        else:
            initial_amount = float(initial_amount)

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
            tech_indicator_list=tech_indicator_list,
            print_verbosity=print_verbosity,
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
            # print("[DEBUG] Step Reward: {:.4f}, Action Penalty: {:.4f}, Total Reward: {:.4f}".format(
            #     reward, penalty, reward - penalty
            # ))
            # modified_reward = float(reward) - float(penalty)
            modified_reward = float(reward)
            return np.array(obs, dtype=np.float32), modified_reward, term, trunc, info
        else:
            obs, reward, done, info = res
            # modified_reward = float(reward) - float(penalty)
            modified_reward = float(reward)
            return np.array(obs, dtype=np.float32), modified_reward, done, False, info

print("[✓] FastContinuousCryptoEnv & ElegantFinRLWrapper classes defined!")