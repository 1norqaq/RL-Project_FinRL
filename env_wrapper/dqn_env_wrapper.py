import os
import pandas as pd
import numpy as np
import torch
import gymnasium as gym
import warnings
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

print("[✓] ElegantRL 导入完成 - 使用 AgentDQN")

# ---------------------------------------------------------
# 3. 核心：NumPy 极速版离散高频环境 (解决采样瓶颈)
# ---------------------------------------------------------
class FastDiscreteCryptoEnv(gym.Env):
    def __init__(
        self,
        df,
        indicators,
        initial_amount=1000000,
        hmax=5,
        cost_pct=0.0002,
        env_name="FastCryptoEnv",
        reward_scale=1000.0,
        **kwargs
    ):
        super().__init__()
        self.env_name = env_name

        self.prices = df['close'].values.astype(np.float32)
        self.features = df[indicators].values.astype(np.float32)
        self.max_step = len(df) - 1

        self.initial_amount = float(initial_amount)
        self.hmax = int(hmax)
        self.cost_pct = float(cost_pct)
        self.reward_scale = float(reward_scale)

        self.price_scale = max(float(self.prices[0]), 1.0)

        # 用初始价格估算最大可持仓，用于归一化 shares
        self.max_possible_shares = max(
            int(self.initial_amount / (self.price_scale * (1 + self.cost_pct))),
            1
        )

        # 动作空间：0(卖出), 1(持有), 2(买入)
        self.action_space = gym.spaces.Discrete(3)
        self.action_dim = 3

        # 状态空间：[余额, 当前价格, 持仓] + 特征
        self.state_dim = 3 + len(indicators)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

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
        norm_balance = self.balance / self.initial_amount
        norm_price = self.prices[self.day] / self.price_scale
        norm_shares = self.shares / float(self.max_possible_shares)

        obs = np.concatenate((
            np.array([norm_balance, norm_price, norm_shares], dtype=np.float32),
            self.features[self.day]
        ))
        return obs.astype(np.float32)

    def step(self, action):
        self.day += 1
        if self.day >= self.max_step:
            return self._get_obs(), 0.0, True, False, {}

        action = int(action)
        price = float(self.prices[self.day])

        trade_shares = 0

        # 先记录交易前资产
        prev_asset = self.balance + self.shares * price

        # 动作执行逻辑
        if action == 0 and self.shares > 0:  # 卖出
            trade_shares = -min(self.shares, self.hmax)

        elif action == 2 and price > 0:  # 买入
            max_buy = int(self.balance / (price * (1 + self.cost_pct)))
            trade_shares = min(max_buy, self.hmax)

        trade_amount = trade_shares * price
        fee = abs(trade_amount) * self.cost_pct

        self.balance -= (trade_amount + fee)
        self.shares += trade_shares

        current_asset = self.balance + self.shares * price

        # 不再用 prev_asset 做分母，也不再 clip
        reward = (current_asset - prev_asset) / self.initial_amount
        reward = float(reward * self.reward_scale)

        self.asset_memory.append(current_asset)

        done = self.day >= self.max_step
        return self._get_obs(), reward, done, False, {}
    

class ElegantDQNWrapper(gym.Wrapper):
    """
    封装 DiscretizedStockTradingEnv 以兼容 ElegantRL 的 DQN Agent
    """
    def __init__(self, df, stock_dim, hmax, initial_amount, buy_cost_pct, sell_cost_pct,
                 reward_scaling, state_space, action_space, tech_indicator_list, **kwargs):
        env = DiscretizedStockTradingEnv(
            df=df,
            stock_dim=stock_dim,
            hmax=hmax,
            initial_amount=initial_amount,
            num_stock_shares=[0] * stock_dim,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            reward_scaling=reward_scaling,
            state_space=state_space,
            action_space=action_space,
            tech_indicator_list=tech_indicator_list
        )
        super().__init__(env)
        self.env = env

    @staticmethod
    def _to_state_array(state):
        # ElegantRL 训练器要求 state 为 np.ndarray 且有 shape 属性
        if isinstance(state, tuple):
            state = state[0]
        return np.asarray(state, dtype=np.float32)
        
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # 确保 result 解包为 (state, info) 恰好 2 个值
        if isinstance(result, tuple):
            if len(result) == 2:
                state, info = result
            elif len(result) > 2:
                # 多于 2 个值，只取前两个
                state, info = result[0], result[1]
            else:
                # 只有 1 个值
                state = result[0]
                info = {}
        else:
            # 不是 tuple，当作单个 state
            state = result
            info = {}
        
        return self._to_state_array(state), info
    
    def step(self, action):
        step_res = self.env.step(action)

        # 同时兼容 gym(4项) 与 gymnasium(5项)
        if len(step_res) == 5:
            state, reward, terminated, truncated, info = step_res
            done = bool(terminated or truncated)
        else:
            state, reward, done, info = step_res

        # ElegantRL 期望 (state, reward, done, truncated, info)
        return self._to_state_array(state), float(reward), bool(done), False, info


class DiscretizedStockTradingEnv(StockTradingEnv):
    """
    包装 StockTradingEnv 为离散动作空间
    每只股票有 3 个动作: 卖出(-1), 持有(0), 买入(+1)
    总动作数 = 3^stock_dim (对于30只股票会很大，所以简化为每次只操作一只股票)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 简化动作空间: stock_dim * 3 (每只股票3个动作) + 1 (不操作)
        # 动作编码: 0 = 不操作, 1-90 = 对应股票的买入/持有/卖出
        self.action_space = gym.spaces.Discrete(self.stock_dim * 3 + 1)
        self.stock_action_dim = 3  # 卖出(-1), 持有(0), 买入(+1)
        
    def step(self, action):
        # 将离散动作转换为连续动作向量
        continuous_actions = np.zeros(self.stock_dim)
        
        if action > 0:  # 不是"不操作"
            action_idx = action - 1
            stock_idx = action_idx // 3
            action_type = action_idx % 3  # 0=卖出, 1=持有, 2=买入
            
            if action_type == 0:  # 卖出
                continuous_actions[stock_idx] = -self.hmax
            elif action_type == 2:  # 买入
                continuous_actions[stock_idx] = self.hmax
            # action_type == 1 时持有，保持为0
        
        return super().step(continuous_actions)