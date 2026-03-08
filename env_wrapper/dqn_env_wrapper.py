import os
import pandas as pd
import numpy as np
import torch
import gymnasium as gym
import warnings

print("[✓] ElegantRL 导入完成 - 使用 AgentDQN")

# ---------------------------------------------------------
# 3. 核心：NumPy 极速版离散高频环境 (解决采样瓶颈)
# ---------------------------------------------------------
class FastDiscreteCryptoEnv(gym.Env):
    """
    高频交易环境：将 DataFrame 转换为 NumPy 数组以获得极速采样性能
    动作空间: 0(卖出) | 1(持有) | 2(买入)
    """
    def __init__(self, df, indicators, initial_amount=1000000, hmax=5, cost_pct=0.0002, env_name="FastCryptoEnv", **kwargs):
        super().__init__()
        self.env_name = env_name
        
        # 将 DataFrame 提前转换为 NumPy 矩阵，消除 CPU 索引瓶颈
        self.prices = df['close'].values.astype(np.float32)
        self.features = df[indicators].values.astype(np.float32)
        self.max_step = len(df) - 1
        
        self.initial_amount = initial_amount
        self.hmax = hmax
        self.cost_pct = cost_pct
        
        # 动作空间：0(卖出), 1(持有), 2(买入)
        self.action_space = gym.spaces.Discrete(3)
        self.action_dim = 3
        
        # 状态空间：[账户余额, 当前价格, 持有份额] + [特征指标...]
        self.state_dim = 3 + len(indicators)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,))
        
        # 记录内部状态
        self.day = 0
        self.balance = self.initial_amount
        self.shares = 0
        self.asset_memory = []
        
    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        self.day = 0
        self.balance = self.initial_amount
        self.shares = 0
        self.asset_memory = [self.initial_amount]
        return self._get_obs(), {}
        
    def _get_obs(self):
        """获取当前观测状态"""
        obs = np.concatenate(([self.balance, self.prices[self.day], self.shares], self.features[self.day]))
        return obs.astype(np.float32)
        
    def step(self, action):
        """执行一步动作"""
        self.day += 1
        if self.day >= self.max_step:
            return self._get_obs(), 0.0, True, False, {}
            
        price = self.prices[self.day]
        trade_shares = 0
        
        # 动作执行逻辑
        if action == 0 and self.shares > 0:  # 卖出
            trade_shares = -min(self.shares, self.hmax)
        elif action == 2:  # 买入
            # 防除零护盾：如果碰到异常脏数据(价格为0)，直接放弃买入
            if price > 0:
                max_buy = int(self.balance / (price * (1 + self.cost_pct)))
                trade_shares = min(max_buy, self.hmax)
            else:
                trade_shares = 0
            
        # 计算交易成本与余额更新
        trade_amount = trade_shares * price
        cost = abs(trade_amount) * self.cost_pct
        self.balance -= (trade_amount + cost)
        self.shares += trade_shares
        
        # 计算奖励 (资产净值变化)
        current_asset = self.balance + self.shares * price
        prev_asset = self.asset_memory[-1]
        reward = current_asset - prev_asset
        self.asset_memory.append(current_asset)
        
        # Reward 缩放，帮助神经网络收敛
        reward = reward * 1e-4
        done = self.day >= self.max_step
        
        return self._get_obs(), float(reward), done, False, {}
