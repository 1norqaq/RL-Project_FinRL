import os
import pandas as pd
import numpy as np
import torch
import yfinance as yf
import gym

original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_torch_load(*args, **kwargs)


torch.load = patched_torch_load

original_download = yf.download


def patched_download(*args, **kwargs):
    kwargs.pop('proxy', None)
    return original_download(*args, **kwargs)


yf.download = patched_download

from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from elegantrl.agents import AgentSAC
from elegantrl.train.run import train_agent

try:
    from elegantrl.train.config import Config as Arguments
except ImportError:
    from elegantrl.train.config import Arguments


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
        self.prev_action = np.zeros(self.action_dim)  # 重置动作记录
        res = self.env.reset()
        if isinstance(res, tuple) and len(res) == 2:
            obs, info = res
        else:
            obs = res
            info = {}
        return np.array(obs, dtype=np.float32), info

    def step(self, action):
        res = self.env.step(action)

        # 调仓惩罚计算 (Action Penalty)
        # 惩罚力度 = 当前动作与昨日动作的绝对值差的综合 * 惩罚系数
        action_diff = np.sum(np.abs(action - self.prev_action))
        penalty = action_diff * self.penalty_coef
        self.prev_action = action.copy()

        if len(res) == 5:
            obs, reward, term, trunc, info = res
            # 从环境给的原始利润奖励中，扣除“乱动症”惩罚
            modified_reward = float(reward) - float(penalty)
            return np.array(obs, dtype=np.float32), modified_reward, term, trunc, info
        else:
            obs, reward, done, info = res
            modified_reward = float(reward) - float(penalty)
            return np.array(obs, dtype=np.float32), modified_reward, done, False, info


def prepare_data():
    cache_file = "finrl_dow30_cache.csv"
    if os.path.exists(cache_file):
        print(f"[INFO] Loading validated cache '{cache_file}'...")
        df = pd.read_csv(cache_file)
        return df.drop_duplicates(subset=['date', 'tic'])
    return None


def setup_sac_args(env_args, cwd_path):
    args = Arguments(agent_class=AgentSAC, env_class=ElegantFinRLWrapper)
    args.env_args = env_args
    args.env_name = env_args['env_name']

    # 扩大网络容量，以应对 1 年期长线数据
    args.net_dims = (128, 64)
    args.state_dim = env_args['state_dim']
    args.action_dim = env_args['action_dim']
    args.if_discrete = env_args['if_discrete']

    args.learning_rate = 1e-4  # 更保守的学习率
    args.batch_size = 128  # 更大的批次以提高稳定性

    # 设定训练目标步数
    args.target_step = 2000
    args.break_step = 40000

    args.worker_num = 1
    args.eval_proc_num = 0
    args.eval_gap = 1000

    args.cwd = cwd_path
    args.if_remove = True
    return args


def real_test_inference(test_df, stock_dim, indicators, args):
    params = {
        "df": test_df, "stock_dim": stock_dim, "hmax": 100,
        "initial_amount": 1000000, "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4,
        "state_space": 1 + stock_dim * (len(indicators) + 2),
        "action_space": stock_dim, "tech_indicator_list": indicators,
        "env_name": "test_inference", "state_dim": 1 + stock_dim * (len(indicators) + 2),
        "action_dim": stock_dim, "if_discrete": False, "target_return": 10.0
    }
    env = ElegantFinRLWrapper(**params)

    agent = AgentSAC(args.net_dims, args.state_dim, args.action_dim)
    agent.save_or_load_agent(args.cwd, if_save=False)
    agent.act.eval()

    res = env.reset()
    state = res[0] if isinstance(res, tuple) else res

    done = False
    while not done:
        s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            action = agent.act(s_tensor).detach().cpu().numpy()[0]

        step_res = env.step(action)
        if len(step_res) == 5:
            state, reward, term, trunc, _ = step_res
            done = term or trunc
        else:
            state, reward, done, _, _ = step_res

    return env.env.save_asset_memory()


if __name__ == '__main__':
    df_raw = prepare_data()
    df_pivot = df_raw.pivot(index='date', columns='tic', values='close')
    valid_dates = df_pivot.dropna().index
    df = df_raw[df_raw['date'].isin(valid_dates)].copy()

    unique_dates = sorted(df['date'].unique())
    stock_dimension = len(df['tic'].unique())
    ALL_INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30", "close_60"]

    TRAIN_WINDOW = 252  # 约 1 年的交易日
    VAL_WINDOW = 20
    TEST_WINDOW = 20  # 每次测试推断 1 个月

    all_test_results = []

    for i in range(TRAIN_WINDOW + VAL_WINDOW, len(unique_dates) - TEST_WINDOW, TEST_WINDOW):
        train_dates = unique_dates[i - TRAIN_WINDOW - VAL_WINDOW: i - VAL_WINDOW]
        test_dates = unique_dates[i: i + TEST_WINDOW]

        train_df = df[df['date'].isin(train_dates)].sort_values(['date', 'tic']).reset_index(drop=True)
        test_df = df[df['date'].isin(test_dates)].sort_values(['date', 'tic']).reset_index(drop=True)

        for d_df in [train_df, test_df]:
            date_map = {date: idx for idx, date in enumerate(sorted(d_df['date'].unique()))}
            d_df['day'] = d_df['date'].map(date_map)
            d_df.set_index('day', inplace=True, drop=False)

        state_dim = 1 + stock_dimension * (len(ALL_INDICATORS) + 2)

        env_params = {
            "env_name": f"FinRL_Window_{i}",
            "df": train_df,
            "stock_dim": stock_dimension,
            "hmax": 100,
            "initial_amount": 1000000,
            "num_stock_shares": [0] * stock_dimension,
            "buy_cost_pct": [0.001] * stock_dimension,
            "sell_cost_pct": [0.001] * stock_dimension,
            "reward_scaling": 1e-4,
            "state_space": state_dim,
            "action_space": stock_dimension,
            "tech_indicator_list": ALL_INDICATORS,
            "state_dim": state_dim,
            "action_dim": stock_dimension,
            "if_discrete": False,
            "target_return": 10.0
        }

        print(f"\n>>> Rolling Window: {train_dates[0]} to {test_dates[-1]}")

        try:
            cwd_path = f"./checkpoints/sac_window_{i}"
            os.makedirs(cwd_path, exist_ok=True)

            args = setup_sac_args(env_params, cwd_path)

            print("[-->] Starting SAC Agent Training...")
            train_agent(args)

            res = real_test_inference(test_df, stock_dimension, ALL_INDICATORS, args)
            all_test_results.append(res)

        except Exception as e:
            print(f"[ERROR] Window {i} failed: {e}")
            import traceback

            traceback.print_exc()
            continue

    if all_test_results:
        pd.concat(all_test_results).reset_index(drop=True).to_csv("sac_rolling_results_v2.csv", index=False)
        print(f"\n[SUCCESS] Backtest V2 complete! File saved as sac_rolling_results_v2.csv")
