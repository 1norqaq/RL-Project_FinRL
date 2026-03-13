import os
import pandas as pd
import numpy as np
import torch
import yfinance as yf
import gym

# ==========================================
# 🐵 补丁 1：解决 PyTorch 2.6 模型加载安全拦截
# ==========================================
original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_torch_load(*args, **kwargs)


torch.load = patched_torch_load

# ==========================================
# 🐵 补丁 2：解决 yfinance 兼容性
# ==========================================
original_download = yf.download


def patched_download(*args, **kwargs):
    kwargs.pop('proxy', None)
    return original_download(*args, **kwargs)


yf.download = patched_download

from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# 💥 修改点 1：引入 PPO
from elegantrl.agents import AgentPPO
from elegantrl.train.run import train_agent

try:
    from elegantrl.train.config import Config as Arguments
except ImportError:
    from elegantrl.train.config import Arguments


# ==========================================
# 🛡️ 终极环境类 (修正 Penalty 比例，唤醒 AI 的交易本能)
# ==========================================
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
        self.penalty_coef = 0.0001

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


def prepare_data():
    cache_file = "finrl_dow30_cache.csv"
    if os.path.exists(cache_file):
        print(f"[INFO] Loading validated cache '{cache_file}'...")
        df = pd.read_csv(cache_file)
        return df.drop_duplicates(subset=['date', 'tic'])
    return None


# 💥 修改点 2：改为 setup_ppo_args，并配置 PPO 专属参数
def setup_ppo_args(env_args, cwd_path):
    args = Arguments(agent_class=AgentPPO, env_class=ElegantFinRLWrapper)
    args.env_args = env_args
    args.env_name = env_args['env_name']

    args.net_dims = (128, 64)
    args.state_dim = env_args['state_dim']
    args.action_dim = env_args['action_dim']
    args.if_discrete = env_args['if_discrete']

    args.learning_rate = 1e-4
    args.batch_size = 128

    # 💡 PPO 核心参数微调
    # PPO 是 On-policy，每次需要收集足够长的数据片段 (trajectory) 才能算出一个比较稳的梯度
    args.target_step = 2048
    # 收集到 target_step 步的数据后，用来重复更新网络的次数（PPO 的精髓所在，数据复用）
    args.repeat_times = 8

    args.break_step = 40000

    args.worker_num = 1
    args.eval_proc_num = 0
    args.eval_gap = 1000

    args.cwd = cwd_path
    args.if_remove = True
    return args


# ==========================================
# 💥 终极修复：绝对防弹的独立账本 (抛弃 FinRL 记账器)
# ==========================================
def real_test_inference(test_df, stock_dim, indicators, args, init_capital, init_shares):
    params = {
        "df": test_df, "stock_dim": stock_dim, "hmax": 100,
        "initial_amount": init_capital,
        "num_stock_shares": init_shares,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4,
        "state_space": 1 + stock_dim * (len(indicators) + 2),
        "action_space": stock_dim, "tech_indicator_list": indicators,
        "env_name": "test_inference", "state_dim": 1 + stock_dim * (len(indicators) + 2),
        "action_dim": stock_dim, "if_discrete": False, "target_return": 10.0
    }
    env = ElegantFinRLWrapper(**params)

    # 💥 修改点 3：推理阶段使用 AgentPPO 加载权重
    agent = AgentPPO(args.net_dims, args.state_dim, args.action_dim)
    agent.save_or_load_agent(args.cwd, if_save=False)
    agent.act.eval()  # 开启 eval 模式，输出确定性动作

    res = env.reset()
    state = res[0] if isinstance(res, tuple) else res

    done = False
    account_values = []
    dates = test_df['date'].unique().tolist()

    while not done:
        # 🛡️ 暴力提取核心资产数据，自己算总账
        current_day = env.env.day
        if current_day >= len(dates):
            current_day = len(dates) - 1

        # 1. 纯现金
        true_cash = float(env.env.state[0])
        # 2. 真实持股数
        true_shares = np.array(env.env.state[1 + stock_dim: 1 + stock_dim * 2])
        # 3. 真实价格！
        current_prices = test_df[test_df['day'] == current_day]['close'].values

        # 组合真实总资产
        true_portfolio_value = true_cash + np.sum(current_prices * true_shares)
        account_values.append(true_portfolio_value)

        s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            action = agent.act(s_tensor).detach().cpu().numpy()[0]

        step_res = env.step(action)
        if len(step_res) == 5:
            state, reward, term, trunc, _ = step_res
            done = term or trunc
        else:
            state, reward, done, _, _ = step_res

    # 补齐最后一天结账
    final_day = env.env.day
    if final_day >= len(dates): final_day = len(dates) - 1
    true_cash = float(env.env.state[0])
    true_shares = np.array(env.env.state[1 + stock_dim: 1 + stock_dim * 2])
    current_prices = test_df[test_df['day'] == final_day]['close'].values

    final_portfolio_value = true_cash + np.sum(current_prices * true_shares)
    account_values.append(final_portfolio_value)

    actual_steps = min(len(account_values), len(dates))
    df_result = pd.DataFrame({
        "date": dates[:actual_steps],
        "account_value": account_values[:actual_steps]
    })

    # 提取传给下个窗口的状态
    final_state = env.env.state
    final_capital = float(final_state[0])
    final_shares_list = final_state[1 + stock_dim: 1 + stock_dim * 2]

    if isinstance(final_shares_list, np.ndarray):
        final_shares_list = final_shares_list.tolist()
    elif not isinstance(final_shares_list, list):
        final_shares_list = list(final_shares_list)

    return df_result, final_capital, final_shares_list


if __name__ == '__main__':
    df_raw = prepare_data()

    # 清洗日历与 0 价格
    df_raw = df_raw[df_raw['close'] > 0]
    df_pivot = df_raw.pivot(index='date', columns='tic', values='close')
    valid_dates = df_pivot.replace(0, np.nan).dropna().index
    df = df_raw[df_raw['date'].isin(valid_dates)].copy()

    unique_dates = sorted(df['date'].unique())
    stock_dimension = len(df['tic'].unique())
    ALL_INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30", "close_60"]

    TRAIN_WINDOW = 252
    VAL_WINDOW = 20
    TEST_WINDOW = 20

    all_test_results = []

    current_capital = 1000000.0
    current_shares = [0] * stock_dimension

    # 💥 修改点 4：修改 checkpoint 保存目录，防止覆盖之前的模型
    global_cwd = "./checkpoints/ppo_continuous_learning"
    os.makedirs(global_cwd, exist_ok=True)
    is_first_window = True

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
            "initial_amount": 1000000.0,
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
            # 💥 修改点 5：调用 setup_ppo_args
            args = setup_ppo_args(env_params, global_cwd)

            if is_first_window:
                args.if_remove = True
                # PPO 往往需要稍微多一点步数来收敛，可以视情况调大 break_step
                args.break_step = 40000
                is_first_window = False
            else:
                args.if_remove = False
                args.break_step = 20000

            print("[-->] Starting PPO Agent Training (Continual Learning)...")
            train_agent(args)

            print(f"[-->] Starting Testing with Capital: ${current_capital:.2f}")
            res, current_capital, current_shares = real_test_inference(
                test_df, stock_dimension, ALL_INDICATORS, args, current_capital, current_shares
            )

            all_test_results.append(res)
            print(f"[INFO] Window Test Complete. Ending Capital: ${current_capital:.2f}")

        except Exception as e:
            print(f"[ERROR] Window {i} failed: {e}")
            import traceback

            traceback.print_exc()
            continue

    if all_test_results:
        # 💥 修改点 6：修改最终结果保存的文件名
        pd.concat(all_test_results).reset_index(drop=True).to_csv("ppo_rolling_results_v3_fixed.csv", index=False)
        print(f"\n[SUCCESS] Backtest V3 Fixed complete! File saved as ppo_rolling_results_v3_fixed.csv")
