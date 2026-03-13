import os
import pandas as pd
import numpy as np
import torch
import yfinance as yf
import gym
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 🐵 补丁区域
# ==========================================
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

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from elegantrl.agents import AgentPPO, AgentSAC, AgentDDPG
from elegantrl.train.run import train_agent

try:
    from elegantrl.train.config import Config as Arguments
except ImportError:
    from elegantrl.train.config import Arguments


# ==========================================
# 🛡️ 终极环境类
# ==========================================
class ElegantFinRLWrapper(gym.Wrapper):
    def __init__(self, df, stock_dim, hmax, initial_amount, num_stock_shares,
                 buy_cost_pct, sell_cost_pct, reward_scaling, state_space,
                 action_space, tech_indicator_list, env_name, state_dim,
                 action_dim, if_discrete, target_return, penalty_coef=0.001, **kwargs):

        buy_cost_list = buy_cost_pct if isinstance(buy_cost_pct, list) else [0.001] * stock_dim
        sell_cost_list = sell_cost_pct if isinstance(sell_cost_pct, list) else [0.001] * stock_dim

        env = StockTradingEnv(
            df=df, stock_dim=stock_dim, hmax=hmax, initial_amount=initial_amount,
            num_stock_shares=num_stock_shares, buy_cost_pct=buy_cost_list,
            sell_cost_pct=sell_cost_list, reward_scaling=reward_scaling,
            state_space=state_space, action_space=action_space,
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
        self.penalty_coef = penalty_coef

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
            return np.array(obs, dtype=np.float32), float(reward) - float(penalty), term, trunc, info
        else:
            obs, reward, done, info = res
            return np.array(obs, dtype=np.float32), float(reward) - float(penalty), done, False, info


def prepare_data():
    cache_file = "finrl_dow30_cache.csv"
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        return df.drop_duplicates(subset=['date', 'tic'])
    return None


def setup_agent_args(agent_class, env_args, cwd_path):
    args = Arguments(agent_class=agent_class, env_class=ElegantFinRLWrapper)
    args.env_args = env_args
    args.env_name = env_args['env_name']

    # ⚖️ 绝对公平：统一脑容量和批次
    args.net_dims = (128, 64)
    args.batch_size = 128
    args.learning_rate = 1e-4
    args.state_dim = env_args['state_dim']
    args.action_dim = env_args['action_dim']
    args.if_discrete = env_args['if_discrete']

    if agent_class == AgentPPO:
        args.target_step = 2048
        args.repeat_times = 8
    elif agent_class == AgentDDPG:
        args.target_step = 2000
        args.explore_noise = 0.15
    else:
        args.target_step = 2000

    args.worker_num = 1
    args.eval_proc_num = 0
    args.eval_gap = 1000
    args.cwd = cwd_path
    return args


# ==========================================
# 💥 验证集探针 (升级版：计算夏普比率)
# ==========================================
def validate_agent_sharpe(val_df, stock_dim, indicators, agent_class, agent_cwd, env_params):
    env = ElegantFinRLWrapper(**env_params)
    agent = agent_class((128, 64), env_params['state_dim'], env_params['action_dim'])
    agent.save_or_load_agent(agent_cwd, if_save=False)
    agent.act.eval()

    state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
    done = False

    portfolio_values = []
    dates = val_df['date'].unique().tolist()

    while not done:
        current_day = min(env.env.day, len(dates) - 1)
        # 获取真实的现金和持仓市值
        true_cash = float(env.env.state[0])
        true_shares = np.array(env.env.state[1 + stock_dim: 1 + stock_dim * 2])
        current_prices = val_df[val_df['day'] == current_day]['close'].values

        portfolio_values.append(true_cash + np.sum(current_prices * true_shares))

        s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            action = agent.act(s_tensor).detach().cpu().numpy()[0]

        step_res = env.step(action)
        state = step_res[0]
        done = step_res[2] or step_res[3] if len(step_res) == 5 else step_res[2]

    # 确保最后一天也记录
    final_day = min(env.env.day, len(dates) - 1)
    current_prices = val_df[val_df['day'] == final_day]['close'].values
    portfolio_values.append(
        float(env.env.state[0]) + np.sum(current_prices * np.array(env.env.state[1 + stock_dim: 1 + stock_dim * 2])))

    # 📈 计算夏普比率核心逻辑
    if len(portfolio_values) < 2:
        return -1.0

    returns = pd.Series(portfolio_values).pct_change().dropna()

    if returns.std() == 0:
        return -1.0

    # 年化夏普比率 (假设一年 252 个交易日)
    # 加上 1e-9 防止除以 0 的极端情况
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-9)
    return sharpe


# ==========================================
# 👑 动态选马推理 (Winner-Takes-All)
# ==========================================
def ensemble_winner_inference(test_df, stock_dim, indicators, agent_configs, best_algo_name, init_capital, init_shares):
    env_params = agent_configs['env_params']
    env_params['df'] = test_df
    env_params['initial_amount'] = init_capital
    env_params['num_stock_shares'] = init_shares
    env_params['env_name'] = f"ensemble_test_inference_{best_algo_name}"
    env_params['penalty_coef'] = 0.0001

    env = ElegantFinRLWrapper(**env_params)

    config = agent_configs['agents'][best_algo_name]
    agent = config['class']((128, 64), env_params['state_dim'], env_params['action_dim'])
    agent.save_or_load_agent(config['cwd'], if_save=False)
    agent.act.eval()

    state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
    done = False
    account_values = []
    dates = test_df['date'].unique().tolist()

    while not done:
        current_day = min(env.env.day, len(dates) - 1)
        true_cash = float(env.env.state[0])
        true_shares = np.array(env.env.state[1 + stock_dim: 1 + stock_dim * 2])
        current_prices = test_df[test_df['day'] == current_day]['close'].values
        account_values.append(true_cash + np.sum(current_prices * true_shares))

        s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            action = agent.act(s_tensor).detach().cpu().numpy()[0]

        step_res = env.step(action)
        state = step_res[0]
        done = step_res[2] or step_res[3] if len(step_res) == 5 else step_res[2]

    final_day = min(env.env.day, len(dates) - 1)
    current_prices = test_df[test_df['day'] == final_day]['close'].values
    account_values.append(
        float(env.env.state[0]) + np.sum(current_prices * np.array(env.env.state[1 + stock_dim: 1 + stock_dim * 2])))

    actual_steps = min(len(account_values), len(dates))
    df_result = pd.DataFrame({"date": dates[:actual_steps], "account_value": account_values[:actual_steps]})

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

    if df_raw is None:
        print("[ERROR] 无法加载缓存文件，请检查 prepare_data()！")
        exit()

    df_raw = df_raw[df_raw['close'] > 0]
    df_pivot = df_raw.pivot(index='date', columns='tic', values='close')
    df = df_raw[df_raw['date'].isin(df_pivot.replace(0, np.nan).dropna().index)].copy()

    unique_dates = sorted(df['date'].unique())

    # 🚨 防弹自检探针：防止无声失败 🚨
    TRAIN_WINDOW = 252
    VAL_WINDOW = 20
    TEST_WINDOW = 20
    MIN_REQUIRED_DAYS = TRAIN_WINDOW + VAL_WINDOW + 1

    print(f"\n[检查] 原始数据总行数: {len(df_raw)}")
    print(f"[检查] 过滤后剩余有效交易日: {len(unique_dates)} 天")

    if len(unique_dates) < MIN_REQUIRED_DAYS:
        print(f"\n❌ [致命错误] 数据长度不足！")
        print(f"你的有效交易日只有 {len(unique_dates)} 天，但滚动回测至少需要 {MIN_REQUIRED_DAYS} 天。")
        print("请检查你的原始数据时间跨度，或者调小 TRAIN_WINDOW。")
        exit()
    else:
        print(f"✅ 数据检查通过，即将开始回测...\n")

    stock_dim = len(df['tic'].unique())
    ALL_INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30", "close_60"]

    all_test_results = []
    current_capital = 1000000.0
    current_shares = [0] * stock_dim

    base_cwd = "./checkpoints/ensemble_sharpe_testing"
    os.makedirs(base_cwd, exist_ok=True)
    is_first_window = True

    for i in range(TRAIN_WINDOW + VAL_WINDOW, len(unique_dates) - TEST_WINDOW, TEST_WINDOW):
        train_dates = unique_dates[i - TRAIN_WINDOW - VAL_WINDOW: i - VAL_WINDOW]
        val_dates = unique_dates[i - VAL_WINDOW: i]
        test_dates = unique_dates[i: i + TEST_WINDOW]

        train_df = df[df['date'].isin(train_dates)].sort_values(['date', 'tic']).reset_index(drop=True)
        val_df = df[df['date'].isin(val_dates)].sort_values(['date', 'tic']).reset_index(drop=True)
        test_df = df[df['date'].isin(test_dates)].sort_values(['date', 'tic']).reset_index(drop=True)

        for d_df in [train_df, val_df, test_df]:
            d_df['day'] = d_df['date'].map({date: idx for idx, date in enumerate(sorted(d_df['date'].unique()))})
            d_df.set_index('day', inplace=True, drop=False)

        state_dim = 1 + stock_dim * (len(ALL_INDICATORS) + 2)
        env_params = {
            "env_name": f"Window_{i}", "df": train_df, "stock_dim": stock_dim, "hmax": 100,
            "initial_amount": 1000000.0, "num_stock_shares": [0] * stock_dim,
            "buy_cost_pct": [0.001] * stock_dim, "sell_cost_pct": [0.001] * stock_dim,
            "reward_scaling": 1e-4, "state_space": state_dim, "action_space": stock_dim,
            "tech_indicator_list": ALL_INDICATORS, "state_dim": state_dim,
            "action_dim": stock_dim, "if_discrete": False, "target_return": 10.0
        }

        agent_classes = {'PPO': AgentPPO, 'SAC': AgentSAC, 'DDPG': AgentDDPG}
        agent_configs = {'env_params': env_params, 'agents': {}}

        print(f"\n" + "=" * 55)
        print(f"🔄 Rolling Window: {train_dates[0]} to {test_dates[-1]}")

        # 1️⃣ 训练阶段
        for algo_name, AgentClass in agent_classes.items():
            print(f"  [>] Training {algo_name}...")
            cwd = os.path.join(base_cwd, algo_name)
            os.makedirs(cwd, exist_ok=True)

            if algo_name == 'PPO':
                env_params['penalty_coef'] = 0.0
            elif algo_name == 'SAC':
                env_params['penalty_coef'] = 0.001
            elif algo_name == 'DDPG':
                env_params['penalty_coef'] = 0.0001

            args = setup_agent_args(AgentClass, env_params, cwd)

            if is_first_window:
                args.if_remove = True
                args.break_step = 40000
            else:
                args.if_remove = False
                args.break_step = 20000

            train_agent(args)
            agent_configs['agents'][algo_name] = {'class': AgentClass, 'cwd': cwd}

        is_first_window = False

        # 2️⃣ 验证集上评估 (使用 Sharpe Ratio)
        val_sharpes = {}
        print("  [⚖️] Validating models using Sharpe Ratio...")
        for algo_name, config in agent_configs['agents'].items():
            val_params = env_params.copy()
            val_params['df'] = val_df
            val_params['env_name'] = f"val_{algo_name}"
            val_params['initial_amount'] = 1000000.0
            val_params['num_stock_shares'] = [0] * stock_dim
            val_params['penalty_coef'] = 0.0 if algo_name == 'PPO' else (0.001 if algo_name == 'SAC' else 0.0001)

            sharpe = validate_agent_sharpe(val_df, stock_dim, ALL_INDICATORS, config['class'], config['cwd'],
                                           val_params)
            val_sharpes[algo_name] = sharpe

        print(
            f"  [*] Sharpe Scores: PPO {val_sharpes['PPO']:.2f}, SAC {val_sharpes['SAC']:.2f}, DDPG {val_sharpes['DDPG']:.2f}")

        # 👑 动态选马逻辑：寻找夏普比率最高者
        best_algo = max(val_sharpes, key=val_sharpes.get)
        best_sharpe = val_sharpes[best_algo]

        if best_sharpe > 0:
            print(f"  [👑] Winner Takes All: {best_algo} takes 100% control for the next window.")
        else:
            print(f"  [🛡️] All algorithms had negative Sharpe. Defensive fallback: {best_algo} (Least terrible).")

        # 3️⃣ 测试集单边推理
        print(f"  [>] Running Inference with {best_algo}...")
        try:
            res, current_capital, current_shares = ensemble_winner_inference(
                test_df, stock_dim, ALL_INDICATORS, agent_configs, best_algo, current_capital, current_shares
            )
            all_test_results.append(res)
            print(f"  [+] Window Complete. Ending Capital: ${current_capital:.2f}")
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            continue

    if all_test_results:
        final_csv = "ensemble_rolling_results_sharpe_winner.csv"
        pd.concat(all_test_results).reset_index(drop=True).to_csv(final_csv, index=False)
        print(f"\n[SUCCESS] Sharpe-Winner Strategy complete! File saved as {final_csv}")
