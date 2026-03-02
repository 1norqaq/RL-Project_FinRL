import os
import pandas as pd
import numpy as np
import torch
import yfinance as yf
import gymnasium as gym
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. 安全补丁与雅虎财经下载补丁
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. 导入必要的 FinRL 与 ElegantRL 组件
# ---------------------------------------------------------
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# 【核心切换】：导入 Double DQN 智能体
from elegantrl.agents import AgentDoubleDQN
from elegantrl.train.run import train_agent

try:
    from elegantrl.train.config import Config as Arguments
except ImportError:
    from elegantrl.train.config import Arguments


# ---------------------------------------------------------
# 3. 离散动作环境包装器 (Discrete FinRL Wrapper)
# ---------------------------------------------------------
class DiscreteFinRLWrapper(gym.Wrapper):
    """
    专为 DQN 等离散动作算法设计的环境包装器。
    将 DQN 输出的离散动作 (0, 1, 2) 翻译为 FinRL 底层需要的连续交易份额。
    """

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
        # 强制设定为 3 个离散动作：0 (Sell), 1 (Hold), 2 (Buy)
        self.action_dim = 3
        self.if_discrete = True
        self.target_return = target_return
        self.max_step = len(df.index.unique())

    def reset(self):
        res = self.env.reset()
        obs = res[0] if isinstance(res, tuple) else res
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        # 将 DQN 的离散动作转换为连续买卖份额
        # action == 0 -> -1 (卖出)
        # action == 1 ->  0 (不动)
        # action == 2 ->  1 (买入)
        action_val = int(action) - 1

        # 乘以 hmax，表示要么满仓买，要么全仓卖
        cont_action = np.array([action_val * self.env.hmax], dtype=np.float32)

        res = self.env.step(cont_action)

        if len(res) == 5:
            obs, reward, term, trunc, info = res
            return np.array(obs, dtype=np.float32), float(reward), term, trunc, info
        else:
            obs, reward, done, info = res
            return np.array(obs, dtype=np.float32), float(reward), done, False, info


print("[✓] Environment and imports ready for Double DQN!")


# ---------------------------------------------------------
# 4. 数据准备与分割
# ---------------------------------------------------------
def prepare_data(cache_file="./data/finrl_dia_cache.csv",
                 TRAIN_START_DATE='2010-01-01',
                 TEST_END_DATE='2023-12-30'):
    """仅下载道琼斯指数 ETF (DIA)"""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    if os.path.exists(cache_file):
        print(f"[INFO] Loading validated cache '{cache_file}'...")
        try:
            df = pd.read_csv(cache_file)
            df = df.drop_duplicates(subset=['date', 'tic'])
            if len(df) > 0:
                print(f"[✓] Cache loaded: {len(df)} records.")
                return df
        except Exception as e:
            print(f"[!] Error loading cache: {e}, will re-download...")

    print(f"[INFO] Downloading DIA ETF data from {TRAIN_START_DATE} to {TEST_END_DATE}...")
    # 只交易一只股票：DIA (道琼斯大盘 ETF)
    df = YahooDownloader(start_date=TRAIN_START_DATE,
                         end_date=TEST_END_DATE,
                         ticker_list=["DIA"]).fetch_data()

    print("\n[STEP] Calculating technical indicators...")
    TECHNICAL_INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]

    try:
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=TECHNICAL_INDICATORS,
            use_vix=False,
            use_turbulence=False,
            user_defined_feature=False
        )
        df = fe.preprocess_data(df)

        if 'close_30_sma' in df.columns:
            df = df.rename(columns={'close_30_sma': 'close_30'})
        if 'close_60_sma' in df.columns:
            df = df.rename(columns={'close_60_sma': 'close_60'})
    except Exception as e:
        print(f"[!] Error calculating indicators: {e}")

    if df is not None and not df.empty:
        df.to_csv(cache_file, index=False)
        print(f"[✓] Data saved to cache: {cache_file}")
    return df


def split_train_test(df,
                     TRAIN_START_DATE='2010-01-07',
                     TRAIN_END_DATE='2023-10-24',
                     TEST_START_DATE='2023-10-25',
                     TEST_END_DATE='2023-11-21'):
    df['date'] = pd.to_datetime(df['date'])
    df_train = df[(df['date'] >= TRAIN_START_DATE) & (df['date'] <= TRAIN_END_DATE)].copy()
    df_test = df[(df['date'] >= TEST_START_DATE) & (df['date'] <= TEST_END_DATE)].copy()

    print(f"[INFO] Train: {df_train['date'].min()} → {df_train['date'].max()}, {len(df_train)} rows")
    print(f"[INFO] Test:  {df_test['date'].min()} → {df_test['date'].max()}, {len(df_test)} rows")
    return df_train, df_test


# ---------------------------------------------------------
# 5. Double DQN 模型参数配置与推断函数
# ---------------------------------------------------------
def setup_ddqn_args(env_args, cwd_path):
    # 使用 AgentDoubleDQN 替代原来的 AgentSAC
    args = Arguments(agent_class=AgentDoubleDQN, env_class=DiscreteFinRLWrapper)
    args.env_args = env_args
    args.env_name = env_args['env_name']

    args.net_dims = (128, 64)
    args.state_dim = env_args['state_dim']
    args.action_dim = env_args['action_dim']
    args.if_discrete = True  # 开启离散模式

    args.learning_rate = 1e-4
    args.batch_size = 128

    args.target_step = 2000
    args.break_step = 40000

    args.worker_num = 1
    args.eval_proc_num = 0
    args.eval_gap = 500

    args.if_save = True
    args.if_overwrite_save = True

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
        "action_dim": 3,  # 测试环境同样定义为 3 个离散动作
        "if_discrete": True, "target_return": 10.0
    }
    env = DiscreteFinRLWrapper(**params)

    agent = AgentDoubleDQN(args.net_dims, args.state_dim, args.action_dim)
    agent.act.eval()

    res = env.reset()
    state = res[0] if isinstance(res, tuple) else res

    done = False
    while not done:
        s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            # DDQN 预测输出离散的 action index
            action_tensor = agent.act(s_tensor)
            if action_tensor.dim() > 1:
                action = action_tensor.argmax(dim=-1).cpu().numpy()[0]
            else:
                action = action_tensor.cpu().numpy()[0]
            action = int(action)

        step_res = env.step(action)
        if len(step_res) == 5:
            state, reward, term, trunc, _ = step_res
            done = term or trunc
        else:
            state, reward, done, _, _ = step_res

    return env.env.save_asset_memory()


# ---------------------------------------------------------
# 6. 指标计算与画图函数
# ---------------------------------------------------------
def calculate_metrics(df, column_name='account_value'):
    daily_return = df[column_name].pct_change().dropna()
    cum_return = (df[column_name].iloc[-1] / df[column_name].iloc[0]) - 1
    if daily_return.std() != 0:
        sharpe_ratio = (252 ** 0.5) * (daily_return.mean() / daily_return.std())
    else:
        sharpe_ratio = 0
    rolling_max = df[column_name].cummax()
    drawdown = (df[column_name] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    return cum_return, sharpe_ratio, max_drawdown


def plot_test_results(result_file="ddqn_test_results.csv"):
    try:
        if not os.path.exists(result_file):
            print(f"[!] Test results file not found: {result_file}")
            return

        df = pd.read_csv(result_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)

        start_date = df['date'].iloc[0].strftime('%Y-%m-%d')
        end_date = df['date'].iloc[-1].strftime('%Y-%m-%d')
        initial_capital = df['account_value'].iloc[0]

        print(f"[*] Downloading DIA benchmark data...")
        benchmark = yf.download("DIA", start=start_date, end=end_date, progress=False)
        if isinstance(benchmark.columns, pd.MultiIndex):
            benchmark.columns = benchmark.columns.get_level_values(0)
        benchmark = benchmark.reset_index()
        if 'Date' not in benchmark.columns and 'index' in benchmark.columns:
            benchmark = benchmark.rename(columns={'index': 'Date'})
        benchmark['Date'] = pd.to_datetime(benchmark['Date'])

        df = pd.merge(df, benchmark[['Date', 'Close']], left_on='date', right_on='Date', how='left')
        df['Close'] = df['Close'].ffill()
        df['benchmark_value'] = (df['Close'] / df['Close'].iloc[0]) * initial_capital

        ddqn_ret, ddqn_sharpe, ddqn_mdd = calculate_metrics(df, 'account_value')
        bm_ret, bm_sharpe, bm_mdd = calculate_metrics(df, 'benchmark_value')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8))

        ax1.plot(df['date'], df['account_value'], label='Double DQN Portfolio', color='#F79256', linewidth=2.5)
        ax1.plot(df['date'], df['benchmark_value'], label='DIA ETF (Benchmark)', color='#7DDF64', linewidth=2,
                 linestyle='--', alpha=0.8)
        ax1.set_title('Double DQN Single Asset Trading Performance (DIA)', fontsize=16, fontweight='bold', pad=15)
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right', fontsize=11, framealpha=0.95)

        textstr = '\n'.join((
            r'$\bf{Double\ DQN}$',
            f'Return: {ddqn_ret * 100:.2f}%',
            f'Sharpe: {ddqn_sharpe:.3f}',
            f'Max DD: {ddqn_mdd * 100:.2f}%',
            '',
            r'$\bf{Benchmark(DIA)}$',
            f'Return: {bm_ret * 100:.2f}%',
            f'Sharpe: {bm_sharpe:.3f}',
            f'Max DD: {bm_mdd * 100:.2f}%'
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props,
                 family='monospace')

        ddqn_returns = df['account_value'].pct_change() * 100
        bm_returns = df['benchmark_value'].pct_change() * 100

        ax2.plot(df['date'], ddqn_returns, label='DDQN Daily Return', color='#F79256', linewidth=1.5, alpha=0.7)
        ax2.plot(df['date'], bm_returns, label='DIA Daily Return', color='#7DDF64', linewidth=1.5, alpha=0.7,
                 linestyle='--')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax2.set_title('Daily Returns Comparison', fontsize=14, fontweight='bold', pad=10)
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Daily Return (%)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right', fontsize=10, framealpha=0.95)

        plt.tight_layout()
        fig_path = 'ddqn_dia_test_analysis.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n[✓] Test results visualization saved to '{fig_path}'!")
        plt.show()

    except Exception as e:
        print(f"[ERROR] Failed to plot test results: {e}")


# ---------------------------------------------------------
# 7. 主程序入口 (Main Execution)
# ---------------------------------------------------------
if __name__ == '__main__':
    print("\n[STEP 1] Loading stock data (DIA Only)...")
    df = prepare_data()
    df_train, df_test = split_train_test(df)

    ALL_INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30", "close_60"]

    # 只交易 1 只股票
    stock_dimension = 1
    # 状态空间维度：1 (初始本金) + 1 * (指标数量8 + 2(当前价格与持股数)) = 11
    state_dim = 1 + stock_dimension * (len(ALL_INDICATORS) + 2)

    train_dates = sorted(df_train['date'].unique())
    date_map_train = {date: idx for idx, date in enumerate(train_dates)}
    df_train['day'] = df_train['date'].map(date_map_train)
    df_train.set_index('day', inplace=True, drop=False)

    test_dates = sorted(df_test['date'].unique())
    date_map_test = {date: idx for idx, date in enumerate(test_dates)}
    df_test['day'] = df_test['date'].map(date_map_test)
    df_test.set_index('day', inplace=True, drop=False)

    env_params = {
        "env_name": "FinRL_DDQN_Train",
        "df": df_train,
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
        "action_dim": 3,  # 动作维度变为固定的 3 (卖、不动、买)
        "if_discrete": True,
        "target_return": 10.0
    }

    print(f"\n[STEP 2] Starting Double DQN Agent Training...")

    try:
        cwd_path = "./checkpoints/ddqn_dia_train"
        os.makedirs(cwd_path, exist_ok=True)

        args = setup_ddqn_args(env_params, cwd_path)

        print("[-->] Starting Training Loop...")
        train_agent(args)
        print("[✓] Training completed successfully!")

        print(f"\n[STEP 3] Running inference on test data...")
        test_results = real_test_inference(df_test, stock_dimension, ALL_INDICATORS, args)

        result_csv_path = "ddqn_test_results.csv"
        test_results.to_csv(result_csv_path, index=False)
        print(f"[✓] Test results saved to '{result_csv_path}'")

        print("\n[STEP 4] Test Results Visualization")
        plot_test_results(result_file=result_csv_path)

    except Exception as e:
        print(f"[ERROR] Execution failed: {e}")
        import traceback

        traceback.print_exc()
