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
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from elegantrl.agents import AgentSAC
from elegantrl.train.run import train_agent

try:
    from elegantrl.train.config import Config as Arguments
except ImportError:
    from elegantrl.train.config import Arguments


# ---------------------------------------------------------
# 3. 内置环境包装器 (Wrapper)，解决兼容性与惩罚机制
# ---------------------------------------------------------
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

        # 记录上一次的动作，用于计算调仓惩罚 (Action Penalty)
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

        # 计算乱动症惩罚
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


print("[✓] All imports and wrappers successful! Ready for SAC.")


# ---------------------------------------------------------
# 4. 数据准备与分割
# ---------------------------------------------------------
def prepare_data(cache_file="./data/finrl_dow30_cache.csv",
                 TRAIN_START_DATE='2010-01-01',
                 TEST_END_DATE='2023-12-30'):
    """下载数据、添加技术指标并缓存"""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    if os.path.exists(cache_file):
        print(f"[INFO] Loading validated cache '{cache_file}'...")
        try:
            df = pd.read_csv(cache_file)
            df = df.drop_duplicates(subset=['date', 'tic'])
            if len(df) > 0:
                print(f"[✓] Cache loaded: {len(df)} records, Date range: {df['date'].min()} to {df['date'].max()}")
                return df
            else:
                print("[!] Cache file is empty, will re-download...")
        except Exception as e:
            print(f"[!] Error loading cache: {e}, will re-download...")

    print(f"[INFO] Downloading Dow 30 data from {TRAIN_START_DATE} to {TEST_END_DATE}...")
    df = YahooDownloader(start_date=TRAIN_START_DATE,
                         end_date=TEST_END_DATE,
                         ticker_list=DOW_30_TICKER).fetch_data()

    print(f"[✓] Data preparation complete!")
    print(f"    - Total records: {len(df)}")
    print(f"    - Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"    - Unique tickers: {df['tic'].nunique()}")

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

        print(f"[✓] Technical indicators calculated!")

    except Exception as e:
        print(f"[!] Error calculating technical indicators: {e}")
        print(f"[!] Attempting alternative method using 'ta' library...")
        import ta
        df_list = []
        for ticker in df['tic'].unique():
            ticker_df = df[df['tic'] == ticker].copy()
            ticker_df = ticker_df.sort_values('date')

            ticker_df['macd'] = ta.trend.macd_diff(ticker_df['close'])
            bb = ta.volatility.BollingerBands(ticker_df['close'])
            ticker_df['boll_ub'] = bb.bollinger_hband()
            ticker_df['boll_lb'] = bb.bollinger_lband()

            ticker_df['rsi_30'] = ta.momentum.rsi(ticker_df['close'], window=30)
            ticker_df['cci_30'] = ta.trend.cci(ticker_df['high'], ticker_df['low'], ticker_df['close'], window=30)
            ticker_df['dx_30'] = ta.trend.adx(ticker_df['high'], ticker_df['low'], ticker_df['close'], window=30)
            ticker_df['close_30'] = ticker_df['close'].rolling(window=30).mean()
            ticker_df['close_60'] = ticker_df['close'].rolling(window=60).mean()

            df_list.append(ticker_df)

        df = pd.concat(df_list, ignore_index=True).dropna()
        print(f"[✓] Technical indicators calculated using ta library!")

    if df is not None and not df.empty:
        df.to_csv(cache_file, index=False)
        print(f"[✓] Data saved to cache: {cache_file}")
    else:
        print("[ERROR] Failed to download data or received empty dataframe")
        return None
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
# 5. SAC 模型参数配置与推断函数
# ---------------------------------------------------------
def setup_sac_args(env_args, cwd_path):
    args = Arguments(agent_class=AgentSAC, env_class=ElegantFinRLWrapper)
    args.env_args = env_args
    args.env_name = env_args['env_name']

    args.net_dims = (128, 64)
    args.state_dim = env_args['state_dim']
    args.action_dim = env_args['action_dim']
    args.if_discrete = env_args['if_discrete']

    args.learning_rate = 1e-4
    args.batch_size = 128

    args.target_step = 2000
    args.break_step = 40000

    args.worker_num = 1
    args.eval_proc_num = 0
    args.if_use_multi_processing = False
    args.eval_gap = 500
    args.save_gap = 500

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
        "action_dim": stock_dim, "if_discrete": False, "target_return": 10.0
    }
    env = ElegantFinRLWrapper(**params)

    agent = AgentSAC(args.net_dims, args.state_dim, args.action_dim)
    # 若需读取已存模型，可取消注释：agent.save_or_load_agent(args.cwd, if_save=False)
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


def plot_test_results(result_file="sac_test_results.csv"):
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

        print(f"[*] Loading SAC test results...")
        print(f"[*] Downloading Dow Jones benchmark data...")

        benchmark = yf.download("^DJI", start=start_date, end=end_date, progress=False)
        if isinstance(benchmark.columns, pd.MultiIndex):
            benchmark.columns = benchmark.columns.get_level_values(0)
        benchmark = benchmark.reset_index()
        if 'Date' not in benchmark.columns and 'index' in benchmark.columns:
            benchmark = benchmark.rename(columns={'index': 'Date'})
        benchmark['Date'] = pd.to_datetime(benchmark['Date'])

        df = pd.merge(df, benchmark[['Date', 'Close']], left_on='date', right_on='Date', how='left')
        df['Close'] = df['Close'].ffill()
        df['benchmark_value'] = (df['Close'] / df['Close'].iloc[0]) * initial_capital

        sac_ret, sac_sharpe, sac_mdd = calculate_metrics(df, 'account_value')
        bm_ret, bm_sharpe, bm_mdd = calculate_metrics(df, 'benchmark_value')

        excess_return = (sac_ret - bm_ret) * 100
        sharpe_diff = sac_sharpe - bm_sharpe

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8))

        # Plot 1: Portfolio Value Comparison
        ax1.plot(df['date'], df['account_value'], label='SAC Agent Portfolio', color='#2E86AB', linewidth=2.5)
        ax1.plot(df['date'], df['benchmark_value'], label='Dow Jones Index (Benchmark)', color='#A23B72', linewidth=2,
                 linestyle='--', alpha=0.8)
        ax1.set_title('SAC Agent Test Period Performance vs Benchmark', fontsize=16, fontweight='bold', pad=15)
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right', fontsize=11, framealpha=0.95)

        textstr = '\n'.join((
            r'$\bf{SAC\ Strategy}$',
            f'Return: {sac_ret * 100:.2f}%',
            f'Sharpe: {sac_sharpe:.3f}',
            f'Max DD: {sac_mdd * 100:.2f}%',
            '',
            r'$\bf{Benchmark}$',
            f'Return: {bm_ret * 100:.2f}%',
            f'Sharpe: {bm_sharpe:.3f}',
            f'Max DD: {bm_mdd * 100:.2f}%'
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props,
                 family='monospace')

        # Plot 2: Daily Returns Comparison
        sac_returns = df['account_value'].pct_change() * 100
        bm_returns = df['benchmark_value'].pct_change() * 100

        ax2.plot(df['date'], sac_returns, label='SAC Daily Return', color='#2E86AB', linewidth=1.5, alpha=0.7)
        ax2.plot(df['date'], bm_returns, label='Benchmark Daily Return', color='#A23B72', linewidth=1.5, alpha=0.7,
                 linestyle='--')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax2.set_title('Daily Returns Comparison', fontsize=14, fontweight='bold', pad=10)
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Daily Return (%)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right', fontsize=10, framealpha=0.95)

        plt.tight_layout()
        fig_path = 'sac_test_analysis.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n[✓] Test results visualization saved to '{fig_path}'!")
        plt.show()

    except Exception as e:
        print(f"[ERROR] Failed to plot test results: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------
# 7. 主程序入口 (Main Execution)
# ---------------------------------------------------------
if __name__ == '__main__':
    print("\n[STEP 1] Loading stock data...")
    df = prepare_data()
    df_train, df_test = split_train_test(df)

    ALL_INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30", "close_60"]

    stock_dimension = len(df_train['tic'].unique())
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
        "env_name": "FinRL_SAC_Train",
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
        "action_dim": stock_dimension,
        "if_discrete": False,
        "target_return": 10.0
    }

    print(f"\n[STEP 2] Starting SAC Agent Training on Full Dataset...")

    try:
        # 模型文件保存路径
        cwd_path = "./checkpoints/sac_full_train"
        os.makedirs(cwd_path, exist_ok=True)

        args = setup_sac_args(env_params, cwd_path)

        print("[-->] Starting SAC Agent Training...")
        train_agent(args)
        print("[✓] Training completed successfully!")

        print(f"\n[STEP 3] Running inference on test data...")
        test_results = real_test_inference(df_test, stock_dimension, ALL_INDICATORS, args)

        result_csv_path = "sac_test_results.csv"
        test_results.to_csv(result_csv_path, index=False)
        print(f"[✓] Test results saved to '{result_csv_path}'")

        print("\n[STEP 4] Test Results Visualization")
        plot_test_results(result_file=result_csv_path)

    except Exception as e:
        print(f"[ERROR] Execution failed: {e}")
        import traceback

        traceback.print_exc()
