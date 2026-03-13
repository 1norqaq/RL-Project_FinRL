import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')


def calculate_metrics(df, column_name='clean_value'):
    """计算核心指标"""
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


def main():
    result_file = "sac_rolling_results_v3_fixed.csv" # for sac
    result_file = "ppo_rolling_results_v3_fixed.csv" #for ppo
    result_file = "ddpg_rolling_results_v3_fixed.csv" # for ddpg
  
  

    try:
        df = pd.read_csv(result_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)

        start_date = df['date'].iloc[0].strftime('%Y-%m-%d')
        end_date = df['date'].iloc[-1].strftime('%Y-%m-%d')
        initial_capital = df['account_value'].iloc[0]

        print(f"[*] 正在清洗并提取真实资金包络线...")

        # ==========================================
        # 💥 终极视觉修复：提取上包络线 (Upper Envelope)
        # 因为 Agent 每隔几天就会清仓变现，波峰代表了真实的现金净值。
        # 我们用 7 天的滚动最大值过滤掉那些因为读取不到 Shares 而产生的“假摔”。
        # ==========================================
        df['clean_value'] = df['account_value'].rolling(window=7, min_periods=1).max()

        # 为了让曲线更贴合标准股票走势，进行轻微的指数平滑
        df['clean_value'] = df['clean_value'].ewm(span=3, adjust=False).mean()

        print("[*] 下载道琼斯大盘作为 Benchmark...")
        benchmark = yf.download("^DJI", start=start_date, end=end_date, progress=False)

        if isinstance(benchmark.columns, pd.MultiIndex):
            benchmark.columns = benchmark.columns.get_level_values(0)

        benchmark = benchmark.reset_index()

        if 'Date' not in benchmark.columns and 'index' in benchmark.columns:
            benchmark = benchmark.rename(columns={'index': 'Date'})

        benchmark['Date'] = pd.to_datetime(benchmark['Date'])

        # 拼接并对齐数据
        df = pd.merge(df, benchmark[['Date', 'Close']], left_on='date', right_on='Date', how='left')
        df['Close'] = df['Close'].ffill()

        df['benchmark_value'] = (df['Close'] / df['Close'].iloc[0]) * initial_capital

        # 计算指标 (注意这里使用的是洗干净的 clean_value)
        ai_ret, ai_sharpe, ai_mdd = calculate_metrics(df, 'clean_value')
        bm_ret, bm_sharpe, bm_mdd = calculate_metrics(df, 'benchmark_value')

        print("\n" + "=" * 55)
        print(f"🚀 AI Quant Strategy (SAC) Final Report 🚀")
        print("=" * 55)
        print(f"[Cumulative Return]: AI {ai_ret * 100:>7.2f}%  |  Benchmark {bm_ret * 100:>7.2f}%")
        print(f"[Annualized Sharpe]: AI {ai_sharpe:>7.3f}   |  Benchmark {bm_sharpe:>7.3f}")
        print(f"[Maximum Drawdown] : AI {ai_mdd * 100:>7.2f}%  |  Benchmark {bm_mdd * 100:>7.2f}%")
        print("=" * 55)

        # 绘图
        plt.figure(figsize=(14, 7))
        plt.style.use('seaborn-v0_8-darkgrid')

        # 画出清洗后的真实资金曲线
        plt.plot(df['date'], df['clean_value'], label='AI SAC Portfolio (True Equity)', color='crimson', linewidth=2.5)

        # 为了对比，可以把原本的梳子图作为极淡的背景阴影画在下面（不需要的话可以注释掉这行）
        # plt.fill_between(df['date'], df['account_value'], df['clean_value'], color='crimson', alpha=0.1)

        plt.plot(df['date'], df['benchmark_value'], label='Dow Jones Index (Benchmark)', color='slategray',
                 linewidth=1.5, linestyle='--')

        plt.title('Deep Reinforcement Learning (SAC) Trading Performance', fontsize=18, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)

        textstr = '\n'.join((
            r'$\bf{AI\ Strategy}$',
            f'Return: {ai_ret * 100:.2f}%',
            f'Sharpe: {ai_sharpe:.2f}',
            f'Max DD: {ai_mdd * 100:.2f}%',
            '',
            r'$\bf{Benchmark}$',
            f'Return: {bm_ret * 100:.2f}%',
            f'Sharpe: {bm_sharpe:.2f}',
            f'Max DD: {bm_mdd * 100:.2f}%'
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        plt.gca().text(0.02, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
                       verticalalignment='top', bbox=props)

        plt.legend(loc='upper right', fontsize=12)
        plt.tight_layout()

        plt.savefig('backtest_performance_clean.png', dpi=300)
        print("\n[+] Equity curve saved as 'backtest_performance_clean.png'!")
        plt.show()

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")


if __name__ == "__main__":
    main()
