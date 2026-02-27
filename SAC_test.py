import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')


def calculate_metrics(df, column_name='account_value'):
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
    result_file = "sac_rolling_results_v2.csv"

    try:
        df = pd.read_csv(result_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)

        start_date = df['date'].iloc[0].strftime('%Y-%m-%d')
        end_date = df['date'].iloc[-1].strftime('%Y-%m-%d')
        initial_capital = df['account_value'].iloc[0]

        print(f"[*] Successfully loaded AI trading records! Testing period: {start_date} to {end_date}")
        print("[*] Downloading Dow Jones Index as benchmark...")

        # 下载大盘数据
        benchmark = yf.download("^DJI", start=start_date, end=end_date, progress=False)

        if isinstance(benchmark.columns, pd.MultiIndex):
            benchmark.columns = benchmark.columns.get_level_values(0)

        benchmark = benchmark.reset_index()

        # 统一日期列名
        if 'Date' not in benchmark.columns and 'index' in benchmark.columns:
            benchmark = benchmark.rename(columns={'index': 'Date'})

        benchmark['Date'] = pd.to_datetime(benchmark['Date'])

        # 拼接表格
        df = pd.merge(df, benchmark[['Date', 'Close']], left_on='date', right_on='Date', how='left')
        df['Close'] = df['Close'].ffill()  # 填补周末/节假日的大盘空缺数据

        df['benchmark_value'] = (df['Close'] / df['Close'].iloc[0]) * initial_capital

        # 计算指标
        ai_ret, ai_sharpe, ai_mdd = calculate_metrics(df, 'account_value')
        bm_ret, bm_sharpe, bm_mdd = calculate_metrics(df, 'benchmark_value')

        print("\n" + "=" * 55)
        print(f"🚀 AI Quant Strategy (SAC) Final Report 🚀")
        print("=" * 55)
        print(f"[Cumulative Return]: AI {ai_ret * 100:>7.2f}%  |  Benchmark {bm_ret * 100:>7.2f}%")
        print(f"[Annualized Sharpe]: AI {ai_sharpe:>7.3f}   |  Benchmark {bm_sharpe:>7.3f}")
        print(f"[Maximum Drawdown] : AI {ai_mdd * 100:>7.2f}%  |  Benchmark {bm_mdd * 100:>7.2f}%")
        print("=" * 55)

        # 画图
        plt.figure(figsize=(14, 7))
        plt.style.use('seaborn-v0_8-darkgrid')

        plt.plot(df['date'], df['account_value'], label='AI SAC Agent Portfolio', color='crimson', linewidth=2.5)
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

        plt.savefig('backtest_performance.png', dpi=300)
        print("\n[+] Equity curve saved as 'backtest_performance.png'!")
        plt.show()

    except Exception as e:
        print(f"[ERROR] An error occurred during plotting: {e}")


if __name__ == "__main__":
    main()
