import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')


def calculate_metrics(df, column_name):
    """计算核心指标"""
    valid_data = df[column_name].dropna()
    if len(valid_data) < 2:
        return 0, 0, 0

    daily_return = valid_data.pct_change().dropna()
    cum_return = (valid_data.iloc[-1] / valid_data.iloc[0]) - 1

    if daily_return.std() != 0:
        sharpe_ratio = (252 ** 0.5) * (daily_return.mean() / daily_return.std())
    else:
        sharpe_ratio = 0

    rolling_max = valid_data.cummax()
    drawdown = (valid_data - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return cum_return, sharpe_ratio, max_drawdown


def main():
    # 📌 设定你要读取的 4 个结果文件
    # (确保这里的 Ensemble 文件名和你本地跑出来的一致，这里用你刚才日志里的 fair_play)
    result_files = {
        'SAC': "sac_rolling_results_v3_fixed.csv",
        'DDPG': "ddpg_rolling_results_v3_fixed.csv",
        'PPO': "ppo_rolling_results_v3_fixed.csv",
        'Ensemble': "ensemble_rolling_results_sharpe_winner.csv"
    }

    # 📌 为不同算法分配固定颜色
    colors = {
        'SAC': 'crimson',  # 猩红色
        'DDPG': 'dodgerblue',  # 亮蓝色
        'PPO': 'forestgreen',  # 森林绿
        'Ensemble': 'darkviolet'  # 深紫色
    }

    dfs = {}
    initial_capital = None

    print("[*] 正在检索算法回测结果...")
    for algo_name, file_path in result_files.items():
        if os.path.exists(file_path):
            print(f"  [+] 找到 {algo_name} 结果文件: {file_path}")
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)

            if initial_capital is None:
                initial_capital = df['account_value'].iloc[0]

            # 提取上包络线并平滑 (过滤瞬间账面波动)
            df['clean_value'] = df['account_value'].rolling(window=7, min_periods=1).max()
            df['clean_value'] = df['clean_value'].ewm(span=3, adjust=False).mean()

            dfs[algo_name] = df[['date', 'clean_value']].rename(columns={'clean_value': algo_name})
        else:
            print(f"  [-] 未找到 {algo_name} 结果文件，将跳过。")

    if not dfs:
        print("[ERROR] 没有找到任何结果文件，请先运行训练脚本！")
        return

    print("\n[*] 正在对齐各算法日期...")
    algo_names = list(dfs.keys())
    master_df = dfs[algo_names[0]][['date']]

    for algo_name, df_algo in dfs.items():
        master_df = pd.merge(master_df, df_algo, on='date', how='outer')

    master_df = master_df.sort_values('date').reset_index(drop=True)

    start_date = master_df['date'].iloc[0].strftime('%Y-%m-%d')
    end_date = (master_df['date'].iloc[-1] + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"[*] 下载道琼斯大盘作为 Benchmark ({start_date} to {end_date})...")
    benchmark = yf.download("^DJI", start=start_date, end=end_date, progress=False)

    if isinstance(benchmark.columns, pd.MultiIndex):
        benchmark.columns = benchmark.columns.get_level_values(0)

    benchmark = benchmark.reset_index()

    if 'Date' not in benchmark.columns and 'index' in benchmark.columns:
        benchmark = benchmark.rename(columns={'index': 'Date'})

    benchmark['Date'] = pd.to_datetime(benchmark['Date'])

    master_df = pd.merge(master_df, benchmark[['Date', 'Close']], left_on='date', right_on='Date', how='left')
    master_df['Close'] = master_df['Close'].ffill()

    master_df['Benchmark'] = (master_df['Close'] / master_df['Close'].iloc[0]) * initial_capital

    print("\n" + "=" * 65)
    print(f"🚀 AI Quant Strategy (Single vs Ensemble) Final Report 🚀")
    print("=" * 65)

    metrics_text = []

    # 1. 计算 AI 算法指标并生成严格对齐的文本 (移除星号)
    for algo_name in algo_names:
        ret, sharpe, mdd = calculate_metrics(master_df, algo_name)
        print(f"[{algo_name:<8} Perf] Return: {ret * 100:>7.2f}% | Sharpe: {sharpe:>6.3f} | Max DD: {mdd * 100:>7.2f}%")

        # 严格的占位符格式化，确保正负号和小数点完美对齐
        line_str = f"{algo_name:<8} Ret: {ret * 100:>+7.2f}% | Shp: {sharpe:>5.2f} | MDD: {mdd * 100:>7.2f}%"
        metrics_text.append(line_str)

    print("-" * 65)

    # 2. 计算大盘指标
    bm_ret, bm_sharpe, bm_mdd = calculate_metrics(master_df, 'Benchmark')
    print(f"[DJI Benchmark ] Return: {bm_ret * 100:>7.2f}% | Sharpe: {bm_sharpe:>6.3f} | Max DD: {bm_mdd * 100:>7.2f}%")
    print("=" * 65)

    # 将 DJI 加入对齐的文本框
    bm_line_str = f"{'DJI':<8} Ret: {bm_ret * 100:>+7.2f}% | Shp: {bm_sharpe:>5.2f} | MDD: {bm_mdd * 100:>7.2f}%"
    metrics_text.append(bm_line_str)

    # ==========================
    # 开始绘图
    # ==========================
    plt.figure(figsize=(16, 8))
    plt.style.use('seaborn-v0_8-darkgrid')

    # 画出各算法曲线
    for algo_name in algo_names:
        plot_data = master_df[algo_name].ffill()
        lw = 3.5 if algo_name == 'Ensemble' else 1.5
        alpha = 1.0 if algo_name == 'Ensemble' else 0.7
        zorder = 5 if algo_name == 'Ensemble' else 3

        plt.plot(master_df['date'], plot_data, label=f'{algo_name} Portfolio',
                 color=colors.get(algo_name, 'black'), linewidth=lw, alpha=alpha, zorder=zorder)

    # 画大盘
    plt.plot(master_df['date'], master_df['Benchmark'], label='Dow Jones (Benchmark)',
             color='slategray', linewidth=2.0, linestyle='--', zorder=2)

    plt.title('DRL Trading Comparison: Single Agents vs. Ensemble Agent',
              fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)

    # 绘制左上角指标信息框 (强制使用等宽字体并整体加粗)
    textstr = '\n'.join(metrics_text)
    props = dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='gray')

    plt.gca().text(0.02, 0.96, textstr, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', bbox=props, family='monospace', fontweight='bold', color='#333333')

    # 🛠️ 修复：使用 framealpha 替代 alpha，并将图例移到左下角
    plt.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=12,
               frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()

    save_filename = 'ultimate_ensemble_comparison.png'
    plt.savefig(save_filename, dpi=300)
    print(f"\n[+] Equity curve comparison saved as '{save_filename}'!")
    plt.show()


if __name__ == "__main__":
    main()
