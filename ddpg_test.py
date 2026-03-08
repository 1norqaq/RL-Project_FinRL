# ddpg_test.py
import argparse
import glob
import os
import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def calculate_metrics(series):
    values = pd.Series(series).dropna()
    if len(values) < 2:
        return 0.0, 0.0, 0.0

    daily_return = values.pct_change().dropna()
    cum_return = float(values.iloc[-1] / values.iloc[0] - 1.0)
    sharpe_ratio = float((252 ** 0.5) * (daily_return.mean() / daily_return.std())) if daily_return.std() != 0 else 0.0
    drawdown = values / values.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    return cum_return, sharpe_ratio, max_drawdown


def normalize_benchmark_df(benchmark_df):
    if benchmark_df is None or benchmark_df.empty:
        return pd.DataFrame(columns=["Date", "Close"])

    if isinstance(benchmark_df.columns, pd.MultiIndex):
        benchmark_df.columns = benchmark_df.columns.get_level_values(0)

    benchmark_df = benchmark_df.reset_index()
    if "Date" not in benchmark_df.columns and "index" in benchmark_df.columns:
        benchmark_df = benchmark_df.rename(columns={"index": "Date"})

    if "Date" not in benchmark_df.columns:
        return pd.DataFrame(columns=["Date", "Close"])

    if "Close" not in benchmark_df.columns and "Adj Close" in benchmark_df.columns:
        benchmark_df = benchmark_df.rename(columns={"Adj Close": "Close"})

    if "Close" not in benchmark_df.columns:
        return pd.DataFrame(columns=["Date", "Close"])

    benchmark_df["Date"] = pd.to_datetime(benchmark_df["Date"])
    benchmark_df = benchmark_df[["Date", "Close"]].dropna(subset=["Close"]).drop_duplicates(subset=["Date"])
    return benchmark_df.sort_values("Date").reset_index(drop=True)


def load_cached_benchmark(cache_file, start_date, end_date):
    if not os.path.exists(cache_file):
        return pd.DataFrame(columns=["Date", "Close"])
    try:
        cached = pd.read_csv(cache_file)
    except Exception:
        return pd.DataFrame(columns=["Date", "Close"])

    cached = normalize_benchmark_df(cached)
    if cached.empty:
        return cached

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    cached = cached[(cached["Date"] >= start_ts) & (cached["Date"] <= end_ts)]
    return cached.reset_index(drop=True)


def fetch_benchmark_with_retry(start_date, end_date, cache_file="benchmark_dji_cache.csv", max_retries=5):
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            benchmark = yf.download(
                "^DJI",
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            benchmark = normalize_benchmark_df(benchmark)
            if not benchmark.empty:
                if os.path.exists(cache_file):
                    old = load_cached_benchmark(cache_file, "1900-01-01", "2100-01-01")
                    benchmark = (
                        pd.concat([old, benchmark], ignore_index=True)
                        .drop_duplicates(subset=["Date"], keep="last")
                        .sort_values("Date")
                        .reset_index(drop=True)
                    )
                benchmark.to_csv(cache_file, index=False)
                period = load_cached_benchmark(cache_file, start_date, end_date)
                if not period.empty:
                    return period
        except Exception as exc:
            last_error = exc

        if attempt < max_retries:
            wait_s = min(45, 5 * attempt + random.randint(0, 4))
            print(f"[WARN] Benchmark download failed (attempt {attempt}/{max_retries}). Retrying in {wait_s}s...")
            time.sleep(wait_s)

    cached = load_cached_benchmark(cache_file, start_date, end_date)
    if not cached.empty:
        print(f"[WARN] Using cached benchmark from '{cache_file}' due to Yahoo rate limit.")
        return cached

    if last_error is not None:
        raise RuntimeError(f"Benchmark download failed after {max_retries} attempts: {last_error}")
    raise RuntimeError(f"Benchmark download failed after {max_retries} attempts and no cache is available.")


def find_stock_cache():
    candidates = [
        "./data/finrl_dow30_cache.csv",
        "./finrl_dow30_cache.csv",
        "../data/finrl_dow30_cache.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    for root in [os.getcwd(), os.path.abspath(os.path.join(os.getcwd(), ".."))]:
        pattern = os.path.join(root, "**", "finrl_dow30_cache.csv")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None


def load_result_df(result_file, default_window_len):
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Result file '{result_file}' not found. Run ddpg_train.py first.")

    df = pd.read_csv(result_file)
    required_columns = {"date", "account_value"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Result file '{result_file}' is missing required columns: {missing_columns}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    if "window_seq" in df.columns:
        df["window_seq"] = df["window_seq"].astype(int)
    elif "window_id" in df.columns:
        order = pd.Series(df["window_id"]).astype(str)
        df["window_seq"] = pd.factorize(order)[0]
    else:
        # Backward-compatible fallback for old result files.
        if len(df) % default_window_len != 0:
            raise ValueError(
                f"Result file has {len(df)} rows, not divisible by window length {default_window_len}. "
                "Provide a result file with window metadata (window_seq/window_id) or the correct --window-len."
            )
        df["window_seq"] = np.arange(len(df)) // default_window_len

    return df


def add_compounded_curve(df, value_col, out_col, start_capital):
    result = []
    running_capital = float(start_capital)

    for _, group in df.groupby("window_seq", sort=True):
        g = group.copy()
        base = float(g[value_col].iloc[0])
        if base == 0:
            g[out_col] = running_capital
        else:
            g[out_col] = running_capital * (g[value_col] / base)
        running_capital = float(g[out_col].iloc[-1])
        result.append(g)

    out = pd.concat(result, ignore_index=True).sort_values("date").reset_index(drop=True)
    return out


def build_window_benchmark_series(df, close_col, out_col, start_capital):
    values = []
    for _, group in df.groupby("window_seq", sort=True):
        g = group.copy()
        c0 = float(g[close_col].iloc[0])
        if c0 == 0:
            g[out_col] = start_capital
        else:
            g[out_col] = start_capital * (g[close_col] / c0)
        values.append(g[[out_col]])

    series_df = pd.concat(values, axis=0)
    return series_df[out_col].reindex(df.index)


def build_equal_weight_window_series(df, start_capital):
    cache_path = find_stock_cache()
    if not cache_path:
        print("[WARN] Could not find finrl_dow30_cache.csv, skipping equal-weight baseline.")
        return pd.Series(index=df.index, dtype=float)

    raw = pd.read_csv(cache_path, usecols=["date", "tic", "close"])
    raw["date"] = pd.to_datetime(raw["date"])
    panel = raw.pivot_table(index="date", columns="tic", values="close", aggfunc="last").sort_index()

    values = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby("window_seq", sort=True):
        g = group.sort_values("date")
        dates = g["date"].tolist()
        if len(dates) == 0:
            continue
        window_prices = panel.reindex(dates).ffill().bfill()
        start_prices = window_prices.iloc[0]
        valid = start_prices.dropna().index
        if len(valid) == 0:
            continue
        rel = window_prices[valid].div(start_prices[valid], axis=1)
        window_value = start_capital * rel.mean(axis=1, skipna=True)
        values.loc[g.index] = window_value.values
    return values


def format_metrics_row(name, series):
    ret, sharpe, mdd = calculate_metrics(series)
    return {
        "strategy": name,
        "cumulative_return": ret,
        "annualized_sharpe": sharpe,
        "max_drawdown": mdd,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DDPG rolling-window results with fair baselines.")
    parser.add_argument("--result-file", type=str, default="ddpg_rolling_results_v2.csv")
    parser.add_argument("--window-len", type=int, default=20)
    parser.add_argument("--benchmark-cache", type=str, default="benchmark_dji_cache.csv")
    parser.add_argument("--out-curve-csv", type=str, default="ddpg_evaluation_curves.csv")
    parser.add_argument("--out-metrics-csv", type=str, default="ddpg_evaluation_metrics.csv")
    parser.add_argument("--out-plot", type=str, default="ddpg_backtest_performance_fair.png")
    args = parser.parse_args()

    try:
        df = load_result_df(args.result_file, args.window_len)

        start_date = df["date"].iloc[0].strftime("%Y-%m-%d")
        benchmark_end = (df["date"].iloc[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        initial_capital = float(df["account_value"].iloc[0])

        print(f"[*] Loaded rolling results. Period: {start_date} -> {df['date'].iloc[-1].date()}")
        print("[*] Loading benchmark (^DJI)...")

        benchmark = fetch_benchmark_with_retry(
            start_date=start_date,
            end_date=benchmark_end,
            cache_file=args.benchmark_cache,
        )
        if benchmark.empty:
            raise ValueError("Benchmark download returned no rows for ^DJI.")

        df = pd.merge(df, benchmark[["Date", "Close"]], left_on="date", right_on="Date", how="left")
        df["Close"] = df["Close"].ffill().bfill()
        if df["Close"].isna().all():
            raise ValueError("Benchmark series is empty after merge.")

        # Strategy curves
        df = add_compounded_curve(df, value_col="account_value", out_col="strategy_compounded", start_capital=initial_capital)

        # Benchmark curves: continuous and protocol-consistent rolling-compounded
        df["benchmark_continuous"] = initial_capital * (df["Close"] / float(df["Close"].iloc[0]))
        df["benchmark_window_value"] = build_window_benchmark_series(
            df, close_col="Close", out_col="benchmark_window_value", start_capital=initial_capital
        )
        df = add_compounded_curve(
            df, value_col="benchmark_window_value", out_col="benchmark_compounded", start_capital=initial_capital
        )

        # Equal-weight stock baseline under same rolling protocol
        df["equal_weight_window_value"] = build_equal_weight_window_series(df, start_capital=initial_capital)
        if df["equal_weight_window_value"].notna().any():
            df = add_compounded_curve(
                df,
                value_col="equal_weight_window_value",
                out_col="equal_weight_compounded",
                start_capital=initial_capital,
            )
        else:
            df["equal_weight_compounded"] = np.nan

        metrics_rows = [
            format_metrics_row("DDPG (rolling compounded)", df["strategy_compounded"]),
            format_metrics_row("DJI (rolling compounded)", df["benchmark_compounded"]),
            format_metrics_row("DJI (continuous)", df["benchmark_continuous"]),
        ]
        if df["equal_weight_compounded"].notna().any():
            metrics_rows.append(format_metrics_row("EqualWeight (rolling compounded)", df["equal_weight_compounded"]))

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(args.out_metrics_csv, index=False)

        print("\n" + "=" * 78)
        print("DDPG Evaluation (Protocol-Consistent)")
        print("=" * 78)
        for _, row in metrics_df.iterrows():
            print(
                f"{row['strategy']:<35} | "
                f"Return {row['cumulative_return'] * 100:>8.2f}% | "
                f"Sharpe {row['annualized_sharpe']:>7.3f} | "
                f"MaxDD {row['max_drawdown'] * 100:>8.2f}%"
            )
        print("=" * 78)

        export_cols = [
            "date",
            "window_seq",
            "account_value",
            "strategy_compounded",
            "Close",
            "benchmark_continuous",
            "benchmark_window_value",
            "benchmark_compounded",
            "equal_weight_window_value",
            "equal_weight_compounded",
        ]
        existing_cols = [c for c in export_cols if c in df.columns]
        df[existing_cols].to_csv(args.out_curve_csv, index=False)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

        ax1.plot(df["date"], df["strategy_compounded"], label="DDPG (rolling compounded)", linewidth=2.2)
        ax1.plot(df["date"], df["benchmark_compounded"], label="DJI (rolling compounded)", linewidth=1.7, linestyle="--")
        if df["equal_weight_compounded"].notna().any():
            ax1.plot(
                df["date"],
                df["equal_weight_compounded"],
                label="EqualWeight (rolling compounded)",
                linewidth=1.5,
                linestyle=":",
            )
        ax1.set_title("Fair Comparison: Same Rolling-Compounded Walk-Forward Protocol")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="upper left")

        ax2.plot(df["date"], df["account_value"], label="DDPG raw stitched account_value", linewidth=1.8)
        ax2.plot(df["date"], df["benchmark_continuous"], label="DJI continuous", linewidth=1.5, linestyle="--")
        ax2.set_title("Reference Only: Raw Stitched Strategy vs Continuous Benchmark (Not apples-to-apples)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Portfolio Value ($)")
        ax2.grid(alpha=0.3)
        ax2.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(args.out_plot, dpi=300)
        print(f"\n[+] Saved fair evaluation plot: '{args.out_plot}'")
        print(f"[+] Saved curve data: '{args.out_curve_csv}'")
        print(f"[+] Saved metrics: '{args.out_metrics_csv}'")
        plt.show()

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")


if __name__ == "__main__":
    main()
