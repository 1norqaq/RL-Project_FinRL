# ddpg_train.py
import argparse
import os
import glob
import time
import random
import pandas as pd
import numpy as np
import torch
import yfinance as yf
try:
    import gymnasium as gym
except ImportError:
    import gym

# -----------------------------
# Compatibility patches
# -----------------------------
original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return original_torch_load(*args, **kwargs)


torch.load = patched_torch_load

original_download = yf.download


def patched_download(*args, **kwargs):
    kwargs.pop("proxy", None)
    kwargs.setdefault("auto_adjust", False)
    kwargs.setdefault("progress", False)
    return original_download(*args, **kwargs)


yf.download = patched_download

# -----------------------------
# Imports from FinRL + ElegantRL
# -----------------------------
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from elegantrl.train.run import train_agent

try:
    from elegantrl.train.config import Config as Arguments
except ImportError:
    from elegantrl.train.config import Arguments

# Agent import with robust fallback (in case of naming differences across ElegantRL versions)
try:
    from elegantrl.agents import AgentDDPG
except ImportError:
    # If your ElegantRL version uses a different name, fail loudly with a clear message
    raise ImportError(
        "Could not import AgentDDPG from elegantrl.agents. "
        "Please check your installed ElegantRL version and agent class names."
    )


# -----------------------------
# Env Wrapper
# -----------------------------
class ElegantFinRLWrapper(gym.Wrapper):
    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        num_stock_shares,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        env_name,
        state_dim,
        action_dim,
        if_discrete,
        target_return,
        penalty_coef=0.05,
        **kwargs,
    ):
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
            tech_indicator_list=tech_indicator_list,
        )
        super().__init__(env)

        self.env_name = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.if_discrete = if_discrete
        self.target_return = target_return
        self.max_step = len(df.index.unique())

        self.prev_action = np.zeros(action_dim, dtype=np.float32)

        self.penalty_coef = float(penalty_coef)

    def reset(self, *, seed=None, options=None):
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        try:
            res = self.env.reset(seed=seed, options=options)
        except TypeError:
            res = self.env.reset()
        if isinstance(res, tuple) and len(res) == 2:
            obs, _info = res
        else:
            obs = res
        # ElegantRL off-policy trainers expect Gym-style reset() -> obs (not tuple).
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        res = self.env.step(action)

        # Action-change penalty
        action_diff = np.sum(np.abs(action - self.prev_action))
        penalty = action_diff * self.penalty_coef
        self.prev_action = action.copy()

        if len(res) == 5:
            obs, reward, term, trunc, info = res
            done = bool(term or trunc)
        else:
            obs, reward, done, info = res
            done = bool(done)
        modified_reward = float(reward) - float(penalty)
        # ElegantRL off-policy trainers expect Gym-style step() -> (obs, reward, done, info).
        return np.array(obs, dtype=np.float32), modified_reward, done, info


# -----------------------------
# Data loader
# -----------------------------
DEFAULT_INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30", "close_60"]


def _load_if_valid_cache(csv_path):
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    required_columns = {"date", "tic", "open", "high", "low", "close", "volume", *DEFAULT_INDICATORS}
    if not required_columns.issubset(df.columns):
        return None
    if df.empty:
        return None
    return df.drop_duplicates(subset=["date", "tic"]).sort_values(["date", "tic"]).reset_index(drop=True)


def _discover_existing_cache(preferred_cache_file):
    candidates = [
        preferred_cache_file,
        "finrl_dow30_cache.csv",
        os.path.join("data", "finrl_dow30_cache.csv"),
        os.path.join("..", "data", "finrl_dow30_cache.csv"),
    ]

    for candidate in candidates:
        df = _load_if_valid_cache(candidate)
        if df is not None:
            print(f"[INFO] Loading validated cache '{candidate}'...")
            return df

    search_roots = [os.getcwd(), os.path.abspath(os.path.join(os.getcwd(), ".."))]
    seen = set()
    for root in search_roots:
        pattern = os.path.join(root, "**", "finrl_dow30_cache.csv")
        for match in glob.glob(pattern, recursive=True):
            norm = os.path.normpath(match)
            if norm in seen:
                continue
            seen.add(norm)
            df = _load_if_valid_cache(norm)
            if df is not None:
                print(f"[INFO] Loading validated cache '{norm}'...")
                return df
    return None


def _compute_indicators_with_pandas(df):
    frames = []
    for _, ticker_df in df.groupby("tic", sort=False):
        ticker_df = ticker_df.sort_values("date").copy()
        close = ticker_df["close"]
        high = ticker_df["high"]
        low = ticker_df["low"]

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        ticker_df["macd"] = macd_line - macd_signal

        mid = close.rolling(window=20, min_periods=20).mean()
        std = close.rolling(window=20, min_periods=20).std(ddof=0)
        ticker_df["boll_ub"] = mid + 2.0 * std
        ticker_df["boll_lb"] = mid - 2.0 * std

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=30, min_periods=30).mean()
        avg_loss = loss.rolling(window=30, min_periods=30).mean().replace(0, np.nan)
        rs = avg_gain / avg_loss
        ticker_df["rsi_30"] = 100 - (100 / (1 + rs))

        tp = (high + low + close) / 3
        tp_ma = tp.rolling(window=30, min_periods=30).mean()
        mad = tp.rolling(window=30, min_periods=30).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        ticker_df["cci_30"] = (tp - tp_ma) / (0.015 * mad.replace(0, np.nan))

        up_move = high.diff()
        down_move = low.shift(1) - low
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=ticker_df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=ticker_df.index)

        tr = pd.concat(
            [
                (high - low),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=30, min_periods=30).mean().replace(0, np.nan)
        plus_di = 100 * plus_dm.rolling(window=30, min_periods=30).mean() / atr
        minus_di = 100 * minus_dm.rolling(window=30, min_periods=30).mean() / atr
        ticker_df["dx_30"] = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

        ticker_df["close_30"] = close.rolling(window=30, min_periods=30).mean()
        ticker_df["close_60"] = close.rolling(window=60, min_periods=60).mean()

        frames.append(ticker_df)

    result = pd.concat(frames, ignore_index=True)
    return result


def _normalize_single_ticker_frame(frame, tic):
    if frame is None or frame.empty:
        return None

    if isinstance(frame.columns, pd.MultiIndex):
        try:
            frame = frame.xs(tic, axis=1, level=0, drop_level=True)
        except Exception:
            try:
                frame = frame.xs(tic, axis=1, level=-1, drop_level=True)
            except Exception:
                return None

    frame = frame.copy().reset_index()

    if "Date" in frame.columns:
        frame = frame.rename(columns={"Date": "date"})
    elif "index" in frame.columns:
        frame = frame.rename(columns={"index": "date"})
    elif "Datetime" in frame.columns:
        frame = frame.rename(columns={"Datetime": "date"})
    else:
        return None

    rename_map = {}
    for col in frame.columns:
        col_str = str(col).strip()
        if col_str == "Open":
            rename_map[col] = "open"
        elif col_str == "High":
            rename_map[col] = "high"
        elif col_str == "Low":
            rename_map[col] = "low"
        elif col_str == "Close":
            rename_map[col] = "close"
        elif col_str == "Adj Close":
            rename_map[col] = "adj_close"
        elif col_str == "Volume":
            rename_map[col] = "volume"
    frame = frame.rename(columns=rename_map)

    if "close" not in frame.columns and "adj_close" in frame.columns:
        frame["close"] = frame["adj_close"]

    needed = {"date", "open", "high", "low", "close", "volume"}
    if not needed.issubset(frame.columns):
        return None

    frame["tic"] = tic
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame[["date", "open", "high", "low", "close", "volume", "tic"]]


def _download_with_yfinance(ticker_list, start_date, end_date, single_retries=3):
    frames_by_ticker = {}
    missing = list(ticker_list)

    try:
        batch = yf.download(
            tickers=ticker_list,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False,
        )
    except Exception as exc:
        batch = None
        print(f"[WARN] Batch yfinance download failed ({exc}).")

    if batch is not None and not batch.empty:
        for tic in ticker_list:
            try:
                candidate = _normalize_single_ticker_frame(batch, tic)
            except Exception:
                candidate = None
            if candidate is not None and not candidate.empty:
                frames_by_ticker[tic] = candidate
        missing = [tic for tic in ticker_list if tic not in frames_by_ticker]

    if missing:
        print(f"[WARN] Batch download missed {len(missing)} tickers. Retrying individually with backoff...")

    for tic in list(missing):
        got = None
        for attempt in range(1, single_retries + 1):
            try:
                single = yf.download(
                    tic,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                got = _normalize_single_ticker_frame(single, tic)
            except Exception:
                got = None

            if got is not None and not got.empty:
                frames_by_ticker[tic] = got
                break

            if attempt < single_retries:
                wait_s = min(45, 6 * attempt + random.randint(0, 4))
                print(f"[INFO] Retry {attempt + 1}/{single_retries} for {tic} after {wait_s}s...")
                time.sleep(wait_s)

    if not frames_by_ticker:
        raise RuntimeError(
            "yfinance download failed for all tickers (likely Yahoo rate limit). "
            "Please wait and retry, or place an existing finrl_dow30_cache.csv in ./data."
        )

    data = pd.concat(frames_by_ticker.values(), ignore_index=True).sort_values(["date", "tic"]).reset_index(drop=True)
    failed = [tic for tic in ticker_list if tic not in frames_by_ticker]
    if failed:
        preview = ", ".join(failed[:10]) + (", ..." if len(failed) > 10 else "")
        print(f"[WARN] Proceeding with {len(frames_by_ticker)} tickers; missing {len(failed)}: {preview}")
    else:
        print(f"[INFO] Downloaded all {len(frames_by_ticker)} tickers.")
    print(f"[INFO] yfinance rows: {len(data)}")
    return data


def prepare_data(
    cache_file="./data/finrl_dow30_cache.csv",
    TRAIN_START_DATE="2010-01-01",
    TEST_END_DATE="2023-12-30",
):
    cache_dir = os.path.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    cached_df = _discover_existing_cache(cache_file)
    if cached_df is not None:
        return cached_df

    print(f"[INFO] Downloading Dow 30 data from {TRAIN_START_DATE} to {TEST_END_DATE}...")
    df = None
    try:
        df = YahooDownloader(
            start_date=TRAIN_START_DATE,
            end_date=TEST_END_DATE,
            ticker_list=DOW_30_TICKER,
        ).fetch_data()
    except Exception as exc:
        print(f"[WARN] FinRL YahooDownloader failed ({exc}). Falling back to direct yfinance downloader...")
        df = None

    base_cols = {"date", "tic", "open", "high", "low", "close", "volume"}
    if df is None or df.empty or not base_cols.issubset(df.columns):
        print("[WARN] FinRL YahooDownloader returned unusable data. Falling back to direct yfinance downloader...")
        df = _download_with_yfinance(DOW_30_TICKER, TRAIN_START_DATE, TEST_END_DATE)

    technical_indicators = [
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "cci_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
    ]
    try:
        from finrl.meta.preprocessor.preprocessors import FeatureEngineer

        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=technical_indicators,
            use_vix=False,
            use_turbulence=False,
            user_defined_feature=False,
        )
        df = fe.preprocess_data(df)

        if "close_30_sma" in df.columns:
            df = df.rename(columns={"close_30_sma": "close_30"})
        if "close_60_sma" in df.columns:
            df = df.rename(columns={"close_60_sma": "close_60"})
    except Exception as exc:
        print(f"[WARN] FeatureEngineer unavailable ({exc}). Falling back to pandas indicators...")
        df = _compute_indicators_with_pandas(df)

    required_columns = {"date", "tic", "open", "high", "low", "close", "volume", *DEFAULT_INDICATORS}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            f"Generated cache is missing required columns: {missing_columns}. "
            "Please verify FinRL FeatureEngineer output."
        )

    df = df.dropna(subset=DEFAULT_INDICATORS)
    df = df.drop_duplicates(subset=["date", "tic"]).sort_values(["date", "tic"]).reset_index(drop=True)
    ticker_count = df["tic"].nunique()
    if ticker_count < 20:
        raise RuntimeError(
            f"Only {ticker_count} tickers available after preprocessing. "
            "This is too low for a stable Dow-30 training run; likely caused by Yahoo rate limits."
        )
    df.to_csv(cache_file, index=False)
    print(f"[INFO] Cache saved to '{cache_file}' ({len(df)} rows).")
    return df


# -----------------------------
# ElegantRL args for DDPG
# -----------------------------
def setup_ddpg_args(
    env_args,
    cwd_path,
    *,
    net_dims=(128, 64),
    learning_rate=1e-4,
    batch_size=128,
    target_step=2000,
    break_step=40000,
    if_remove=True,
):
    args = Arguments(agent_class=AgentDDPG, env_class=ElegantFinRLWrapper)
    args.env_args = env_args
    args.env_name = env_args["env_name"]

    args.net_dims = tuple(net_dims)
    args.state_dim = env_args["state_dim"]
    args.action_dim = env_args["action_dim"]
    args.if_discrete = env_args["if_discrete"]

    args.learning_rate = float(learning_rate)
    args.batch_size = int(batch_size)

    args.target_step = int(target_step)
    args.break_step = int(break_step)

    args.worker_num = 1
    args.eval_proc_num = 0
    args.if_use_multi_processing = False
    args.eval_gap = 500
    args.save_gap = 500

    args.if_save = True
    args.if_overwrite_save = True

    args.cwd = cwd_path
    args.if_remove = bool(if_remove)
    return args


# -----------------------------
# Inference on a test window
# -----------------------------
def real_test_inference(test_df, stock_dim, indicators, args, *, initial_amount=1000000, penalty_coef=0.05):
    params = {
        "df": test_df,
        "stock_dim": stock_dim,
        "hmax": 100,
        "initial_amount": initial_amount,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4,
        "state_space": 1 + stock_dim * (len(indicators) + 2),
        "action_space": stock_dim,
        "tech_indicator_list": indicators,
        "env_name": "test_inference",
        "state_dim": 1 + stock_dim * (len(indicators) + 2),
        "action_dim": stock_dim,
        "if_discrete": False,
        "target_return": 10.0,
        "penalty_coef": penalty_coef,
    }
    env = ElegantFinRLWrapper(**params)

    checkpoint_files = [f for f in os.listdir(args.cwd) if f.lower().endswith(".pth")]
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint files were found in '{args.cwd}' after training. "
            "Training likely ended before any model was saved."
        )

    agent = AgentDDPG(args.net_dims, args.state_dim, args.action_dim)
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
            state, reward, done, _ = step_res

    return env.env.save_asset_memory()


def parse_net_dims(text):
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError("net_dims cannot be empty.")
    dims = tuple(int(p) for p in parts)
    if any(d <= 0 for d in dims):
        raise ValueError(f"net_dims must be positive integers, got: {dims}")
    return dims


def parse_args():
    parser = argparse.ArgumentParser(description="DDPG rolling-window trainer (FinRL + ElegantRL)")
    parser.add_argument("--cache-file", type=str, default="./data/finrl_dow30_cache.csv")
    parser.add_argument("--train-start", type=str, default="2010-01-01")
    parser.add_argument("--test-end", type=str, default="2023-12-30")

    parser.add_argument("--train-window", type=int, default=252)
    parser.add_argument("--val-window", type=int, default=20)
    parser.add_argument("--test-window", type=int, default=20)
    parser.add_argument("--max-windows", type=int, default=0, help="0 means use all rolling windows.")

    parser.add_argument("--initial-amount", type=float, default=1_000_000)
    parser.add_argument("--penalty-coef", type=float, default=0.05)

    parser.add_argument("--net-dims", type=str, default="128,64", help="Comma-separated, e.g. 256,256")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-step", type=int, default=2000)
    parser.add_argument("--break-step", type=int, default=40000)

    parser.add_argument("--checkpoint-root", type=str, default="./checkpoints")
    parser.add_argument("--warm-start", action="store_true", help="Continue model weights across windows.")
    parser.add_argument(
        "--reset-warm-start",
        action="store_true",
        help="If set with --warm-start, clear shared warm-start checkpoint at first window.",
    )
    parser.add_argument("--output-csv", type=str, default="ddpg_rolling_results_v2.csv")
    return parser.parse_args()


# -----------------------------
# Main: rolling window training + inference
# -----------------------------
if __name__ == "__main__":
    cli = parse_args()

    net_dims = parse_net_dims(cli.net_dims)

    df_raw = prepare_data(
        cache_file=cli.cache_file,
        TRAIN_START_DATE=cli.train_start,
        TEST_END_DATE=cli.test_end,
    )
    if df_raw is None or df_raw.empty:
        raise FileNotFoundError(
            "Could not load cache file. "
            "Expected either './finrl_dow30_cache.csv' or './data/finrl_dow30_cache.csv'."
        )

    # Keep only dates where all tickers have data
    df_pivot = df_raw.pivot(index="date", columns="tic", values="close")
    valid_dates = df_pivot.dropna().index
    df = df_raw[df_raw["date"].isin(valid_dates)].copy()

    unique_dates = sorted(df["date"].unique())
    stock_dimension = len(df["tic"].unique())
    ALL_INDICATORS = DEFAULT_INDICATORS

    TRAIN_WINDOW = int(cli.train_window)
    VAL_WINDOW = int(cli.val_window)
    TEST_WINDOW = int(cli.test_window)

    min_required_days = TRAIN_WINDOW + VAL_WINDOW + TEST_WINDOW
    if len(unique_dates) < min_required_days:
        raise RuntimeError(
            f"Insufficient aligned trading days after preprocessing: {len(unique_dates)} found, "
            f"but at least {min_required_days} are required."
        )

    all_test_results = []
    shared_warmstart_dir = os.path.join(cli.checkpoint_root, "ddpg_warmstart_shared")

    for window_seq, i in enumerate(range(TRAIN_WINDOW + VAL_WINDOW, len(unique_dates) - TEST_WINDOW + 1, TEST_WINDOW)):
        if int(cli.max_windows) > 0 and window_seq >= int(cli.max_windows):
            break

        train_dates = unique_dates[i - TRAIN_WINDOW - VAL_WINDOW : i - VAL_WINDOW]
        test_dates = unique_dates[i : i + TEST_WINDOW]

        train_df = df[df["date"].isin(train_dates)].sort_values(["date", "tic"]).reset_index(drop=True)
        test_df = df[df["date"].isin(test_dates)].sort_values(["date", "tic"]).reset_index(drop=True)

        # Map day index separately for each window
        for d_df in [train_df, test_df]:
            date_map = {date: idx for idx, date in enumerate(sorted(d_df["date"].unique()))}
            d_df["day"] = d_df["date"].map(date_map)
            d_df.set_index("day", inplace=True, drop=False)

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
            "target_return": 10.0,
            "penalty_coef": float(cli.penalty_coef),
        }

        print(f"\n>>> Rolling Window: {train_dates[0]} to {test_dates[-1]}")

        try:
            if cli.warm_start:
                cwd_path = shared_warmstart_dir
                remove_flag = bool(cli.reset_warm_start and window_seq == 0)
            else:
                cwd_path = os.path.join(cli.checkpoint_root, f"ddpg_window_{i}")
                remove_flag = True
            os.makedirs(cwd_path, exist_ok=True)

            args = setup_ddpg_args(
                env_params,
                cwd_path,
                net_dims=net_dims,
                learning_rate=cli.learning_rate,
                batch_size=cli.batch_size,
                target_step=cli.target_step,
                break_step=cli.break_step,
                if_remove=remove_flag,
            )

            print("[-->] Starting DDPG Agent Training...")
            train_agent(args)

            res = real_test_inference(
                test_df,
                stock_dimension,
                ALL_INDICATORS,
                args,
                initial_amount=cli.initial_amount,
                penalty_coef=cli.penalty_coef,
            )
            res["window_seq"] = window_seq
            res["window_id"] = i
            res["window_train_start"] = train_dates[0]
            res["window_train_end"] = train_dates[-1]
            res["window_test_start"] = test_dates[0]
            res["window_test_end"] = test_dates[-1]
            all_test_results.append(res)

        except Exception as e:
            print(f"[ERROR] Window {i} failed: {e}")
            import traceback

            traceback.print_exc()
            continue

    if all_test_results:
        pd.concat(all_test_results).reset_index(drop=True).to_csv(cli.output_csv, index=False)
        print(f"\n[SUCCESS] Backtest complete! File saved as {cli.output_csv}")
    else:
        raise RuntimeError("All rolling windows failed; no backtest results were produced.")
