import argparse
import collections
import glob
import json
import os
import random
import re
import shutil
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf

DEPENDENCY_IMPORT_ERRORS: List[str] = []
AgentD3QN = None
AgentDQN = None
AgentDoubleDQN = None
train_agent = None
Arguments = None

try:
    from elegantrl.agents import AgentD3QN as _AgentD3QN

    AgentD3QN = _AgentD3QN
except Exception as exc:
    DEPENDENCY_IMPORT_ERRORS.append(f"ElegantRL AgentD3QN import failed: {exc}")

try:
    from elegantrl.agents import AgentDQN as _AgentDQN

    AgentDQN = _AgentDQN
except Exception:
    AgentDQN = None

try:
    from elegantrl.agents import AgentDoubleDQN as _AgentDoubleDQN

    AgentDoubleDQN = _AgentDoubleDQN
except Exception:
    AgentDoubleDQN = None

try:
    from elegantrl.train.run import train_agent as _train_agent

    train_agent = _train_agent
except Exception as exc:
    DEPENDENCY_IMPORT_ERRORS.append(f"ElegantRL train import failed: {exc}")

try:
    from elegantrl.train.config import Config as _Arguments

    Arguments = _Arguments
except Exception:
    try:
        from elegantrl.train.config import Arguments as _Arguments

        Arguments = _Arguments
    except Exception as exc:
        DEPENDENCY_IMPORT_ERRORS.append(f"ElegantRL config import failed: {exc}")


def parse_net_dims(text: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    dims = tuple(int(p) for p in parts)
    if not dims or any(d <= 0 for d in dims):
        raise ValueError(f"Invalid --net-dims: {text}")
    return dims


def parse_position_levels(text: str) -> Tuple[float, ...]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    levels = tuple(float(p) for p in parts)
    if len(levels) < 2:
        raise ValueError("--position-levels must contain at least 2 levels, e.g. 0,0.5,1.0")
    if any((x < 0.0 or x > 1.0) for x in levels):
        raise ValueError("--position-levels values must be in [0, 1].")
    if any(levels[i] > levels[i + 1] for i in range(len(levels) - 1)):
        raise ValueError("--position-levels must be sorted ascending.")
    return levels


def get_agent_registry() -> Dict[str, Any]:
    registry = {}
    if AgentD3QN is not None:
        registry["d3qn"] = AgentD3QN
    if AgentDQN is not None:
        registry["dqn"] = AgentDQN
    if AgentDoubleDQN is not None:
        registry["double_dqn"] = AgentDoubleDQN
    return registry


def ensure_d3qn_dependencies() -> None:
    registry = get_agent_registry()
    missing = []
    if not registry:
        missing.append("elegantrl-agents")
    if train_agent is None:
        missing.append("elegantrl-train-run")
    if Arguments is None:
        missing.append("elegantrl-config")
    if missing:
        detail = "; ".join(DEPENDENCY_IMPORT_ERRORS) if DEPENDENCY_IMPORT_ERRORS else "missing imports"
        raise ImportError(
            "d3qn_crypto.py missing runtime dependencies: "
            f"{', '.join(missing)}. Details: {detail}. "
            "Install dependencies from requirements.txt and retry."
        )


def resolve_agent_families(agent_family: str) -> List[str]:
    registry = get_agent_registry()
    family = str(agent_family).strip().lower()
    if family == "ensemble":
        ordered = [name for name in ["dqn", "double_dqn", "d3qn"] if name in registry]
        if len(ordered) < 2:
            raise RuntimeError(
                "Ensemble requires at least two available agents among dqn/double_dqn/d3qn. "
                "Your ElegantRL install does not expose enough agent classes."
            )
        return ordered
    if family not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise RuntimeError(f"Requested --agent-family={family} is unavailable. Available: {available}")
    return [family]


def agent_label(agent_family: str) -> str:
    mapping = {"dqn": "DQN", "double_dqn": "DoubleDQN", "d3qn": "D3QN", "ensemble": "Ensemble"}
    return mapping.get(str(agent_family).lower(), str(agent_family))


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_serializable(obj: Any):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def save_json(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2, sort_keys=True)


def export_csv_artifacts(run_dir: str, artifact_root: str) -> Tuple[str, int]:
    run_base = os.path.abspath(run_dir)
    run_name = os.path.basename(run_base.rstrip("\\/"))
    dst_root = os.path.abspath(os.path.join(artifact_root, run_name))
    dst_root_norm = os.path.normcase(dst_root)
    copied = 0

    for src in glob.glob(os.path.join(run_base, "**", "*.csv"), recursive=True):
        src_abs = os.path.abspath(src)
        src_norm = os.path.normcase(src_abs)
        if src_norm.startswith(dst_root_norm + os.sep) or src_norm == dst_root_norm:
            continue

        rel = os.path.relpath(src_abs, run_base)
        dst = os.path.join(dst_root, rel)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy2(src_abs, dst)
        copied += 1

    return dst_root, copied


def resolve_input_path(path: str, fallback_names: Optional[Sequence[str]] = None) -> str:
    raw = str(path).strip()
    base = os.path.basename(raw) if raw else ""
    candidates = []

    if raw:
        candidates.append(raw)
        candidates.append(os.path.abspath(raw))
        expanded = os.path.expandvars(os.path.expanduser(raw))
        candidates.append(expanded)
        candidates.append(os.path.abspath(expanded))

    search_roots = [os.getcwd(), os.path.abspath("."), os.path.abspath("./data")]
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    if os.path.isdir(downloads):
        search_roots.append(downloads)

    if base:
        for root in search_roots:
            candidates.append(os.path.join(root, base))
    if fallback_names:
        for name in fallback_names:
            for root in search_roots:
                candidates.append(os.path.join(root, str(name)))

    seen = set()
    for cand in candidates:
        if not cand:
            continue
        ap = os.path.abspath(cand)
        if ap in seen:
            continue
        seen.add(ap)
        if os.path.exists(ap):
            return ap

    msg = f"File not found: {path}."
    if base:
        msg += f" Checked current/data/downloads for {base}."
    raise FileNotFoundError(msg)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def normalize_yf_frame(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    out = df.reset_index().copy()
    if "Date" in out.columns:
        out = out.rename(columns={"Date": "date"})
    elif "Datetime" in out.columns:
        out = out.rename(columns={"Datetime": "date"})
    elif "index" in out.columns:
        out = out.rename(columns={"index": "date"})
    else:
        raise ValueError("Downloaded data has no date column.")

    if "Adj Close" in out.columns and "Close" not in out.columns:
        out = out.rename(columns={"Adj Close": "Close"})

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in out.columns:
            raise ValueError(f"Missing required price column: {col}")

    out = out.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    out["date"] = pd.to_datetime(out["date"])
    out = out[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
    return out


def _download_with_retry(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    max_retries: int = 5,
    backoff_base: int = 5,
) -> pd.DataFrame:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if raw is not None and not raw.empty:
                return raw
        except Exception as exc:
            last_err = exc

        if attempt < max_retries:
            wait_s = min(120, int(backoff_base) * attempt + random.randint(0, 3))
            print(f"[WARN] Yahoo download failed attempt {attempt}/{max_retries}, retrying in {wait_s}s...")
            time.sleep(wait_s)

    if last_err is not None:
        raise RuntimeError(f"Yahoo download failed after {max_retries} attempts: {last_err}")
    raise RuntimeError(f"Yahoo download failed after {max_retries} attempts with empty data.")


def _normalize_alt_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {str(c).strip().lower(): c for c in df.columns}
    required_keys = ["date", "open", "high", "low", "close"]
    if not all(k in colmap for k in required_keys):
        raise ValueError("Alternative source frame missing required OHLC columns.")

    vol_key = None
    for k in ["volume", "volume usd", "volume btc", "volumefrom", "volumeto"]:
        if k in colmap:
            vol_key = k
            break
    if vol_key is None:
        vol_key = "close"

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[colmap["date"]], errors="coerce"),
            "open": pd.to_numeric(df[colmap["open"]], errors="coerce"),
            "high": pd.to_numeric(df[colmap["high"]], errors="coerce"),
            "low": pd.to_numeric(df[colmap["low"]], errors="coerce"),
            "close": pd.to_numeric(df[colmap["close"]], errors="coerce"),
            "volume": pd.to_numeric(df[colmap[vol_key]], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return out


def _download_from_cryptodatadownload(ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    if str(interval).lower() not in {"1d", "1wk", "1mo"}:
        return pd.DataFrame()

    symbol = str(ticker).upper().replace("/", "-")
    sources = {
        "BTC-USD": ["Coinbase_BTCUSD_d.csv", "Binance_BTCUSDT_d.csv"],
        "ETH-USD": ["Coinbase_ETHUSD_d.csv", "Binance_ETHUSDT_d.csv"],
    }
    files = sources.get(symbol, [])
    if not files:
        return pd.DataFrame()

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    for fname in files:
        url = f"https://www.cryptodatadownload.com/cdd/{fname}"
        try:
            raw = pd.read_csv(url, skiprows=1)
            df = _normalize_alt_ohlcv(raw)
            if df.empty:
                continue
            df = df[(df["date"] >= start_ts) & (df["date"] < end_ts)].reset_index(drop=True)
            if df.empty:
                continue

            rule = None
            if interval == "1wk":
                rule = "W"
            elif interval == "1mo":
                rule = "MS"

            if rule is not None:
                df = (
                    df.set_index("date")
                    .resample(rule)
                    .agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                        }
                    )
                    .dropna(subset=["open", "high", "low", "close"])
                    .reset_index()
                )
            if not df.empty:
                print(f"[WARN] Using fallback source cryptodatadownload: {url}")
                return df
        except Exception:
            continue
    return pd.DataFrame()


def _find_existing_crypto_cache(cache_file: str, ticker: str, interval: str) -> Optional[str]:
    candidates = []
    if os.path.exists(cache_file):
        candidates.append(cache_file)

    safe_ticker = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_").lower()
    safe_interval = str(interval).lower()
    roots = [os.path.abspath("./data"), os.getcwd()]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for path in glob.glob(os.path.join(root, "*.csv")):
            name = os.path.basename(path).lower()
            if safe_ticker in name or "btc" in name or "crypto" in name:
                if safe_interval in name or "1d" in name or "daily" in name:
                    candidates.append(path)

    seen = set()
    uniq = []
    for p in candidates:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(ap)
    for path in uniq:
        try:
            head = pd.read_csv(path, nrows=3)
        except Exception:
            continue
        cols = {str(c).strip().lower() for c in head.columns}
        if {"date", "close"}.issubset(cols) or {"date", "open", "high", "low", "close", "volume"}.issubset(cols):
            return path
    return None


def load_crypto_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    cache_file: str,
    force_download: bool,
    download_retries: int = 5,
    download_backoff_seconds: int = 5,
    fallback_source: str = "auto",
) -> pd.DataFrame:
    if (not force_download) and os.path.exists(cache_file):
        df = pd.read_csv(cache_file, parse_dates=["date"])
    else:
        df = pd.DataFrame()
        print(f"[INFO] Downloading {ticker} data ({start_date} -> {end_date}, interval={interval})...")
        try:
            raw = _download_with_retry(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                max_retries=int(download_retries),
                backoff_base=int(download_backoff_seconds),
            )
            df = normalize_yf_frame(raw)
        except Exception as exc:
            print(f"[WARN] Yahoo download unavailable: {exc}")

        if df.empty and str(fallback_source).lower() in {"auto", "cryptodatadownload"}:
            alt_df = _download_from_cryptodatadownload(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )
            if not alt_df.empty:
                df = alt_df

        if df.empty:
            local_cache = _find_existing_crypto_cache(
                cache_file=cache_file,
                ticker=ticker,
                interval=interval,
            )
            if local_cache is not None:
                print(f"[WARN] Using existing local cache fallback: {local_cache}")
                df = pd.read_csv(local_cache, parse_dates=["date"])

        if df.empty:
            raise RuntimeError(
                "No usable crypto data source is available (Yahoo rate-limited and no fallback cache/source found). "
                "Provide a local CSV via --cache-file or rerun later."
            )

        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        df.to_csv(cache_file, index=False)
        print(f"[INFO] Saved cache: {cache_file}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return df


def load_lob_rnn_data(
    csv_file: str,
    rnn_feature_file: str,
    rnn_pretrain_ratio: float = 0.6,
    stride: int = 1,
    max_rows: int = 0,
) -> Tuple[pd.DataFrame, List[str]]:
    csv_resolved = resolve_input_path(csv_file, fallback_names=["BTC_1sec.csv"])
    rnn_resolved = resolve_input_path(rnn_feature_file, fallback_names=["BTC_1sec_predict.npy"])
    if os.path.abspath(str(csv_resolved)) != os.path.abspath(str(csv_file)):
        print(f"[WARN] LOB csv path resolved to existing file: {csv_resolved}")
    if os.path.abspath(str(rnn_resolved)) != os.path.abspath(str(rnn_feature_file)):
        print(f"[WARN] RNN feature path resolved to existing file: {rnn_resolved}")

    usecols = ["system_time", "midpoint", "buys", "sells", "timestep"]
    raw = pd.read_csv(csv_resolved, usecols=lambda c: c in usecols)
    rnn = np.load(rnn_resolved, mmap_mode="r")
    if rnn.ndim != 2:
        raise ValueError(f"Expected 2D npy features, got shape={rnn.shape}")

    n = min(len(raw), int(rnn.shape[0]))
    if n < 2000:
        raise ValueError(f"LOB dataset too small after alignment: {n} rows")

    raw = raw.iloc[len(raw) - n :].reset_index(drop=True)
    rnn_arr = np.asarray(rnn[int(rnn.shape[0]) - n :], dtype=np.float32)

    pre_ratio = float(rnn_pretrain_ratio)
    if pre_ratio < 0.0 or pre_ratio >= 0.95:
        raise ValueError("--lob-rnn-pretrain-ratio must be in [0, 0.95).")
    pre_n = int(n * pre_ratio)
    if pre_n >= n - 1000:
        raise ValueError("RNN pretrain cut leaves too little data. Lower --lob-rnn-pretrain-ratio.")

    raw = raw.iloc[pre_n:].reset_index(drop=True)
    rnn_arr = rnn_arr[pre_n:]

    stride = max(1, int(stride))
    if stride > 1:
        raw = raw.iloc[::stride].reset_index(drop=True)
        rnn_arr = rnn_arr[::stride]

    if int(max_rows) > 0 and len(raw) > int(max_rows):
        raw = raw.iloc[-int(max_rows) :].reset_index(drop=True)
        rnn_arr = rnn_arr[-int(max_rows) :]

    date = pd.to_datetime(raw.get("system_time"), errors="coerce")
    if date.isna().all():
        date = pd.to_datetime(raw.get("timestep"), unit="s", errors="coerce")
    if date.isna().all():
        date = pd.date_range(start="2021-01-01", periods=len(raw), freq="s")

    close = pd.to_numeric(raw.get("midpoint"), errors="coerce").fillna(method="ffill").fillna(method="bfill")
    if close.isna().all():
        raise ValueError("LOB csv has invalid midpoint values.")
    open_ = close.shift(1).fillna(close)
    high = np.maximum(open_.to_numpy(dtype=float), close.to_numpy(dtype=float))
    low = np.minimum(open_.to_numpy(dtype=float), close.to_numpy(dtype=float))
    volume = (
        pd.to_numeric(raw.get("buys"), errors="coerce").fillna(0.0)
        + pd.to_numeric(raw.get("sells"), errors="coerce").fillna(0.0)
    )

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(date),
            "open": open_.to_numpy(dtype=float),
            "high": high.astype(float),
            "low": low.astype(float),
            "close": close.to_numpy(dtype=float),
            "volume": volume.to_numpy(dtype=float),
        }
    )

    feature_cols = []
    for i in range(rnn_arr.shape[1]):
        col = f"rnn_factor_{i}"
        out[col] = rnn_arr[:, i].astype(np.float32)
        feature_cols.append(col)
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out = out.dropna(subset=["close"]).sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    print(
        f"[INFO] Loaded LOB+RNN dataset rows={len(out)} (after RNN pretrain cut ratio={pre_ratio:.2f}, rows={pre_n}, stride={stride})."
    )
    return out, feature_cols


def add_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    out["ret_1"] = out["close"].pct_change()
    out["ret_5"] = out["close"].pct_change(5)
    out["vol_20"] = out["ret_1"].rolling(20).std()
    out["sma_10_ratio"] = out["close"] / out["close"].rolling(10).mean() - 1.0
    out["sma_30_ratio"] = out["close"] / out["close"].rolling(30).mean() - 1.0
    out["rsi_14"] = compute_rsi(out["close"], window=14) / 100.0
    out["vol_chg_1"] = out["volume"].pct_change()

    feature_cols = [
        "ret_1",
        "ret_5",
        "vol_20",
        "sma_10_ratio",
        "sma_30_ratio",
        "rsi_14",
        "vol_chg_1",
    ]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, feature_cols


def split_train_val_test(df: pd.DataFrame, val_size: int, test_size: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    min_train = 120
    if len(df) <= min_train + val_size + test_size:
        raise ValueError(
            f"Dataset too small ({len(df)} rows). Need > {min_train + val_size + test_size}. "
            "Use longer history or smaller val/test windows."
        )
    train_df = df.iloc[: -(val_size + test_size)].copy().reset_index(drop=True)
    val_df = df.iloc[-(val_size + test_size) : -test_size].copy().reset_index(drop=True)
    test_df = df.iloc[-test_size:].copy().reset_index(drop=True)
    return train_df, val_df, test_df


def fit_clipping_bounds(train_df: pd.DataFrame, feature_cols: Sequence[str], clip_q: float) -> Dict[str, Tuple[float, float]]:
    bounds = {}
    for col in feature_cols:
        lo = float(train_df[col].quantile(clip_q))
        hi = float(train_df[col].quantile(1.0 - clip_q))
        if not np.isfinite(lo):
            lo = -1.0
        if not np.isfinite(hi):
            hi = 1.0
        if hi < lo:
            lo, hi = hi, lo
        bounds[col] = (lo, hi)
    return bounds


def apply_clipping(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, (lo, hi) in bounds.items():
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def fit_standardizer(train_df: pd.DataFrame, feature_cols: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    stats = {}
    for col in feature_cols:
        mu = float(train_df[col].mean())
        sigma = float(train_df[col].std())
        if (not np.isfinite(sigma)) or sigma < 1e-8:
            sigma = 1.0
        stats[col] = (mu, sigma)
    return stats


def apply_standardizer(df: pd.DataFrame, stats: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, (mu, sigma) in stats.items():
        out[col] = (out[col] - mu) / sigma
    return out


def parse_interval_to_periods(interval: str, fallback: float = 365.0) -> float:
    text = str(interval).strip().lower()
    if text.endswith("mo"):
        num = int(text[:-2]) if text[:-2].isdigit() else 1
        return 12.0 / max(1, num)
    if text.endswith("d"):
        num = int(text[:-1]) if text[:-1].isdigit() else 1
        return 365.0 / max(1, num)
    if text.endswith("h"):
        num = int(text[:-1]) if text[:-1].isdigit() else 1
        return (24.0 * 365.0) / max(1, num)
    if text.endswith("m"):
        num = int(text[:-1]) if text[:-1].isdigit() else 1
        return (24.0 * 60.0 * 365.0) / max(1, num)
    if text.endswith("wk"):
        num = int(text[:-2]) if text[:-2].isdigit() else 1
        return 52.0 / max(1, num)
    return fallback


def calculate_metrics(series: pd.Series, periods_per_year: float) -> Tuple[float, float, float]:
    values = pd.Series(series).dropna()
    if len(values) < 2:
        return 0.0, 0.0, 0.0
    period_ret = values.pct_change().dropna()
    cum_ret = float(values.iloc[-1] / values.iloc[0] - 1.0)
    if period_ret.std() == 0:
        sharpe = 0.0
    else:
        sharpe = float(np.sqrt(periods_per_year) * period_ret.mean() / period_ret.std())
    drawdown = values / values.cummax() - 1.0
    max_dd = float(drawdown.min())
    return cum_ret, sharpe, max_dd


def calculate_extended_metrics(series: pd.Series, periods_per_year: float) -> Dict[str, float]:
    values = pd.Series(series).dropna()
    if len(values) < 2:
        return {
            "cumulative_return": 0.0,
            "annualized_sharpe": 0.0,
            "max_drawdown": 0.0,
            "annualized_sortino": 0.0,
            "romad": 0.0,
            "omega": 0.0,
            "win_loss_ratio": 0.0,
        }

    period_ret = values.pct_change().dropna()
    cum_ret, sharpe, max_dd = calculate_metrics(values, periods_per_year=periods_per_year)

    downside = period_ret[period_ret < 0.0]
    downside_std = float(downside.std()) if len(downside) > 0 else 0.0
    if downside_std > 0:
        sortino = float(np.sqrt(periods_per_year) * period_ret.mean() / downside_std)
    else:
        sortino = 0.0

    max_dd_abs = abs(float(max_dd))
    romad = float(cum_ret / max_dd_abs) if max_dd_abs > 1e-12 else 0.0

    gains = float(period_ret[period_ret > 0.0].sum())
    losses = float((-period_ret[period_ret < 0.0]).sum())
    omega = float(gains / losses) if losses > 1e-12 else 0.0

    wins = int((period_ret > 0.0).sum())
    losses_n = int((period_ret < 0.0).sum())
    win_loss_ratio = float(wins / losses_n) if losses_n > 0 else float(wins > 0)

    return {
        "cumulative_return": float(cum_ret),
        "annualized_sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "annualized_sortino": float(sortino),
        "romad": float(romad),
        "omega": float(omega),
        "win_loss_ratio": float(win_loss_ratio),
    }


def summarize_policy_behavior(
    result_df: pd.DataFrame,
    position_levels: Sequence[float],
    initial_cash: float,
    context: str,
    degenerate_action_share_threshold: float = 0.95,
    degenerate_turnover_threshold: float = 1e-5,
    degenerate_trade_cost_threshold: float = -1.0,
    degenerate_account_change_threshold: float = 1e-3,
) -> Dict[str, Any]:
    actions = pd.Series(result_df.get("action", pd.Series(dtype=int))).fillna(-1).astype(int)
    actions = actions[actions >= 0]
    total_steps = int(len(actions))
    counts = collections.Counter(actions.tolist())

    if total_steps > 0:
        dominant_action, dominant_count = counts.most_common(1)[0]
        dominant_share = float(dominant_count / total_steps)
    else:
        dominant_action, dominant_share = -1, 1.0

    levels = [float(x) for x in position_levels]
    target_pos = [levels[a] if 0 <= a < len(levels) else 0.0 for a in actions.tolist()]
    target_pos_series = pd.Series(target_pos, dtype=float)
    average_target_position = float(target_pos_series.mean()) if len(target_pos_series) else 0.0
    invested_fraction = float((target_pos_series > 1e-12).mean()) if len(target_pos_series) else 0.0

    turnovers = pd.Series(result_df.get("turnover", pd.Series(dtype=float))).fillna(0.0)
    trade_cost = pd.Series(result_df.get("trade_cost", pd.Series(dtype=float))).fillna(0.0)
    account = pd.Series(result_df.get("account_value", pd.Series(dtype=float))).dropna()

    mean_turnover = float(turnovers.mean()) if len(turnovers) else 0.0
    total_trade_cost = float(trade_cost.sum()) if len(trade_cost) else 0.0
    account_change = 0.0
    if len(account) >= 2 and float(account.iloc[0]) != 0.0:
        account_change = float(account.iloc[-1] / account.iloc[0] - 1.0)

    action_share = []
    for a in range(len(levels)):
        share = float(counts.get(a, 0) / max(total_steps, 1))
        action_share.append({"action": int(a), "target_position": float(levels[a]), "share": share})

    trade_cost_threshold = (
        float(degenerate_trade_cost_threshold)
        if float(degenerate_trade_cost_threshold) >= 0.0
        else max(1e-8, float(initial_cash) * 1e-6)
    )

    diagnostics = {
        "context": context,
        "steps": total_steps,
        "dominant_action": int(dominant_action),
        "dominant_action_share": float(dominant_share),
        "average_turnover": float(mean_turnover),
        "total_trade_cost": float(total_trade_cost),
        "account_change": float(account_change),
        "average_target_position": float(average_target_position),
        "invested_fraction": float(invested_fraction),
        "degenerate_thresholds": {
            "action_share": float(degenerate_action_share_threshold),
            "avg_turnover": float(degenerate_turnover_threshold),
            "trade_cost": float(trade_cost_threshold),
            "account_change_abs": float(degenerate_account_change_threshold),
        },
        "action_share": action_share,
        "is_degenerate": bool(
            (dominant_share > float(degenerate_action_share_threshold))
            and (mean_turnover <= float(degenerate_turnover_threshold))
            and (total_trade_cost <= trade_cost_threshold)
            and (abs(account_change) <= float(degenerate_account_change_threshold))
        ),
    }
    return diagnostics


def save_policy_diagnostics(result_df: pd.DataFrame, cli, run_dir: str, stem: str):
    levels = parse_position_levels(cli.position_levels)
    diag = summarize_policy_behavior(
        result_df=result_df,
        position_levels=levels,
        initial_cash=float(cli.initial_cash),
        context=stem,
        degenerate_action_share_threshold=float(cli.degenerate_action_share_threshold),
        degenerate_turnover_threshold=float(cli.degenerate_turnover_threshold),
        degenerate_trade_cost_threshold=float(cli.degenerate_trade_cost_threshold),
        degenerate_account_change_threshold=float(cli.degenerate_account_change_threshold),
    )

    action_share_df = pd.DataFrame(diag["action_share"])
    action_share_df.to_csv(os.path.join(run_dir, f"{stem}_action_share.csv"), index=False)

    action_series = result_df["action"].fillna(-1).astype(int)
    target_pos_series = pd.Series(
        [float(levels[a]) if 0 <= int(a) < len(levels) else 0.0 for a in action_series.tolist()],
        dtype=float,
    )
    action_time_df = pd.DataFrame(
        {
            "date": result_df.get("date", pd.Series(np.arange(len(result_df)))),
            "action": action_series.to_numpy(dtype=int),
            "target_position": target_pos_series.to_numpy(dtype=float),
            "turnover": pd.Series(result_df["turnover"]).to_numpy(dtype=float),
            "trade_cost": pd.Series(result_df["trade_cost"]).to_numpy(dtype=float),
            "account_value": pd.Series(result_df["account_value"]).to_numpy(dtype=float),
        }
    )
    action_time_df.to_csv(os.path.join(run_dir, f"{stem}_action_timeseries.csv"), index=False)
    exposure = target_pos_series
    exposure_df = pd.DataFrame({"exposure": exposure})
    exposure_df.to_csv(os.path.join(run_dir, f"{stem}_exposure_distribution.csv"), index=False)

    save_json(
        os.path.join(run_dir, f"{stem}_diagnostics.json"),
        {
            "summary": diag,
            "turnover_summary": {
                "mean": float(pd.Series(result_df["turnover"]).mean()),
                "median": float(pd.Series(result_df["turnover"]).median()),
                "p95": float(pd.Series(result_df["turnover"]).quantile(0.95)),
                "sum": float(pd.Series(result_df["turnover"]).sum()),
            },
            "trade_cost_summary": {
                "mean": float(pd.Series(result_df["trade_cost"]).mean()),
                "median": float(pd.Series(result_df["trade_cost"]).median()),
                "p95": float(pd.Series(result_df["trade_cost"]).quantile(0.95)),
                "sum": float(pd.Series(result_df["trade_cost"]).sum()),
            },
        },
    )

    if diag["is_degenerate"]:
        print(
            f"[WARN] Degenerate policy detected ({stem}): "
            f"dominant_action_share={diag['dominant_action_share']:.4f}, "
            f"avg_turnover={diag['average_turnover']:.8f}, "
            f"account_change={diag['account_change']:.6f}"
        )
    return diag


def rebalance_to_target(
    cash: float,
    coin: float,
    price: float,
    target_position: float,
    fee_pct: float,
    slippage_pct: float,
) -> Tuple[float, float, float, float]:
    total_asset = cash + coin * price
    target_coin_value = target_position * total_asset
    current_coin_value = coin * price
    delta_value = target_coin_value - current_coin_value

    turnover = 0.0
    trade_cost = 0.0
    if delta_value > 0:
        buy_value = min(delta_value, cash)
        exec_price = price * (1.0 + slippage_pct)
        if exec_price > 0 and buy_value > 0:
            qty = buy_value / (exec_price * (1.0 + fee_pct))
            spent = qty * exec_price * (1.0 + fee_pct)
            cash -= spent
            coin += qty
            turnover = spent / max(total_asset, 1e-8)
            trade_cost = max(0.0, spent - qty * price)
    elif delta_value < 0:
        sell_value = min(-delta_value, current_coin_value)
        exec_price = price * (1.0 - slippage_pct)
        if exec_price > 0 and sell_value > 0:
            qty = min(coin, sell_value / exec_price)
            proceeds = qty * exec_price * (1.0 - fee_pct)
            coin -= qty
            cash += proceeds
            turnover = (qty * exec_price) / max(total_asset, 1e-8)
            trade_cost = max(0.0, qty * price - proceeds)

    cash = max(0.0, float(cash))
    coin = max(0.0, float(coin))
    return cash, coin, float(turnover), float(trade_cost)


class CryptoDiscreteTradingEnv:
    """
    Old Gym API for ElegantRL compatibility:
      reset() -> state
      step(action) -> state, reward, done, info
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        position_levels: Sequence[float],
        initial_cash: float = 100000.0,
        fee_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        reward_scale: float = 100.0,
        turnover_penalty: float = 0.0,
        drawdown_penalty: float = 0.0,
        terminal_liquidation: bool = True,
        env_name: str = "CryptoD3QNEnv",
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        if_discrete: bool = True,
        max_step: Optional[int] = None,
        **kwargs,
    ):
        self.df = df.reset_index(drop=True)
        self.feature_cols = list(feature_cols)
        self.position_levels = tuple(float(x) for x in position_levels)

        self.price_array = self.df["close"].to_numpy(dtype=np.float32)
        self.features = self.df[self.feature_cols].to_numpy(dtype=np.float32)

        self.initial_cash = float(initial_cash)
        self.fee_pct = float(fee_pct)
        self.slippage_pct = float(slippage_pct)
        self.reward_scale = float(reward_scale)
        self.turnover_penalty = float(turnover_penalty)
        self.drawdown_penalty = float(drawdown_penalty)
        self.terminal_liquidation = bool(terminal_liquidation)

        self.env_name = env_name
        self.num_envs = 1
        self.max_step = int(max_step) if max_step is not None else len(self.price_array) - 1
        self.state_dim = int(state_dim) if state_dim is not None else 3 + len(self.feature_cols)
        self.action_dim = int(action_dim) if action_dim is not None else len(self.position_levels)
        self.if_discrete = bool(if_discrete)

        self.day = 0
        self.cash = self.initial_cash
        self.coin = 0.0
        self.peak_asset = self.initial_cash
        self.account_value: List[float] = []
        self.action_memory: List[int] = []
        self.turnover_memory: List[float] = []
        self.trade_cost_memory: List[float] = []

    def _total_asset(self, price: float) -> float:
        return float(self.cash + self.coin * price)

    def _get_state(self) -> np.ndarray:
        price = float(self.price_array[self.day])
        total_asset = self._total_asset(price) + 1e-8
        cash_ratio = self.cash / self.initial_cash
        coin_value_ratio = (self.coin * price) / self.initial_cash
        position_ratio = (self.coin * price) / total_asset
        obs = np.concatenate(([cash_ratio, coin_value_ratio, position_ratio], self.features[self.day]), axis=0)
        return obs.astype(np.float32)

    def reset(self):
        self.day = 0
        self.cash = self.initial_cash
        self.coin = 0.0
        self.peak_asset = self.initial_cash
        self.account_value = [self.initial_cash]
        self.action_memory = [-1]
        self.turnover_memory = [0.0]
        self.trade_cost_memory = [0.0]
        return self._get_state()

    def step(self, action):
        action = int(action)
        if action < 0 or action >= len(self.position_levels):
            action = len(self.position_levels) // 2
        target_position = self.position_levels[action]

        price = float(self.price_array[self.day])
        prev_asset = self._total_asset(price)
        self.cash, self.coin, turnover, trade_cost = rebalance_to_target(
            cash=self.cash,
            coin=self.coin,
            price=price,
            target_position=target_position,
            fee_pct=self.fee_pct,
            slippage_pct=self.slippage_pct,
        )

        self.day += 1
        done = self.day >= self.max_step
        next_price = float(self.price_array[self.day])

        if done and self.terminal_liquidation and self.coin > 0:
            exec_price = next_price * (1.0 - self.slippage_pct)
            qty = self.coin
            proceeds = qty * exec_price * (1.0 - self.fee_pct)
            total_before_liq = self._total_asset(next_price)
            turnover += (qty * exec_price) / max(total_before_liq, 1e-8)
            trade_cost += max(0.0, qty * next_price - proceeds)
            self.cash += proceeds
            self.coin = 0.0

        new_asset = self._total_asset(next_price)
        self.account_value.append(new_asset)
        self.action_memory.append(action)
        self.turnover_memory.append(turnover)
        self.trade_cost_memory.append(trade_cost)
        self.peak_asset = max(self.peak_asset, new_asset)

        log_ret = np.log((new_asset + 1e-8) / (prev_asset + 1e-8))
        drawdown = max(0.0, (self.peak_asset - new_asset) / max(self.peak_asset, 1e-8))
        reward = (
            log_ret * self.reward_scale
            - self.turnover_penalty * turnover
            - self.drawdown_penalty * drawdown
        )
        return self._get_state(), float(reward), done, {}

    def close(self):
        return None


def setup_q_agent_args(cli, env_args: Dict, cwd_path: str, if_remove: bool, agent_class, agent_family: str):
    args = Arguments(agent_class=agent_class, env_class=CryptoDiscreteTradingEnv)
    args.env_args = env_args
    args.env_name = env_args["env_name"]

    args.net_dims = parse_net_dims(cli.net_dims)
    args.state_dim = env_args["state_dim"]
    args.action_dim = env_args["action_dim"]
    args.if_discrete = True

    args.gamma = float(cli.gamma)
    args.learning_rate = float(cli.learning_rate)
    args.batch_size = int(cli.batch_size)
    args.buffer_size = int(cli.buffer_size)
    args.horizon_len = int(cli.horizon_len)
    args.repeat_times = float(cli.repeat_times)

    args.break_step = int(cli.break_step)
    args.eval_per_step = int(cli.eval_per_step)
    args.eval_times = int(cli.eval_times)
    args.num_workers = 1
    args.random_seed = int(cli.seed)

    auto_gpu = 0 if torch.cuda.is_available() else -1
    args.gpu_id = auto_gpu if cli.gpu_id is None else int(cli.gpu_id)

    args.cwd = cwd_path
    args.if_remove = bool(if_remove)
    args.if_keep_save = True
    args.if_over_write = False
    args.agent_family = str(agent_family).lower()
    return args


def torch_load_compat(path: str, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _looks_like_state_dict(obj: Any) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    for k, v in obj.items():
        if not isinstance(k, str) or (not torch.is_tensor(v)):
            return False
    return True


def _extract_state_dict(obj: Any) -> Optional[Dict[str, Any]]:
    if _looks_like_state_dict(obj):
        return dict(obj)

    if isinstance(obj, dict):
        for key in [
            "state_dict",
            "model_state_dict",
            "actor_state_dict",
            "policy_state_dict",
            "act",
            "actor",
            "model",
            "net",
            "module",
        ]:
            if key in obj:
                extracted = _extract_state_dict(obj[key])
                if extracted is not None:
                    return extracted
        for value in obj.values():
            extracted = _extract_state_dict(value)
            if extracted is not None:
                return extracted
        return None

    if isinstance(obj, (list, tuple)):
        for value in obj:
            extracted = _extract_state_dict(value)
            if extracted is not None:
                return extracted
        return None

    state_dict_fn = getattr(obj, "state_dict", None)
    if callable(state_dict_fn):
        try:
            sd = state_dict_fn()
        except Exception:
            return None
        if _looks_like_state_dict(sd):
            return dict(sd)

    return None


def _strip_prefix_if_all_keys_match(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _state_dict_variants(state_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants = [state_dict]
    variants.append(_strip_prefix_if_all_keys_match(state_dict, "module."))
    variants.append(_strip_prefix_if_all_keys_match(state_dict, "act."))
    variants.append(_strip_prefix_if_all_keys_match(state_dict, "actor."))
    variants.append(_strip_prefix_if_all_keys_match(variants[1], "act."))
    variants.append(_strip_prefix_if_all_keys_match(variants[1], "actor."))

    unique = []
    seen = set()
    for sd in variants:
        sig = (len(sd), next(iter(sd.keys()), ""))
        if sig not in seen:
            seen.add(sig)
            unique.append(sd)
    return unique


def load_actor_weights(agent, model_path: str) -> bool:
    try:
        obj = torch_load_compat(model_path, map_location=agent.device)
        state_dict = _extract_state_dict(obj)
        if state_dict is None:
            print(f"[WARN] Unsupported checkpoint format for {model_path}: {type(obj)}")
            return False

        for candidate_sd in _state_dict_variants(state_dict):
            try:
                agent.act.load_state_dict(candidate_sd, strict=True)
                return True
            except Exception:
                pass

        for candidate_sd in _state_dict_variants(state_dict):
            try:
                info = agent.act.load_state_dict(candidate_sd, strict=False)
                missing = len(getattr(info, "missing_keys", []))
                total = len(agent.act.state_dict())
                loaded = max(0, total - missing)
                if loaded > 0:
                    print(f"[WARN] Non-strict checkpoint load for {model_path}: loaded {loaded}/{total} tensors.")
                    return True
            except Exception:
                pass
        print(f"[WARN] Checkpoint keys do not match actor network: {model_path}")
        return False
    except Exception as exc:
        print(f"[WARN] Failed to load checkpoint {model_path}: {exc}")
        return False


def list_candidate_checkpoints(cwd_path: str, max_checkpoints_eval: int) -> List[str]:
    candidates = []
    for pattern in ["actor*.pt", "actor*.pth", "act.pth"]:
        candidates.extend(glob.glob(os.path.join(cwd_path, pattern)))
    candidates = sorted(set(candidates), key=lambda p: os.path.getmtime(p))
    if not candidates:
        return []

    act_path = os.path.join(cwd_path, "act.pth")
    if act_path in candidates:
        candidates = [p for p in candidates if p != act_path]
        keep = candidates[-max(1, int(max_checkpoints_eval) - 1) :]
        keep.append(act_path)
        return keep
    return candidates[-max(1, int(max_checkpoints_eval)) :]


def checkpoint_inventory(cwd_path: str) -> pd.DataFrame:
    rows = []
    for pattern in ["actor*.pt", "actor*.pth", "act.pth", "cri*.pth"]:
        for path in glob.glob(os.path.join(cwd_path, pattern)):
            rows.append(
                {
                    "checkpoint": path,
                    "filename": os.path.basename(path),
                    "bytes": int(os.path.getsize(path)),
                    "mtime": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["checkpoint", "filename", "bytes", "mtime"])
    inv = pd.DataFrame(rows).drop_duplicates(subset=["checkpoint"]).sort_values("mtime").reset_index(drop=True)
    return inv


def cleanup_checkpoints_for_warm_start(cwd_path: str):
    for pattern in ["actor*.pt", "actor*.pth", "act.pth", "cri*.pth", "act_optimizer.pth", "cri_optimizer.pth"]:
        for path in glob.glob(os.path.join(cwd_path, pattern)):
            try:
                os.remove(path)
            except OSError:
                pass


def run_policy_once(
    data_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_path: str,
    args,
    cli,
    agent_class=AgentD3QN,
    result_tag: str = "D3QN",
) -> pd.DataFrame:
    env = CryptoDiscreteTradingEnv(
        df=data_df,
        feature_cols=feature_cols,
        position_levels=parse_position_levels(cli.position_levels),
        initial_cash=cli.initial_cash,
        fee_pct=cli.fee_pct,
        slippage_pct=cli.slippage_pct,
        reward_scale=cli.reward_scale,
        turnover_penalty=cli.turnover_penalty,
        drawdown_penalty=cli.drawdown_penalty,
        terminal_liquidation=not cli.no_terminal_liquidation,
        env_name=f"Crypto{result_tag}Inference",
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        if_discrete=True,
        max_step=len(data_df) - 1,
    )

    agent = agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    if not load_actor_weights(agent, model_path):
        raise RuntimeError(f"Could not load actor weights from {model_path}")
    agent.act.eval()

    state = env.reset()
    done = False
    while not done:
        s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            q_values = agent.act(s_tensor)
            action = int(q_values.argmax(dim=1).item())
        state, _, done, _ = env.step(action)

    result = pd.DataFrame(
        {
            "date": data_df["date"].to_numpy(),
            "close": data_df["close"].to_numpy(dtype=float),
            "account_value": np.asarray(env.account_value, dtype=float),
            "action": np.asarray(env.action_memory, dtype=int),
            "turnover": np.asarray(env.turnover_memory, dtype=float),
            "trade_cost": np.asarray(env.trade_cost_memory, dtype=float),
        }
    )
    return result


def run_ensemble_policy_once(
    data_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_specs: Sequence[Dict[str, Any]],
    args,
    cli,
    vote_mode: str = "majority",
) -> pd.DataFrame:
    env = CryptoDiscreteTradingEnv(
        df=data_df,
        feature_cols=feature_cols,
        position_levels=parse_position_levels(cli.position_levels),
        initial_cash=cli.initial_cash,
        fee_pct=cli.fee_pct,
        slippage_pct=cli.slippage_pct,
        reward_scale=cli.reward_scale,
        turnover_penalty=cli.turnover_penalty,
        drawdown_penalty=cli.drawdown_penalty,
        terminal_liquidation=not cli.no_terminal_liquidation,
        env_name="CryptoEnsembleInference",
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        if_discrete=True,
        max_step=len(data_df) - 1,
    )

    agents = []
    for spec in model_specs:
        cls = spec["agent_class"]
        model_path = spec["model_path"]
        a = cls(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
        if not load_actor_weights(a, model_path):
            raise RuntimeError(f"Could not load actor weights for ensemble member from {model_path}")
        a.act.eval()
        agents.append(a)

    state = env.reset()
    done = False
    while not done:
        s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=agents[0].device)
        member_actions = []
        member_q = []
        with torch.no_grad():
            for a in agents:
                q = a.act(s_tensor)
                member_q.append(q.detach().cpu().numpy().reshape(-1))
                member_actions.append(int(q.argmax(dim=1).item()))

        if str(vote_mode).lower() == "avg_q":
            avg_q = np.mean(np.vstack(member_q), axis=0)
            action = int(np.argmax(avg_q))
        else:
            counts = collections.Counter(member_actions)
            top_count = max(counts.values())
            ties = sorted([k for k, v in counts.items() if v == top_count])
            if len(ties) == 1:
                action = int(ties[0])
            else:
                avg_q = np.mean(np.vstack(member_q), axis=0)
                action = int(ties[np.argmax([avg_q[t] for t in ties])])

        state, _, done, _ = env.step(action)

    result = pd.DataFrame(
        {
            "date": data_df["date"].to_numpy(),
            "close": data_df["close"].to_numpy(dtype=float),
            "account_value": np.asarray(env.account_value, dtype=float),
            "action": np.asarray(env.action_memory, dtype=int),
            "turnover": np.asarray(env.turnover_memory, dtype=float),
            "trade_cost": np.asarray(env.trade_cost_memory, dtype=float),
        }
    )
    return result


def select_best_checkpoint(
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    args,
    cli,
    periods_per_year: float,
    agent_class=AgentD3QN,
    agent_family: str = "d3qn",
) -> Tuple[str, pd.DataFrame]:
    candidates = list_candidate_checkpoints(args.cwd, cli.max_checkpoints_eval)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {args.cwd}")
    print(f"[INFO] Evaluating {len(candidates)} candidate checkpoints for validation selection ({agent_label(agent_family)}).")

    rows = []
    levels = parse_position_levels(cli.position_levels)
    for path in candidates:
        try:
            val_result = run_policy_once(
                val_df,
                feature_cols,
                path,
                args,
                cli,
                agent_class=agent_class,
                result_tag=agent_label(agent_family),
            )
        except Exception as exc:
            print(f"[WARN] Skipping checkpoint (not loadable): {path} | {exc}")
            continue
        ext = calculate_extended_metrics(val_result["account_value"], periods_per_year=periods_per_year)
        diag = summarize_policy_behavior(
            result_df=val_result,
            position_levels=levels,
            initial_cash=float(cli.initial_cash),
            context=f"val_candidate_{os.path.basename(path)}",
            degenerate_action_share_threshold=float(cli.degenerate_action_share_threshold),
            degenerate_turnover_threshold=float(cli.degenerate_turnover_threshold),
            degenerate_trade_cost_threshold=float(cli.degenerate_trade_cost_threshold),
            degenerate_account_change_threshold=float(cli.degenerate_account_change_threshold),
        )
        rows.append(
            {
                "checkpoint": path,
                "agent_family": str(agent_family).lower(),
                **ext,
                "dominant_action_share": float(diag["dominant_action_share"]),
                "avg_turnover_per_step": float(diag["average_turnover"]),
                "is_degenerate": bool(diag["is_degenerate"]),
            }
        )
    if not rows:
        raise RuntimeError(f"No loadable checkpoints found in {args.cwd}.")
    table = pd.DataFrame(rows)
    table["drawdown_abs"] = table["max_drawdown"].abs()

    metric = cli.selection_metric
    if metric == "sharpe":
        threshold = float(cli.selection_min_return)
        eligible = table[table["cumulative_return"] >= threshold]
        if eligible.empty:
            eligible = table
        best = eligible.sort_values(
            ["is_degenerate", "annualized_sharpe", "cumulative_return"],
            ascending=[True, False, False],
        ).iloc[0]
    elif metric == "return":
        best = table.sort_values(
            ["is_degenerate", "cumulative_return", "annualized_sharpe"],
            ascending=[True, False, False],
        ).iloc[0]
    elif metric == "composite":
        threshold = float(cli.selection_min_return)
        penalty = np.where(table["cumulative_return"] >= threshold, 0.0, 1.0)
        degenerate_penalty = np.where(table["is_degenerate"], 1.0, 0.0)
        table["composite_score"] = (
            table["annualized_sharpe"]
            + float(cli.selection_w_return) * table["cumulative_return"]
            - float(cli.selection_w_drawdown) * table["drawdown_abs"]
            - penalty
            - degenerate_penalty
        )
        best = table.sort_values(
            ["is_degenerate", "composite_score", "annualized_sharpe"],
            ascending=[True, False, False],
        ).iloc[0]
    else:
        best = table.sort_values(
            ["is_degenerate", "drawdown_abs", "annualized_sharpe"],
            ascending=[True, True, False],
        ).iloc[0]

    table["selected"] = table["checkpoint"] == best["checkpoint"]

    return str(best["checkpoint"]), table


def simulate_buy_hold(close: pd.Series, initial_cash: float, fee_pct: float, slippage_pct: float) -> pd.Series:
    p0 = float(close.iloc[0])
    effective_buy = p0 * (1.0 + slippage_pct) * (1.0 + fee_pct)
    qty = 0.0 if effective_buy <= 0 else initial_cash / effective_buy
    values = qty * close.to_numpy(dtype=float)
    return pd.Series(values, index=close.index)


def simulate_cash(close: pd.Series, initial_cash: float) -> pd.Series:
    return pd.Series(np.full(len(close), initial_cash, dtype=float), index=close.index)


def simulate_target_strategy(
    close: pd.Series,
    target_position: pd.Series,
    initial_cash: float,
    fee_pct: float,
    slippage_pct: float,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    cash = float(initial_cash)
    coin = 0.0
    values = []
    turnovers = []
    costs = []
    for idx, price in enumerate(close.to_numpy(dtype=float)):
        target = float(target_position.iloc[idx])
        cash, coin, turnover, trade_cost = rebalance_to_target(cash, coin, price, target, fee_pct, slippage_pct)
        values.append(cash + coin * price)
        turnovers.append(turnover)
        costs.append(trade_cost)
    return (
        pd.Series(values, index=close.index),
        pd.Series(turnovers, index=close.index),
        pd.Series(costs, index=close.index),
    )


def simulate_sma_timing(
    close: pd.Series,
    initial_cash: float,
    fee_pct: float,
    slippage_pct: float,
    fast: int = 10,
    slow: int = 30,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()
    signal = (sma_fast > sma_slow).shift(1).fillna(False).astype(float)
    return simulate_target_strategy(close, signal, initial_cash, fee_pct, slippage_pct)


def simulate_constant_mix(
    close: pd.Series,
    initial_cash: float,
    fee_pct: float,
    slippage_pct: float,
    target: float = 0.5,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    target_series = pd.Series(np.full(len(close), float(target), dtype=float), index=close.index)
    return simulate_target_strategy(close, target_series, initial_cash, fee_pct, slippage_pct)


def simulate_vol_scaled_momentum(
    close: pd.Series,
    initial_cash: float,
    fee_pct: float,
    slippage_pct: float,
    mom_window: int = 20,
    vol_window: int = 20,
    scale: float = 5.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ret = close.pct_change().fillna(0.0)
    mom = close.pct_change(mom_window).fillna(0.0)
    vol = ret.rolling(vol_window).std().replace(0.0, np.nan)
    signal_raw = (mom / (vol + 1e-8)) / max(scale, 1e-8)
    target = signal_raw.clip(lower=0.0, upper=1.0).shift(1).fillna(0.0)
    return simulate_target_strategy(close, target, initial_cash, fee_pct, slippage_pct)


def add_compounded_curve(df: pd.DataFrame, value_col: str, out_col: str, initial_cash: float, window_col: str) -> pd.DataFrame:
    out_chunks = []
    capital = float(initial_cash)
    for _, g in df.groupby(window_col, sort=True):
        base = float(g[value_col].iloc[0])
        gg = g.copy()
        if base <= 0:
            gg[out_col] = capital
        else:
            gg[out_col] = capital * (gg[value_col] / base)
        capital = float(gg[out_col].iloc[-1])
        out_chunks.append(gg)
    return pd.concat(out_chunks, ignore_index=True)


def evaluate_and_plot(
    result_df: pd.DataFrame,
    cli,
    periods_per_year: float,
    plot_file: str,
    metrics_file: str,
    window_metrics_file: Optional[str] = None,
):
    df = result_df.copy()
    window_col = "window_id" if "window_id" in df.columns else "_single_window"
    if window_col == "_single_window":
        df[window_col] = 0
    primary_label = agent_label(getattr(cli, "agent_family", "d3qn"))

    baseline_chunks = []
    for _, g in df.groupby(window_col, sort=True):
        gg = g.copy().sort_values("date").reset_index(drop=True)
        close = gg["close"]
        gg["buy_hold"] = simulate_buy_hold(close, cli.initial_cash, cli.fee_pct, cli.slippage_pct)
        gg["cash"] = simulate_cash(close, cli.initial_cash)
        gg["sma_timing"], gg["sma_turnover"], gg["sma_trade_cost"] = simulate_sma_timing(
            close, cli.initial_cash, cli.fee_pct, cli.slippage_pct
        )
        gg["constant_mix_50"], gg["cmix_turnover"], gg["cmix_trade_cost"] = simulate_constant_mix(
            close, cli.initial_cash, cli.fee_pct, cli.slippage_pct, target=0.5
        )
        gg["vol_mom"], gg["vol_mom_turnover"], gg["vol_mom_trade_cost"] = simulate_vol_scaled_momentum(
            close=close,
            initial_cash=cli.initial_cash,
            fee_pct=cli.fee_pct,
            slippage_pct=cli.slippage_pct,
            mom_window=int(cli.baseline_mom_window),
            vol_window=int(cli.baseline_vol_window),
            scale=float(cli.baseline_mom_scale),
        )
        baseline_chunks.append(gg)
    df = pd.concat(baseline_chunks, ignore_index=True)

    df = add_compounded_curve(df, "account_value", "d3qn_compounded", cli.initial_cash, window_col)
    df = add_compounded_curve(df, "buy_hold", "buy_hold_compounded", cli.initial_cash, window_col)
    df = add_compounded_curve(df, "cash", "cash_compounded", cli.initial_cash, window_col)
    df = add_compounded_curve(df, "sma_timing", "sma_timing_compounded", cli.initial_cash, window_col)
    df = add_compounded_curve(df, "constant_mix_50", "constant_mix_50_compounded", cli.initial_cash, window_col)
    df = add_compounded_curve(df, "vol_mom", "vol_mom_compounded", cli.initial_cash, window_col)

    rows = []
    for name, col, turnover_col, cost_col in [
        (primary_label, "d3qn_compounded", "turnover", "trade_cost"),
        ("BuyHold", "buy_hold_compounded", None, None),
        ("Cash(Always)", "cash_compounded", None, None),
        ("SMA(10,30)-Timing", "sma_timing_compounded", "sma_turnover", "sma_trade_cost"),
        ("ConstantMix50", "constant_mix_50_compounded", "cmix_turnover", "cmix_trade_cost"),
        ("VolScaledMomentum", "vol_mom_compounded", "vol_mom_turnover", "vol_mom_trade_cost"),
    ]:
        ext = calculate_extended_metrics(df[col], periods_per_year=periods_per_year)
        avg_turnover = float(df[turnover_col].mean()) if turnover_col is not None else np.nan
        total_trade_cost = float(df[cost_col].sum()) if cost_col is not None else np.nan
        rows.append(
            {
                "strategy": name,
                **ext,
                "avg_turnover_per_step": avg_turnover,
                "total_trade_cost": total_trade_cost,
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_file, index=False)

    window_metrics_rows = []
    for wid, g in df.groupby(window_col, sort=True):
        row = {"window_id": int(wid)}
        for name, col in [
            (primary_label, "d3qn_compounded"),
            ("BuyHold", "buy_hold_compounded"),
            ("SMA", "sma_timing_compounded"),
            ("ConstMix50", "constant_mix_50_compounded"),
            ("VolMom", "vol_mom_compounded"),
        ]:
            ext = calculate_extended_metrics(g[col], periods_per_year=periods_per_year)
            key = name.lower()
            row[f"{key}_return"] = ext["cumulative_return"]
            row[f"{key}_sharpe"] = ext["annualized_sharpe"]
            row[f"{key}_sortino"] = ext["annualized_sortino"]
            row[f"{key}_max_drawdown"] = ext["max_drawdown"]
            row[f"{key}_romad"] = ext["romad"]
            row[f"{key}_omega"] = ext["omega"]
            row[f"{key}_win_loss_ratio"] = ext["win_loss_ratio"]
        row["d3qn_avg_turnover_per_step"] = float(g["turnover"].mean())
        row["d3qn_total_trade_cost"] = float(g["trade_cost"].sum())
        window_metrics_rows.append(row)
    window_metrics = pd.DataFrame(window_metrics_rows)
    if window_metrics_file:
        window_metrics.to_csv(window_metrics_file, index=False)

    plt.figure(figsize=(16, 8))
    plt.plot(df["date"], df["d3qn_compounded"], label=primary_label, linewidth=2.3)
    plt.plot(df["date"], df["buy_hold_compounded"], label="Buy & Hold", linewidth=1.7, linestyle="--")
    plt.plot(df["date"], df["sma_timing_compounded"], label="SMA Timing", linewidth=1.5, linestyle=":")
    plt.plot(df["date"], df["constant_mix_50_compounded"], label="Constant Mix 50%", linewidth=1.5, linestyle="-.")
    plt.plot(df["date"], df["vol_mom_compounded"], label="Vol-Scaled Momentum", linewidth=1.4, linestyle=(0, (3, 2)))
    plt.plot(df["date"], df["cash_compounded"], label="Cash(Always)", linewidth=1.2, alpha=0.75)
    plt.title(f"{primary_label} Crypto Backtest")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(alpha=0.25)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=240)

    print("\n" + "=" * 84)
    print(f"{primary_label} Evaluation")
    print("=" * 84)
    for _, r in metrics.iterrows():
        turn_text = (
            f" | Turnover/step {r['avg_turnover_per_step']:.5f} | Cost {r['total_trade_cost']:.2f}"
            if pd.notna(r["avg_turnover_per_step"])
            else ""
        )
        print(
            f"{r['strategy']:<18} | "
            f"Return {r['cumulative_return'] * 100:>8.2f}% | "
            f"Sharpe {r['annualized_sharpe']:>7.3f} | "
            f"Sortino {r['annualized_sortino']:>7.3f} | "
            f"RoMaD {r['romad']:>7.3f} | "
            f"Omega {r['omega']:>7.3f} | "
            f"W/L {r['win_loss_ratio']:>6.3f} | "
            f"MaxDD {r['max_drawdown'] * 100:>8.2f}%{turn_text}"
        )
    print("=" * 84)

    return df, metrics, window_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate Q-learning agents on crypto with validation-selected checkpoints.")
    parser.add_argument(
        "--data-mode",
        type=str,
        default="yfinance",
        choices=["yfinance", "lob_rnn"],
        help="yfinance: daily OHLCV from Yahoo; lob_rnn: BTC_1sec.csv + BTC_1sec_predict.npy",
    )
    parser.add_argument(
        "--benchmark-mode",
        type=str,
        default="paper_baseline",
        choices=["paper_baseline", "extended"],
        help="paper_baseline: single train/val/test benchmark. extended: supports walk-forward.",
    )
    parser.add_argument(
        "--agent-family",
        type=str,
        default="d3qn",
        choices=["dqn", "double_dqn", "d3qn", "ensemble"],
        help="Q-agent family to run. ensemble trains dqn/double_dqn/d3qn and votes at inference.",
    )
    parser.add_argument(
        "--ensemble-vote-mode",
        type=str,
        default="majority",
        choices=["majority", "avg_q"],
        help="Voting mode for --agent-family ensemble.",
    )
    parser.add_argument("--ticker", type=str, default="BTC-USD")
    parser.add_argument("--start-date", type=str, default="2018-01-01")
    parser.add_argument("--end-date", type=str, default="2026-01-01")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--cache-file", type=str, default="./data/crypto_btc_usd_1d.csv")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--download-retries", type=int, default=6)
    parser.add_argument("--download-backoff-seconds", type=int, default=6)
    parser.add_argument(
        "--fallback-source",
        type=str,
        default="auto",
        choices=["auto", "none", "cryptodatadownload"],
        help="Data fallback source when Yahoo is unavailable.",
    )
    parser.add_argument("--lob-csv-file", type=str, default="./data/BTC_1sec.csv")
    parser.add_argument("--lob-rnn-file", type=str, default="./data/BTC_1sec_predict.npy")
    parser.add_argument("--lob-rnn-pretrain-ratio", type=float, default=0.6)
    parser.add_argument("--lob-stride", type=int, default=1, help="Use every n-th row from LOB data.")
    parser.add_argument("--lob-max-rows", type=int, default=0, help="0 means use all rows after pretrain cut.")
    parser.add_argument("--periods-per-year", type=float, default=0.0, help="0 means auto from --interval.")

    parser.add_argument("--val-size", type=int, default=180)
    parser.add_argument("--test-size", type=int, default=365)
    parser.add_argument("--walk-forward", action="store_true", help="Run multiple rolling train/val/test windows.")
    parser.add_argument("--wf-train-size", type=int, default=900)
    parser.add_argument("--wf-val-size", type=int, default=120)
    parser.add_argument("--wf-test-size", type=int, default=120)
    parser.add_argument("--wf-step-size", type=int, default=120)
    parser.add_argument("--wf-max-windows", type=int, default=0, help="0 means use all windows.")
    parser.add_argument("--wf-warm-start", action="store_true", help="Warm-start each window from previous selected checkpoint.")

    parser.add_argument("--clip-quantile", type=float, default=0.01, help="Feature clipping quantile based on train split.")
    parser.add_argument("--no-standardize-features", action="store_true")

    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--fee-pct", type=float, default=0.001)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--position-levels", type=str, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--turnover-penalty", type=float, default=0.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.0)
    parser.add_argument("--no-terminal-liquidation", action="store_true")

    parser.add_argument("--net-dims", type=str, default="256,128")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--horizon-len", type=int, default=512)
    parser.add_argument("--repeat-times", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--break-step", type=int, default=300000)
    parser.add_argument("--eval-per-step", type=int, default=5000)
    parser.add_argument("--eval-times", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=None, help="Auto if omitted. Use -1 for CPU.")
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="composite",
        choices=["sharpe", "return", "drawdown", "composite"],
        help="Validation metric to choose best checkpoint.",
    )
    parser.add_argument("--selection-min-return", type=float, default=-0.05, help="Minimum validation return threshold for selection.")
    parser.add_argument("--selection-w-return", type=float, default=0.5, help="Composite metric return weight.")
    parser.add_argument("--selection-w-drawdown", type=float, default=0.5, help="Composite metric drawdown penalty weight.")
    parser.add_argument("--max-checkpoints-eval", type=int, default=8)
    parser.add_argument("--degenerate-action-share-threshold", type=float, default=0.95)
    parser.add_argument("--degenerate-turnover-threshold", type=float, default=1e-5)
    parser.add_argument("--degenerate-trade-cost-threshold", type=float, default=-1.0, help="<0 means auto from initial cash.")
    parser.add_argument("--degenerate-account-change-threshold", type=float, default=0.001)

    parser.add_argument("--baseline-mom-window", type=int, default=20)
    parser.add_argument("--baseline-vol-window", type=int, default=20)
    parser.add_argument("--baseline-mom-scale", type=float, default=5.0)

    parser.add_argument("--run-root", type=str, default="./checkpoints/d3qn")
    parser.add_argument("--artifact-root", type=str, default="./logs/d3qn")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--run-tag", type=str, default="", help="Optional label, e.g. benchmark_comparable or debug.")
    parser.add_argument("--keep-checkpoint", action="store_true")
    parser.add_argument("--skip-train", action="store_true")

    parser.add_argument("--result-csv", type=str, default="d3qn_crypto_results.csv")
    parser.add_argument("--metrics-csv", type=str, default="d3qn_crypto_metrics.csv")
    parser.add_argument("--plot-file", type=str, default="d3qn_crypto_performance.png")
    parser.add_argument("--val-checkpoint-metrics-csv", type=str, default="d3qn_val_checkpoint_metrics.csv")
    return parser.parse_args()


def build_run_dir(run_root: str, run_name: str, ticker: str, seed: int) -> str:
    if run_name:
        name = run_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_ticker = re.sub(r"[^A-Za-z0-9_]+", "_", ticker)
        name = f"{safe_ticker}_s{seed}_{ts}"
    run_dir = os.path.join(run_root, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def preprocess_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    clip_quantile: float,
    standardize: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Tuple[float, float]], Optional[Dict[str, Tuple[float, float]]]]:
    bounds = fit_clipping_bounds(train_df, feature_cols, clip_quantile)
    train_pp = apply_clipping(train_df, bounds)
    val_pp = apply_clipping(val_df, bounds)
    test_pp = apply_clipping(test_df, bounds)

    stats = None
    if standardize:
        stats = fit_standardizer(train_pp, feature_cols)
        train_pp = apply_standardizer(train_pp, stats)
        val_pp = apply_standardizer(val_pp, stats)
        test_pp = apply_standardizer(test_pp, stats)

    return train_pp, val_pp, test_pp, bounds, stats


def run_single_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    cli,
    run_dir: str,
    periods_per_year: float,
    init_actor_checkpoint: Optional[str] = None,
    agent_family: str = "d3qn",
    agent_class=AgentD3QN,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, Dict[str, Any]]:
    train_pp, val_pp, test_pp, clip_bounds, standardizer_stats = preprocess_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        clip_quantile=float(cli.clip_quantile),
        standardize=(not cli.no_standardize_features),
    )

    levels = parse_position_levels(cli.position_levels)
    state_dim = 3 + len(feature_cols)
    env_args = {
        "env_name": f"Crypto{agent_label(agent_family)}TrainEnv",
        "df": train_pp,
        "feature_cols": list(feature_cols),
        "position_levels": levels,
        "initial_cash": float(cli.initial_cash),
        "fee_pct": float(cli.fee_pct),
        "slippage_pct": float(cli.slippage_pct),
        "reward_scale": float(cli.reward_scale),
        "turnover_penalty": float(cli.turnover_penalty),
        "drawdown_penalty": float(cli.drawdown_penalty),
        "terminal_liquidation": not bool(cli.no_terminal_liquidation),
        "state_dim": state_dim,
        "action_dim": len(levels),
        "if_discrete": True,
        "max_step": len(train_pp) - 1,
    }

    os.makedirs(run_dir, exist_ok=True)
    if_remove_flag = not bool(cli.keep_checkpoint)
    warm_start_from = None
    if (not cli.skip_train) and init_actor_checkpoint:
        cleanup_checkpoints_for_warm_start(run_dir)
        dst = os.path.join(run_dir, "act.pth")
        ckpt_obj = torch_load_compat(init_actor_checkpoint, map_location="cpu")
        state_dict = _extract_state_dict(ckpt_obj)
        if state_dict is not None:
            torch.save(state_dict, dst)
        else:
            shutil.copy2(init_actor_checkpoint, dst)
            print("[WARN] Warm-start checkpoint copied raw (state_dict extraction failed).")
        warm_start_from = init_actor_checkpoint
        if_remove_flag = False
        print(f"[INFO] Warm-start initialized from: {init_actor_checkpoint}")

    args = setup_q_agent_args(
        cli=cli,
        env_args=env_args,
        cwd_path=run_dir,
        if_remove=if_remove_flag,
        agent_class=agent_class,
        agent_family=agent_family,
    )

    if not cli.skip_train:
        print(f"[INFO] Training {agent_label(agent_family)} in {run_dir} ...")
        train_agent(args)
    else:
        print(f"[INFO] Skip training. Using existing checkpoints in {run_dir}")

    inv = checkpoint_inventory(run_dir)
    inv_path = os.path.join(run_dir, "checkpoint_inventory.csv")
    inv.to_csv(inv_path, index=False)
    print(f"[INFO] Checkpoint inventory saved: {inv_path} ({len(inv)} files)")

    best_ckpt, ckpt_metrics = select_best_checkpoint(
        val_df=val_pp,
        feature_cols=feature_cols,
        args=args,
        cli=cli,
        periods_per_year=periods_per_year,
        agent_class=agent_class,
        agent_family=agent_family,
    )
    ckpt_metrics.to_csv(os.path.join(run_dir, cli.val_checkpoint_metrics_csv), index=False)
    print(f"[INFO] Selected {agent_label(agent_family)} checkpoint by {cli.selection_metric}: {best_ckpt}")

    val_result = run_policy_once(
        data_df=val_pp,
        feature_cols=feature_cols,
        model_path=best_ckpt,
        args=args,
        cli=cli,
        agent_class=agent_class,
        result_tag=agent_label(agent_family),
    )
    val_result["date"] = val_df["date"].to_numpy()
    val_result["close"] = val_df["close"].to_numpy(dtype=float)
    val_diag = save_policy_diagnostics(val_result, cli=cli, run_dir=run_dir, stem=f"validation_{agent_family}")

    test_result = run_policy_once(
        data_df=test_pp,
        feature_cols=feature_cols,
        model_path=best_ckpt,
        args=args,
        cli=cli,
        agent_class=agent_class,
        result_tag=agent_label(agent_family),
    )
    test_result["date"] = test_df["date"].to_numpy()
    test_result["close"] = test_df["close"].to_numpy(dtype=float)
    test_result["selected_checkpoint"] = best_ckpt
    test_result["warm_start_from"] = warm_start_from if warm_start_from else ""
    test_result["agent_family"] = str(agent_family).lower()
    test_diag = save_policy_diagnostics(test_result, cli=cli, run_dir=run_dir, stem=f"test_{agent_family}")

    preprocess_meta = {
        "agent_family": str(agent_family).lower(),
        "feature_cols": list(feature_cols),
        "position_levels": list(levels),
        "clip_bounds": {k: [float(v[0]), float(v[1])] for k, v in clip_bounds.items()},
        "standardizer_stats": (
            {k: [float(v[0]), float(v[1])] for k, v in standardizer_stats.items()}
            if standardizer_stats is not None
            else None
        ),
        "split_info": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_date_start": str(train_df["date"].iloc[0]),
            "train_date_end": str(train_df["date"].iloc[-1]),
            "val_date_start": str(val_df["date"].iloc[0]),
            "val_date_end": str(val_df["date"].iloc[-1]),
            "test_date_start": str(test_df["date"].iloc[0]),
            "test_date_end": str(test_df["date"].iloc[-1]),
        },
        "selection_info": {
            "selection_metric": cli.selection_metric,
            "selection_min_return": float(cli.selection_min_return),
            "selection_w_return": float(cli.selection_w_return),
            "selection_w_drawdown": float(cli.selection_w_drawdown),
            "selected_checkpoint": best_ckpt,
            "warm_start_from": warm_start_from,
            "validation_degenerate": bool(val_diag["is_degenerate"]),
            "test_degenerate": bool(test_diag["is_degenerate"]),
        },
    }
    save_json(os.path.join(run_dir, "preprocess_artifacts.json"), preprocess_meta)
    return test_result, ckpt_metrics, best_ckpt, preprocess_meta


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    cli,
    run_dir: str,
    periods_per_year: float,
    agent_family: str = "d3qn",
    agent_class=AgentD3QN,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = []
    all_ckpt_rows = []
    all_window_meta = []

    train_size = int(cli.wf_train_size)
    val_size = int(cli.wf_val_size)
    test_size = int(cli.wf_test_size)
    step_size = int(cli.wf_step_size)

    max_windows = int(cli.wf_max_windows)
    window_id = 0
    start = 0
    prev_best_ckpt = None
    while start + train_size + val_size + test_size <= len(df):
        if max_windows > 0 and window_id >= max_windows:
            break
        tr_end = start + train_size
        va_end = tr_end + val_size
        te_end = va_end + test_size

        train_df = df.iloc[start:tr_end].copy().reset_index(drop=True)
        val_df = df.iloc[tr_end:va_end].copy().reset_index(drop=True)
        test_df = df.iloc[va_end:te_end].copy().reset_index(drop=True)

        window_dir = os.path.join(run_dir, f"window_{window_id:03d}")
        os.makedirs(window_dir, exist_ok=True)

        print(
            f"[INFO] Walk-forward window {window_id}: "
            f"train {train_df['date'].iloc[0].date()} -> {train_df['date'].iloc[-1].date()}, "
            f"val {val_df['date'].iloc[0].date()} -> {val_df['date'].iloc[-1].date()}, "
            f"test {test_df['date'].iloc[0].date()} -> {test_df['date'].iloc[-1].date()}"
        )

        init_ckpt = prev_best_ckpt if bool(cli.wf_warm_start) else None

        result_df, ckpt_df, best_ckpt, preprocess_meta = run_single_experiment(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
            cli=cli,
            run_dir=window_dir,
            periods_per_year=periods_per_year,
            init_actor_checkpoint=init_ckpt,
            agent_family=agent_family,
            agent_class=agent_class,
        )
        result_df["window_id"] = window_id
        result_df["agent_family"] = str(agent_family).lower()
        results.append(result_df)

        ckpt_df = ckpt_df.copy()
        ckpt_df["window_id"] = window_id
        ckpt_df["agent_family"] = str(agent_family).lower()
        all_ckpt_rows.append(ckpt_df)

        all_window_meta.append(
            {
                "window_id": int(window_id),
                "window_dir": window_dir,
                "selected_checkpoint": best_ckpt,
                "warm_start_from": init_ckpt if init_ckpt else "",
                "agent_family": str(agent_family).lower(),
                "train_date_start": preprocess_meta["split_info"]["train_date_start"],
                "train_date_end": preprocess_meta["split_info"]["train_date_end"],
                "val_date_start": preprocess_meta["split_info"]["val_date_start"],
                "val_date_end": preprocess_meta["split_info"]["val_date_end"],
                "test_date_start": preprocess_meta["split_info"]["test_date_start"],
                "test_date_end": preprocess_meta["split_info"]["test_date_end"],
            }
        )

        prev_best_ckpt = best_ckpt

        window_id += 1
        start += step_size

    if not results:
        raise RuntimeError("Walk-forward produced no windows. Check wf_* sizes or date range.")

    result_all = pd.concat(results, ignore_index=True)
    ckpt_all = pd.concat(all_ckpt_rows, ignore_index=True)
    meta_all = pd.DataFrame(all_window_meta)
    return result_all, ckpt_all, meta_all


def main():
    cli = parse_args()
    ensure_d3qn_dependencies()
    set_global_seed(int(cli.seed))

    if str(cli.run_tag).strip() == "":
        cli.run_tag = "benchmark_comparable" if str(cli.benchmark_mode).lower() == "paper_baseline" else "debug"

    if str(cli.benchmark_mode).lower() == "paper_baseline" and bool(cli.walk_forward):
        print("[WARN] benchmark_mode=paper_baseline enforces single train/val/test. Disabling --walk-forward.")
        cli.walk_forward = False

    if str(cli.benchmark_mode).lower() == "paper_baseline" and str(cli.data_mode).lower() != "lob_rnn":
        print("[WARN] paper_baseline is designed for lob_rnn data. You are running yfinance.")

    if str(cli.data_mode).lower() == "lob_rnn":
        if float(cli.fee_pct) == 0.001 and float(cli.slippage_pct) == 0.0005:
            cli.fee_pct = 0.00002
            cli.slippage_pct = 0.00002
            print("[WARN] Applied LOB diagnostic frictions: fee_pct=0.00002, slippage_pct=0.00002")

    periods_per_year = float(cli.periods_per_year)
    if periods_per_year <= 0:
        if str(cli.data_mode).lower() == "lob_rnn":
            periods_per_year = 24.0 * 60.0 * 60.0 * 365.0
        else:
            periods_per_year = parse_interval_to_periods(cli.interval, fallback=365.0)

    agent_registry = get_agent_registry()
    families = resolve_agent_families(cli.agent_family)

    run_dir = build_run_dir(
        run_root=cli.run_root,
        run_name=cli.run_name,
        ticker=cli.ticker,
        seed=int(cli.seed),
    )
    print(f"[INFO] Run directory: {run_dir}")
    print(f"[INFO] Run tag: {cli.run_tag}")
    print(f"[INFO] Benchmark mode: {cli.benchmark_mode} | Agent family: {cli.agent_family}")
    print(f"[INFO] Using periods_per_year={periods_per_year:.2f} for annualized Sharpe.")
    if bool(cli.walk_forward):
        print("[INFO] Evaluation protocol: rolling retrain + validation-selected checkpoint + compounded out-of-sample.")
    else:
        print("[INFO] Evaluation protocol: single train/validation/test with validation-selected checkpoint.")

    save_json(
        os.path.join(run_dir, "run_config.json"),
        {
            "cli_args": vars(cli),
            "resolved": {
                "periods_per_year": periods_per_year,
                "timestamp": datetime.now().isoformat(),
                "run_tag": cli.run_tag,
                "benchmark_mode": cli.benchmark_mode,
                "agent_family": cli.agent_family,
            },
        },
    )

    if str(cli.data_mode).lower() == "lob_rnn":
        df, feature_cols = load_lob_rnn_data(
            csv_file=cli.lob_csv_file,
            rnn_feature_file=cli.lob_rnn_file,
            rnn_pretrain_ratio=float(cli.lob_rnn_pretrain_ratio),
            stride=int(cli.lob_stride),
            max_rows=int(cli.lob_max_rows),
        )
        if bool(cli.walk_forward):
            print("[WARN] --walk-forward is enabled. Paper-style setup is typically single split for this dataset.")
    else:
        df = load_crypto_data(
            ticker=cli.ticker,
            start_date=cli.start_date,
            end_date=cli.end_date,
            interval=cli.interval,
            cache_file=cli.cache_file,
            force_download=cli.force_download,
            download_retries=cli.download_retries,
            download_backoff_seconds=cli.download_backoff_seconds,
            fallback_source=cli.fallback_source,
        )
        df, feature_cols = add_features(df)

    print(f"[INFO] Data rows: {len(df)}, features: {len(feature_cols)}")

    window_meta_df = pd.DataFrame()
    if str(cli.agent_family).lower() != "ensemble":
        family = families[0]
        agent_class = agent_registry[family]

        if cli.walk_forward:
            result_df, ckpt_df, window_meta_df = run_walk_forward(
                df=df,
                feature_cols=feature_cols,
                cli=cli,
                run_dir=run_dir,
                periods_per_year=periods_per_year,
                agent_family=family,
                agent_class=agent_class,
            )
        else:
            train_df, val_df, test_df = split_train_val_test(df=df, val_size=int(cli.val_size), test_size=int(cli.test_size))
            print(
                f"[INFO] Single split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
            )
            result_df, ckpt_df, best_ckpt, preprocess_meta = run_single_experiment(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_cols=feature_cols,
                cli=cli,
                run_dir=run_dir,
                periods_per_year=periods_per_year,
                agent_family=family,
                agent_class=agent_class,
            )
            result_df["agent_family"] = family
            window_meta_df = pd.DataFrame(
                [
                    {
                        "window_id": 0,
                        "window_dir": run_dir,
                        "selected_checkpoint": best_ckpt,
                        "warm_start_from": "",
                        "agent_family": family,
                        "train_date_start": preprocess_meta["split_info"]["train_date_start"],
                        "train_date_end": preprocess_meta["split_info"]["train_date_end"],
                        "val_date_start": preprocess_meta["split_info"]["val_date_start"],
                        "val_date_end": preprocess_meta["split_info"]["val_date_end"],
                        "test_date_start": preprocess_meta["split_info"]["test_date_start"],
                        "test_date_end": preprocess_meta["split_info"]["test_date_end"],
                    }
                ]
            )
    else:
        if cli.walk_forward:
            print("[WARN] Ensemble mode currently runs single split only. Disabling --walk-forward.")
            cli.walk_forward = False

        train_df, val_df, test_df = split_train_val_test(df=df, val_size=int(cli.val_size), test_size=int(cli.test_size))
        print(
            f"[INFO] Ensemble single split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

        train_pp, val_pp, test_pp, _, _ = preprocess_splits(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
            clip_quantile=float(cli.clip_quantile),
            standardize=(not cli.no_standardize_features),
        )
        levels = parse_position_levels(cli.position_levels)
        state_dim = 3 + len(feature_cols)
        env_args_template = {
            "env_name": "CryptoEnsembleTemplateEnv",
            "df": train_pp,
            "feature_cols": list(feature_cols),
            "position_levels": levels,
            "initial_cash": float(cli.initial_cash),
            "fee_pct": float(cli.fee_pct),
            "slippage_pct": float(cli.slippage_pct),
            "reward_scale": float(cli.reward_scale),
            "turnover_penalty": float(cli.turnover_penalty),
            "drawdown_penalty": float(cli.drawdown_penalty),
            "terminal_liquidation": not bool(cli.no_terminal_liquidation),
            "state_dim": state_dim,
            "action_dim": len(levels),
            "if_discrete": True,
            "max_step": len(train_pp) - 1,
        }
        ensemble_args = setup_q_agent_args(
            cli=cli,
            env_args=env_args_template,
            cwd_path=os.path.join(run_dir, "_ensemble_template"),
            if_remove=False,
            agent_class=agent_registry[families[0]],
            agent_family=families[0],
        )

        member_best = {}
        member_ckpts = []
        member_meta_rows = []
        for family in families:
            member_dir = os.path.join(run_dir, family)
            member_result, member_ckpt_df, best_ckpt, preprocess_meta = run_single_experiment(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_cols=feature_cols,
                cli=cli,
                run_dir=member_dir,
                periods_per_year=periods_per_year,
                agent_family=family,
                agent_class=agent_registry[family],
            )
            member_result.to_csv(os.path.join(member_dir, "member_test_result.csv"), index=False)
            member_ckpt_df["member_family"] = family
            member_ckpts.append(member_ckpt_df)
            member_best[family] = best_ckpt
            member_meta_rows.append(
                {
                    "member_family": family,
                    "selected_checkpoint": best_ckpt,
                    "train_date_start": preprocess_meta["split_info"]["train_date_start"],
                    "train_date_end": preprocess_meta["split_info"]["train_date_end"],
                    "val_date_start": preprocess_meta["split_info"]["val_date_start"],
                    "val_date_end": preprocess_meta["split_info"]["val_date_end"],
                    "test_date_start": preprocess_meta["split_info"]["test_date_start"],
                    "test_date_end": preprocess_meta["split_info"]["test_date_end"],
                }
            )

        model_specs = [
            {
                "agent_family": family,
                "agent_class": agent_registry[family],
                "model_path": member_best[family],
            }
            for family in families
        ]

        val_ensemble = run_ensemble_policy_once(
            data_df=val_pp,
            feature_cols=feature_cols,
            model_specs=model_specs,
            args=ensemble_args,
            cli=cli,
            vote_mode=cli.ensemble_vote_mode,
        )
        val_ensemble["date"] = val_df["date"].to_numpy()
        val_ensemble["close"] = val_df["close"].to_numpy(dtype=float)
        save_policy_diagnostics(val_ensemble, cli=cli, run_dir=run_dir, stem="validation_ensemble")

        result_df = run_ensemble_policy_once(
            data_df=test_pp,
            feature_cols=feature_cols,
            model_specs=model_specs,
            args=ensemble_args,
            cli=cli,
            vote_mode=cli.ensemble_vote_mode,
        )
        result_df["date"] = test_df["date"].to_numpy()
        result_df["close"] = test_df["close"].to_numpy(dtype=float)
        result_df["agent_family"] = "ensemble"
        result_df["selected_checkpoint"] = json.dumps(member_best, sort_keys=True)
        result_df["warm_start_from"] = ""
        save_policy_diagnostics(result_df, cli=cli, run_dir=run_dir, stem="test_ensemble")

        ckpt_df = pd.concat(member_ckpts, ignore_index=True) if member_ckpts else pd.DataFrame()
        window_meta_df = pd.DataFrame(
            [
                {
                    "window_id": 0,
                    "window_dir": run_dir,
                    "selected_checkpoint": json.dumps(member_best, sort_keys=True),
                    "warm_start_from": "",
                    "agent_family": "ensemble",
                    "members": json.dumps(families),
                    "test_date_start": str(test_df["date"].iloc[0]),
                    "test_date_end": str(test_df["date"].iloc[-1]),
                }
            ]
        )
        save_json(os.path.join(run_dir, "ensemble_members.json"), {"members": member_meta_rows, "vote_mode": cli.ensemble_vote_mode})

    result_csv = os.path.join(run_dir, cli.result_csv)
    metrics_csv = os.path.join(run_dir, cli.metrics_csv)
    plot_file = os.path.join(run_dir, cli.plot_file)
    ckpt_csv = os.path.join(run_dir, cli.val_checkpoint_metrics_csv)
    window_meta_csv = os.path.join(run_dir, "walkforward_window_meta.csv")
    window_metrics_csv = os.path.join(run_dir, "window_metrics.csv")

    result_df.to_csv(result_csv, index=False)
    ckpt_df.to_csv(ckpt_csv, index=False)
    window_meta_df.to_csv(window_meta_csv, index=False)

    curve_df, _, window_metrics = evaluate_and_plot(
        result_df=result_df,
        cli=cli,
        periods_per_year=periods_per_year,
        plot_file=plot_file,
        metrics_file=metrics_csv,
        window_metrics_file=window_metrics_csv,
    )
    curve_df.to_csv(os.path.join(run_dir, "evaluation_curves.csv"), index=False)
    if isinstance(window_metrics, pd.DataFrame):
        window_metrics.to_csv(window_metrics_csv, index=False)

    overall_diag = save_policy_diagnostics(result_df, cli=cli, run_dir=run_dir, stem="overall")
    print("\n" + "=" * 84)
    print("Sanity Check")
    print("=" * 84)
    print(f"Most common action: {overall_diag['dominant_action']} (share={overall_diag['dominant_action_share']:.4f})")
    print(f"Average target position: {overall_diag['average_target_position']:.6f}")
    print(f"Average turnover: {overall_diag['average_turnover']:.8f}")
    print(f"Invested fraction: {overall_diag['invested_fraction']:.6f}")
    print(f"Effectively cash policy: {bool(overall_diag['is_degenerate'])}")
    print("=" * 84)

    print(f"[INFO] Saved result rows: {result_csv}")
    print(f"[INFO] Saved checkpoint validation metrics: {ckpt_csv}")
    print(f"[INFO] Saved walk-forward window meta: {window_meta_csv}")
    print(f"[INFO] Saved per-window metrics: {window_metrics_csv}")
    print(f"[INFO] Saved evaluation metrics: {metrics_csv}")
    print(f"[INFO] Saved plot: {plot_file}")
    artifact_dir, artifact_count = export_csv_artifacts(run_dir=run_dir, artifact_root=cli.artifact_root)
    print(f"[INFO] Exported CSV artifacts ({artifact_count} files): {artifact_dir}")


if __name__ == "__main__":
    main()
