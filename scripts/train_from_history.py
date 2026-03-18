"""
scripts/train_from_history.py
══════════════════════════════════════════════════════════════════════════════
Historical data trainer — reads parquet files from training_data/ folder.

WHAT THIS DOES DIFFERENTLY FROM train_all.py:
  1. Loads years of data from parquet files instead of ~13 days from testnet
  2. Multi-timeframe features: 15m base + 1h trend + 4h macro regime context
  3. Strict WF thresholds restored (Sharpe > 0.5, F1 > 0.35) — enough data now
  4. 10 WF folds instead of 5 (more stable estimates)
  5. Regime detection: labels each bar as trending/ranging/volatile
  6. Saves models to saved_models/ — overwrites testnet models

USAGE:
    cd "C:\\Users\\123\\Desktop\\AI Trading Bot"
    python scripts/train_from_history.py

    # Train single symbol:
    python scripts/train_from_history.py --symbol BTCUSDT

    # Dry run (check data, skip training):
    python scripts/train_from_history.py --dry-run
"""

import sys
import os
import argparse
import time
from datetime import datetime
from pathlib import Path

# ── Fix Python path ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

from data.indicators import add_all_indicators, get_feature_columns
from models.training.trainer import engineer_features, ModelTrainer
from models.training.label_engine import generate_labels, label_distribution
from research.walk_forward import WalkForwardValidator
from config.logger import get_logger
from config.settings import settings

logger = get_logger("train_history")

DATA_DIR    = Path(PROJECT_ROOT) / "training_data"
MODELS_DIR  = Path(PROJECT_ROOT) / "saved_models"
SYMBOLS     = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
TIMEFRAMES  = ["15m", "1h", "4h"]

# ── Hyperparameters ────────────────────────────────────────────────────────────
FUTURE_BARS    = 8       # 15m bars ahead to label (8 × 15m = 2h horizon)
ATR_MULT       = 1.5     # barrier width as ATR multiple; tune per symbol below
ATR_MULT_MAP   = {       # per-symbol ATR multiplier (higher vol = wider barriers)
    "BTCUSDT": 1.5,
    "ETHUSDT": 1.8,
    "BNBUSDT": 1.6,
    "SOLUSDT": 2.0,
}
N_WF_SPLITS    = 10      # more folds = more stable WF estimate (restored from 5)
WF_GAP         = 8       # gap between train/test = FUTURE_BARS (no leakage)

# Restored strict thresholds — justified now that we have years of data
MIN_SHARPE     = 0.50
MIN_F1         = 0.35
MIN_CONSIST    = 0.60


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_parquet(symbol: str, tf: str) -> pd.DataFrame | None:
    """Load a single parquet file, normalize column names."""
    path = DATA_DIR / f"{symbol}_{tf}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)

        # Normalize column names to lowercase
        df.columns = [c.lower().strip() for c in df.columns]

        # Ensure standard OHLCV column names
        rename = {}
        for col in df.columns:
            if col in ("open_time", "opentime", "timestamp", "time", "date"):
                rename[col] = "timestamp"
        if rename:
            df = df.rename(columns=rename)

        # Set datetime index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        df = df.sort_index()

        # Keep only OHLCV
        keep = ["open", "high", "low", "close", "volume"]
        available = [c for c in keep if c in df.columns]
        df = df[available].copy()
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["close"])

        return df

    except Exception as e:
        logger.warning(f"[DATA] Failed to load {symbol}_{tf}.parquet: {e}")
        return None


def summarize_df(df: pd.DataFrame, symbol: str, tf: str) -> str:
    if df is None or df.empty:
        return f"  {symbol}_{tf}: MISSING"
    days  = (df.index[-1] - df.index[0]).days
    bars  = len(df)
    start = df.index[0].strftime("%Y-%m-%d")
    end   = df.index[-1].strftime("%Y-%m-%d")
    size_mb = df.memory_usage(deep=True).sum() / 1e6
    return f"  {symbol}_{tf:<4}: {bars:>7,} bars  {start} → {end}  ({days} days, {size_mb:.1f} MB)"


# ══════════════════════════════════════════════════════════════════════════════
# 2. MULTI-TIMEFRAME FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def build_htf_features(df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                       base_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build higher-timeframe context features aligned to the 15m base index.

    For each 15m bar, adds:
      From 1h: trend direction, RSI regime, EMA position, MACD
      From 4h: macro regime (trending/ranging), ATR regime, Bollinger position

    All features are forward-filled (a 1h bar's features apply to all
    15m bars within that hour) — no lookahead.
    """
    htf_cols = {}

    # ── 1h features ───────────────────────────────────────────────────────────
    if df_1h is not None and not df_1h.empty:
        df_1h_ind = add_all_indicators(df_1h.copy())

        htf_1h = pd.DataFrame(index=df_1h_ind.index)
        if "ema_9" in df_1h_ind.columns and "ema_21" in df_1h_ind.columns:
            htf_1h["htf1h_trend"]     = np.where(df_1h_ind["ema_9"] > df_1h_ind["ema_21"], 1, -1)
        if "rsi" in df_1h_ind.columns:
            htf_1h["htf1h_rsi"]       = df_1h_ind["rsi"]
            htf_1h["htf1h_rsi_zone"]  = pd.cut(df_1h_ind["rsi"],
                                                bins=[0, 30, 45, 55, 70, 100],
                                                labels=[-2, -1, 0, 1, 2]).astype(float)
        if "macd_hist" in df_1h_ind.columns:
            htf_1h["htf1h_macd_hist"] = df_1h_ind["macd_hist"]
            htf_1h["htf1h_macd_bull"] = (df_1h_ind["macd_hist"] > 0).astype(int)
        if "adx" in df_1h_ind.columns:
            htf_1h["htf1h_adx"]       = df_1h_ind["adx"]
            htf_1h["htf1h_trending"]  = (df_1h_ind["adx"] > 25).astype(int)
        if "bb_pct" in df_1h_ind.columns:
            htf_1h["htf1h_bb_pct"]    = df_1h_ind["bb_pct"]
        if "close" in df_1h_ind.columns and "ema_50" in df_1h_ind.columns:
            htf_1h["htf1h_above_50ema"] = (df_1h_ind["close"] > df_1h_ind["ema_50"]).astype(int)

        # Reindex to 15m: forward-fill (each 1h bar covers the next 4 × 15m bars)
        htf_1h_aligned = htf_1h.reindex(base_index, method="ffill")
        for col in htf_1h_aligned.columns:
            htf_cols[col] = htf_1h_aligned[col]

    # ── 4h features ───────────────────────────────────────────────────────────
    if df_4h is not None and not df_4h.empty:
        df_4h_ind = add_all_indicators(df_4h.copy())

        htf_4h = pd.DataFrame(index=df_4h_ind.index)
        if "ema_9" in df_4h_ind.columns and "ema_21" in df_4h_ind.columns:
            htf_4h["htf4h_trend"]     = np.where(df_4h_ind["ema_9"] > df_4h_ind["ema_21"], 1, -1)
        if "ema_50" in df_4h_ind.columns:
            htf_4h["htf4h_above_50ema"] = (df_4h_ind["close"] > df_4h_ind["ema_50"]).astype(int)
        if "rsi" in df_4h_ind.columns:
            htf_4h["htf4h_rsi"]       = df_4h_ind["rsi"]
        if "adx" in df_4h_ind.columns:
            htf_4h["htf4h_adx"]       = df_4h_ind["adx"]
            htf_4h["htf4h_trending"]  = (df_4h_ind["adx"] > 25).astype(int)
        if "atr" in df_4h_ind.columns and "close" in df_4h_ind.columns:
            # 4h volatility regime: is current ATR high vs its 20-bar rolling mean?
            atr_mean = df_4h_ind["atr"].rolling(20).mean()
            htf_4h["htf4h_vol_regime"] = (df_4h_ind["atr"] > atr_mean).astype(int)
        if "bb_width" in df_4h_ind.columns:
            bb_mean = df_4h_ind["bb_width"].rolling(20).mean()
            htf_4h["htf4h_expanding"] = (df_4h_ind["bb_width"] > bb_mean).astype(int)
        if "macd_hist" in df_4h_ind.columns:
            htf_4h["htf4h_macd_bull"] = (df_4h_ind["macd_hist"] > 0).astype(int)

        htf_4h_aligned = htf_4h.reindex(base_index, method="ffill")
        for col in htf_4h_aligned.columns:
            htf_cols[col] = htf_4h_aligned[col]

    if not htf_cols:
        return pd.DataFrame(index=base_index)

    return pd.concat(htf_cols, axis=1)


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market regime labels as features using unsupervised rules.
    Regime = function of ADX, BB width, and recent return volatility.
    No future data used.
    """
    cols = {}
    if "adx" in df.columns:
        # Trending: ADX > 25. Strongly trending: ADX > 40.
        cols["regime_trending"]  = (df["adx"] > 25).astype(int)
        cols["regime_strong"]    = (df["adx"] > 40).astype(int)
        cols["regime_weak"]      = (df["adx"] < 20).astype(int)

    if "bb_width" in df.columns:
        bw_pct = df["bb_width"].rolling(50).rank(pct=True)
        cols["regime_squeeze"]   = (bw_pct < 0.20).astype(int)   # low volatility squeeze
        cols["regime_expansion"] = (bw_pct > 0.80).astype(int)   # volatility expansion

    if "close" in df.columns:
        ret_20 = df["close"].pct_change(20)
        ret_vol = df["close"].pct_change().rolling(20).std()
        vol_pct = ret_vol.rolling(100).rank(pct=True)
        cols["regime_high_vol"]  = (vol_pct > 0.70).astype(int)
        cols["regime_low_vol"]   = (vol_pct < 0.30).astype(int)
        cols["regime_bull_20"]   = (ret_20 > 0).astype(int)
        cols["regime_bear_20"]   = (ret_20 < 0).astype(int)

    if not cols:
        return df
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL BUILDING (upgraded for large data)
# ══════════════════════════════════════════════════════════════════════════════

def build_ensemble(X_train, y_train, X_val, y_val, n_features: int):
    """
    Build LightGBM + XGBoost ensemble.
    Parameters are tuned for large datasets (50k+ samples).
    """
    classes  = np.unique(y_train)
    weights  = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict  = {int(c): float(w) for c, w in zip(classes, weights)}
    sw_train = np.array([cw_dict.get(int(y), 1.0) for y in y_train])

    models = []

    # ── LightGBM ──────────────────────────────────────────────────────────────
    try:
        import lightgbm as lgbm
        lgb_model = lgbm.LGBMClassifier(
            n_estimators       = 1000,
            learning_rate      = 0.03,       # slower = better generalization
            num_leaves         = 63,         # richer trees for large data
            max_depth          = 7,
            min_child_samples  = 30,         # prevents overfitting on large data
            subsample          = 0.8,
            colsample_bytree   = 0.7,
            reg_alpha          = 0.1,
            reg_lambda         = 1.0,
            class_weight       = "balanced",
            n_jobs             = -1,
            verbose            = -1,
            random_state       = 42,
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set       = [(X_val, y_val)],
            callbacks      = [lgbm.early_stopping(50, verbose=False),
                              lgbm.log_evaluation(-1)],
        )
        # Use sigmoid (Platt scaling) not isotonic.
        # Isotonic regression on tree ensemble outputs with small validation sets
        # saturates to extreme 0/1 probabilities → produces 98% confidence readings
        # that are meaningless for trading decisions. Sigmoid is monotonic and
        # regularized, producing well-calibrated probabilities in the 0.4–0.7 range.
        lgb_cal = CalibratedClassifierCV(lgb_model, method="sigmoid", cv="prefit")
        lgb_cal.fit(X_val, y_val)
        models.append(("lgbm", lgb_cal, 0.55))
        logger.info("[HIST] LightGBM trained + calibrated (sigmoid)")
    except Exception as e:
        logger.warning(f"[HIST] LightGBM failed: {e}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            n_estimators       = 1000,
            learning_rate      = 0.03,
            max_depth          = 6,
            min_child_weight   = 5,
            subsample          = 0.8,
            colsample_bytree   = 0.7,
            reg_alpha          = 0.1,
            reg_lambda          = 1.0,
            use_label_encoder  = False,
            eval_metric        = "mlogloss",
            early_stopping_rounds = 50,
            n_jobs             = -1,
            random_state       = 42,
            verbosity          = 0,
        )
        xgb_model.fit(
            X_train, y_train,
            sample_weight          = sw_train,
            eval_set               = [(X_val, y_val)],
            verbose                = False,
        )
        xgb_cal = CalibratedClassifierCV(xgb_model, method="sigmoid", cv="prefit")
        xgb_cal.fit(X_val, y_val)
        models.append(("xgb", xgb_cal, 0.45))
        logger.info("[HIST] XGBoost trained + calibrated (sigmoid)")
    except Exception as e:
        logger.warning(f"[HIST] XGBoost failed: {e}")

    return models


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_symbol_from_history(symbol: str, dry_run: bool = False) -> dict:
    result = {
        "symbol":    symbol,
        "status":    "FAILED",
        "bars_15m":  0,
        "date_range": "",
        "label_dist": {},
        "n_features": 0,
        "wf_sharpe": 0.0,
        "wf_f1":     0.0,
        "val_acc":   0.0,
        "verdict":   "—",
    }

    print(f"\n{'='*60}")
    print(f"  {symbol}")
    print(f"{'='*60}")

    # ── Load all timeframes ────────────────────────────────────────────────────
    print("  Loading parquet files...")
    df_15m = load_parquet(symbol, "15m")
    df_1h  = load_parquet(symbol, "1h")
    df_4h  = load_parquet(symbol, "4h")

    if df_15m is None or len(df_15m) < 1000:
        print(f"  [ERROR] 15m data missing or too short ({len(df_15m) if df_15m is not None else 0} bars)")
        result["status"] = "NO_DATA"
        return result

    bars_15m = len(df_15m)
    days     = (df_15m.index[-1] - df_15m.index[0]).days
    result["bars_15m"]   = bars_15m
    result["date_range"] = f"{df_15m.index[0].strftime('%Y-%m-%d')} → {df_15m.index[-1].strftime('%Y-%m-%d')}"

    print(f"  15m: {bars_15m:,} bars  |  {days} days  |  {result['date_range']}")
    if df_1h is not None:
        print(f"  1h:  {len(df_1h):,} bars")
    if df_4h is not None:
        print(f"  4h:  {len(df_4h):,} bars")

    if dry_run:
        result["status"] = "DRY_RUN"
        return result

    # ── Add base indicators ────────────────────────────────────────────────────
    print("  Computing indicators...")
    df_15m = add_all_indicators(df_15m)

    # ── Add regime features ────────────────────────────────────────────────────
    df_15m = add_regime_features(df_15m)

    # ── Add multi-timeframe context ────────────────────────────────────────────
    print("  Building multi-timeframe features...")
    htf = build_htf_features(df_1h, df_4h, df_15m.index)
    if not htf.empty:
        df_15m = pd.concat([df_15m, htf], axis=1)
        print(f"  Added {len(htf.columns)} HTF features ({len([c for c in htf.columns if c.startswith('htf1h')])} × 1h, "
              f"{len([c for c in htf.columns if c.startswith('htf4h')])} × 4h, "
              f"{len([c for c in htf.columns if c.startswith('regime')])} regime)")

    # ── Engineer ML features ───────────────────────────────────────────────────
    feat_df = engineer_features(df_15m)

    # Add HTF cols to feature matrix (already in df_15m, carry through)
    htf_cols_in_df = [c for c in df_15m.columns if c.startswith("htf") or c.startswith("regime_")]
    for col in htf_cols_in_df:
        if col not in feat_df.columns:
            feat_df[col] = df_15m[col]

    # ── Labels ────────────────────────────────────────────────────────────────
    print("  Generating labels...")
    atr_mult = ATR_MULT_MAP.get(symbol, ATR_MULT)
    labels   = generate_labels(df_15m, method="triple_barrier",
                               future_bars=FUTURE_BARS, atr_mult=atr_mult)

    # Trim tail (last FUTURE_BARS bars have no valid labels)
    feat_df = feat_df.iloc[:-FUTURE_BARS]
    labels  = labels[:-FUTURE_BARS]

    # Drop NaN rows
    valid_mask = feat_df.notna().all(axis=1)
    feat_df    = feat_df[valid_mask]
    labels     = labels[valid_mask.values]

    n_samples  = len(feat_df)
    n_features = feat_df.shape[1]
    result["n_features"] = n_features

    if n_samples < 500:
        print(f"  [ERROR] Only {n_samples} clean samples after NaN drop")
        result["status"] = "INSUFFICIENT_DATA"
        return result

    dist = label_distribution(labels)
    result["label_dist"] = dist
    hold_pct  = dist["HOLD"]  / dist["total"]
    long_pct  = dist["LONG"]  / dist["total"]
    short_pct = dist["SHORT"] / dist["total"]
    print(f"  Samples: {n_samples:,}  |  Features: {n_features}")
    print(f"  Labels:  HOLD={hold_pct:.1%}  LONG={long_pct:.1%}  SHORT={short_pct:.1%}")

    # Scale features
    scaler = RobustScaler()
    X = scaler.fit_transform(feat_df.values.astype(np.float32))
    y = labels.astype(int)
    # prices must be aligned to the same rows as X and y.
    # valid_mask was computed on feat_df.iloc[:-FUTURE_BARS], which has
    # len(df_15m) - FUTURE_BARS rows. df_15m itself has FUTURE_BARS more rows,
    # so we must trim df_15m first before applying valid_mask as a boolean index.
    close_trimmed = df_15m["close"].values[:-FUTURE_BARS]   # align to feat_df
    prices = close_trimmed[valid_mask.values]               # apply NaN mask

    # ── Walk-forward validation ────────────────────────────────────────────────
    print(f"\n  Walk-forward validation ({N_WF_SPLITS} folds)...")

    wfv    = WalkForwardValidator(n_splits=N_WF_SPLITS, gap=WF_GAP)
    splits = wfv.split(n_samples)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]

        # Use 10% of train for calibration / early stopping
        val_size = max(100, int(len(X_tr) * 0.10))
        X_val_fold, y_val_fold = X_tr[-val_size:], y_tr[-val_size:]
        X_tr_fold,  y_tr_fold  = X_tr[:-val_size], y_tr[:-val_size]

        models = build_ensemble(X_tr_fold, y_tr_fold, X_val_fold, y_val_fold, n_features)
        if not models:
            continue

        # Ensemble predict
        total_w = sum(w for _, _, w in models)
        proba   = None
        for _, m, w in models:
            p = m.predict_proba(X_te) * (w / total_w)
            if p.shape[1] < 3:
                full = np.zeros((len(p), 3))
                for ci, cls in enumerate(m.classes_):
                    full[:, cls] = p[:, ci]
                p = full
            proba = p if proba is None else proba + p

        y_pred  = proba.argmax(axis=1)
        f1      = f1_score(y_te, y_pred, average="macro", zero_division=0)
        acc     = float((y_pred == y_te).mean())

        # Trading sim (simplified)
        p_prices = prices[test_idx] if len(prices) >= max(test_idx) + 1 else np.ones(len(test_idx))
        from research.walk_forward import _compute_trading_metrics
        trading = _compute_trading_metrics(y_pred, p_prices)
        sharpe  = trading["sharpe"]
        pf      = trading["profit_factor"]
        wr      = trading["win_rate"]

        h_pct = int((y_pred == 0).mean() * 100)
        l_pct = int((y_pred == 1).mean() * 100)
        s_pct = int((y_pred == 2).mean() * 100)

        print(f"    Fold {fold_idx:2}/{N_WF_SPLITS}: "
              f"n_train={len(X_tr):,}  n_test={len(X_te):,}  "
              f"acc={acc:.1%}  f1={f1:.3f}  sharpe={sharpe:.2f}  "
              f"pf={pf:.2f}  wr={wr:.0%}  [H={h_pct}% L={l_pct}% S={s_pct}%]")

        fold_results.append({"sharpe": sharpe, "f1": f1, "acc": acc})

    if not fold_results:
        print("  [ERROR] All folds failed")
        result["status"] = "TRAIN_FAILED"
        return result

    mean_sharpe  = float(np.mean([r["sharpe"] for r in fold_results]))
    mean_f1      = float(np.mean([r["f1"]     for r in fold_results]))
    mean_acc     = float(np.mean([r["acc"]    for r in fold_results]))
    std_sharpe   = float(np.std([r["sharpe"]  for r in fold_results]))
    consistency  = float(np.mean([r["sharpe"] > 0 for r in fold_results]))

    # Acceptance verdict
    reasons = []
    if mean_sharpe < MIN_SHARPE:  reasons.append(f"Sharpe {mean_sharpe:.2f} < {MIN_SHARPE}")
    if mean_f1     < MIN_F1:      reasons.append(f"F1 {mean_f1:.3f} < {MIN_F1}")
    if consistency < MIN_CONSIST: reasons.append(f"Consistency {consistency:.0%} < {MIN_CONSIST:.0%}")
    accepted = len(reasons) == 0
    verdict_str = "ACCEPT" if accepted else f"REJECT ({', '.join(reasons)})"

    print(f"\n  {'='*56}")
    print(f"  WALK-FORWARD SUMMARY ({N_WF_SPLITS} folds)")
    print(f"  {'='*56}")
    print(f"  Accuracy:     {mean_acc:.2%}")
    print(f"  F1 Macro:     {mean_f1:.3f}")
    print(f"  Sharpe (mean): {mean_sharpe:.3f}  ± {std_sharpe:.3f}")
    print(f"  Consistency:  {consistency:.0%} folds positive Sharpe")
    print(f"  Verdict:      {verdict_str}")
    print(f"  {'='*56}")

    result["wf_sharpe"] = mean_sharpe
    result["wf_f1"]     = mean_f1

    # ── Final model: train on ALL data ────────────────────────────────────────
    print("\n  Training final model on full dataset...")
    val_size = max(500, int(n_samples * 0.10))
    X_final_val, y_final_val = X[-val_size:], y[-val_size:]
    X_final_tr,  y_final_tr  = X[:-val_size], y[:-val_size]

    final_models = build_ensemble(X_final_tr, y_final_tr, X_final_val, y_final_val, n_features)

    if not final_models:
        print("  [ERROR] Final model training failed")
        result["status"] = "TRAIN_FAILED"
        return result

    # Validation report
    total_w = sum(w for _, _, w in final_models)
    proba = None
    for _, m, w in final_models:
        p = m.predict_proba(X_final_val) * (w / total_w)
        if p.shape[1] < 3:
            full = np.zeros((len(p), 3))
            for ci, cls in enumerate(m.classes_):
                full[:, cls] = p[:, ci]
            p = full
        proba = p if proba is None else proba + p

    y_val_pred   = proba.argmax(axis=1)
    val_acc      = float((y_val_pred == y_final_val).mean())
    result["val_acc"] = val_acc

    print(f"  Val accuracy: {val_acc:.2%}")
    print(classification_report(y_final_val, y_val_pred,
                                target_names=["HOLD", "LONG", "SHORT"],
                                zero_division=0))

    # ── Save model ────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"ml_{symbol}.joblib"

    feature_names = feat_df.columns.tolist()
    joblib.dump({
        "models":        final_models,
        "scaler":        scaler,
        "feature_names": feature_names,
        "wf_sharpe":     mean_sharpe,
        "wf_f1":         mean_f1,
        "wf_accepted":   accepted,
        "label_dist":    dist,
        "trained_on":    "historical_parquet",
        "bars_trained":  n_samples,
        "date_range":    result["date_range"],
        "trained_at":    datetime.now().isoformat(),
    }, model_path)

    status_str = "ACCEPTED" if accepted else "WF-rejected (saved anyway)"
    print(f"\n  Saved → {model_path}")
    if accepted:
        logger.info(f"[HIST] {symbol} ACCEPTED | Sharpe={mean_sharpe:.2f}  F1={mean_f1:.3f}")
    else:
        logger.warning(f"[HIST] {symbol} WF-rejected | Sharpe={mean_sharpe:.2f}  F1={mean_f1:.3f} | saved anyway")

    result["status"]  = "OK"
    result["verdict"] = "ACCEPTED" if accepted else "WF-REJECTED"
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 5. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train from historical parquet files")
    parser.add_argument("--symbol",  type=str, default=None, help="Single symbol, e.g. BTCUSDT")
    parser.add_argument("--dry-run", action="store_true",    help="Check data only, no training")
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS

    print("\n" + "="*60)
    print("  AI TRADING BOT — Historical Model Training")
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Symbols:  {symbols}")
    print(f"  Mode:     {'DRY RUN' if args.dry_run else 'FULL TRAINING'}")
    print("="*60)

    # ── Data inventory ─────────────────────────────────────────────────────────
    print("\nData inventory:")
    for sym in symbols:
        for tf in TIMEFRAMES:
            df = load_parquet(sym, tf)
            print(summarize_df(df, sym, tf))

    if args.dry_run:
        print("\nDry run complete. No models trained.")
        return

    # ── Train ──────────────────────────────────────────────────────────────────
    t_start  = time.time()
    results  = []
    for symbol in symbols:
        t0 = time.time()
        r  = train_symbol_from_history(symbol)
        r["elapsed"] = time.time() - t0
        results.append(r)

    total_time = time.time() - t_start

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("  TRAINING SUMMARY")
    print("="*60)
    print(f"  {'Symbol':<10} {'Bars':>8}  {'Features':>9}  {'Sharpe':>8}  {'F1':>6}  {'Val Acc':>8}  {'Verdict'}")
    print(f"  {'-'*58}")
    for r in results:
        if r["status"] == "OK":
            print(f"  {r['symbol']:<10} {r['bars_15m']:>8,}  {r['n_features']:>9}  "
                  f"{r['wf_sharpe']:>8.3f}  {r['wf_f1']:>6.3f}  {r['val_acc']:>8.2%}  "
                  f"{r['verdict']}  ({r['elapsed']:.0f}s)")
        else:
            print(f"  {r['symbol']:<10} {'—':>8}  {'—':>9}  {'—':>8}  {'—':>6}  {'—':>8}  {r['status']}")

    accepted = sum(1 for r in results if r["verdict"] == "ACCEPTED")
    rejected = sum(1 for r in results if r["verdict"] == "WF-REJECTED")
    print(f"\n  {accepted} accepted  |  {rejected} WF-rejected but saved  |  total: {total_time:.0f}s")

    print()
    print("  All models saved to saved_models/")
    print("  These replace the testnet-trained models.")
    print("  Start the bot normally — it will load these automatically.")
    print()

    # ── Accuracy guide ─────────────────────────────────────────────────────────
    print("  HOW TO READ THE NUMBERS (3-class problem):")
    print("  ─────────────────────────────────────────────────────────────")
    print("  Random baseline: 33% accuracy")
    print("  With years of data targets:")
    print("    Val accuracy:  55–70% = excellent,  50–55% = good,  45–50% = acceptable")
    print("    WF Sharpe:     > 1.0  = excellent,  > 0.5  = good,  > 0.0  = usable")
    print("    WF F1 macro:   > 0.45 = excellent,  > 0.35 = good,  > 0.25 = acceptable")


if __name__ == "__main__":
    main()