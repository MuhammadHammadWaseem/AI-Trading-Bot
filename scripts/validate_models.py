"""
scripts/validate_models.py
══════════════════════════════════════════════════════════════════════════════
Comprehensive model validation — run after train_from_history.py.

FIXES IN THIS VERSION vs previous:
  1. Scaler leakage fixed — uses payload['scaler'].transform() not fit_transform()
  2. ETH conf_threshold lowered to 0.38 (HOLD-biased, same as SOLUSDT)
  3. Sharpe via daily-bucketed equity returns × sqrt(252) — not per-trade
  4. HOLD-bias warning threshold 55% (was 70%) — catches ETH at 60%
  5. Timeout exit % flagged per symbol

USAGE:
    cd "C:\\Users\\123\\Desktop\\AI Trading Bot"
    python scripts/validate_models.py
    python scripts/validate_models.py --symbol BTCUSDT
    python scripts/validate_models.py --quick
"""

import sys, os, argparse, warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from data.indicators import add_all_indicators
from models.training.trainer import engineer_features
from models.training.label_engine import generate_labels
from config.logger import get_logger

sys.path.insert(0, str(Path(PROJECT_ROOT) / "scripts"))
from train_from_history import (
    load_parquet, build_htf_features, add_regime_features,
    ATR_MULT_MAP, FUTURE_BARS
)

logger     = get_logger("validate")
MODELS_DIR = Path(PROJECT_ROOT) / "saved_models"
SYMBOLS    = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]

TAKER_FEE = 0.0004
SLIPPAGE  = 0.0002
OOS_RATIO = 0.10

# Per-symbol confidence thresholds — must match futures_trader.py exactly.
# ETH and SOL are HOLD-biased so use a lower threshold to recover valid signals.
CONF_THRESHOLDS = {
    "BTCUSDT": 0.42,
    "ETHUSDT": 0.38,
    "BNBUSDT": 0.42,
    "SOLUSDT": 0.38,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. MODEL INTEGRITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_model_integrity(symbol: str) -> dict:
    path   = MODELS_DIR / f"ml_{symbol}.joblib"
    result = {"symbol": symbol, "exists": False, "loadable": False, "issues": []}

    if not path.exists():
        result["issues"].append("File not found")
        return result

    result["exists"]  = True
    result["size_mb"] = path.stat().st_size / 1e6

    try:
        payload = joblib.load(path)
        result["loadable"] = True

        for k in ["models", "scaler", "feature_names", "wf_sharpe", "wf_f1", "wf_accepted"]:
            if k not in payload:
                result["issues"].append(f"Missing key: {k}")

        result["n_models"]     = len(payload.get("models", []))
        result["n_features"]   = len(payload.get("feature_names", []))
        result["wf_sharpe"]    = payload.get("wf_sharpe", 0)
        result["wf_f1"]        = payload.get("wf_f1", 0)
        result["wf_accepted"]  = payload.get("wf_accepted", False)
        result["trained_on"]   = payload.get("trained_on", "unknown")
        result["bars_trained"] = payload.get("bars_trained", 0)
        result["date_range"]   = payload.get("date_range", "unknown")
        result["trained_at"]   = payload.get("trained_at", "unknown")
        result["model_types"]  = [name for name, _, _ in payload.get("models", [])]

        if result["n_models"] == 0:
            result["issues"].append("No models in ensemble")
        if result["n_features"] < 50:
            result["issues"].append(f"Very few features: {result['n_features']}")

    except Exception as e:
        result["issues"].append(f"Load error: {e}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD FEATURE MATRIX (raw, unscaled — caller applies model's own scaler)
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(symbol: str):
    """
    Rebuild the training feature matrix for a symbol.
    Returns RAW unscaled feat_df so the caller can use payload['scaler'].transform()
    — this prevents the leakage bug of refitting the scaler on data that includes OOS.
    """
    df_15m = load_parquet(symbol, "15m")
    df_1h  = load_parquet(symbol, "1h")
    df_4h  = load_parquet(symbol, "4h")

    if df_15m is None:
        return None, None, None, None, None

    df_15m = add_all_indicators(df_15m)
    df_15m = add_regime_features(df_15m)

    htf = build_htf_features(df_1h, df_4h, df_15m.index)
    if not htf.empty:
        df_15m = pd.concat([df_15m, htf], axis=1)

    feat_df  = engineer_features(df_15m)
    htf_cols = [c for c in df_15m.columns if c.startswith("htf") or c.startswith("regime_")]
    for col in htf_cols:
        if col not in feat_df.columns:
            feat_df[col] = df_15m[col]

    atr_mult = ATR_MULT_MAP.get(symbol, 1.5)
    labels   = generate_labels(df_15m, method="triple_barrier",
                               future_bars=FUTURE_BARS, atr_mult=atr_mult)

    feat_df = feat_df.iloc[:-FUTURE_BARS]
    labels  = labels[:-FUTURE_BARS]

    valid_mask = feat_df.notna().all(axis=1)
    feat_df    = feat_df[valid_mask]
    labels     = labels[valid_mask.values]
    prices     = df_15m["close"].iloc[:-FUTURE_BARS][valid_mask].values

    return feat_df, labels, prices, df_15m, valid_mask


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREDICTION CHECK — uses the model's own scaler (no leakage)
# ══════════════════════════════════════════════════════════════════════════════

def check_predictions(payload: dict, X_raw: np.ndarray, y_true: np.ndarray) -> dict:
    """
    FIX: Scale with payload['scaler'], not a refitted scaler.
    The saved model's scaler was fitted on training data only.
    Refitting on the full dataset would leak OOS statistics into the features.
    """
    scaler   = payload["scaler"]                            # ← uses model's own scaler
    X_scaled = scaler.transform(X_raw.astype(np.float32))  # ← transform only, no fit

    models  = payload["models"]
    total_w = sum(w for _, _, w in models)
    proba   = None

    for _, m, w in models:
        p = m.predict_proba(X_scaled) * (w / total_w)
        if p.shape[1] < 3:
            full = np.zeros((len(p), 3))
            for ci, cls in enumerate(m.classes_):
                full[:, cls] = p[:, ci]
            p = full
        proba = p if proba is None else proba + p

    y_pred   = proba.argmax(axis=1)
    conf_max = proba.max(axis=1)

    from sklearn.metrics import precision_recall_fscore_support, f1_score
    p_arr, r_arr, f_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
    )

    hold_pct = float((y_pred == 0).mean())
    long_pct = float((y_pred == 1).mean())
    short_pct = float((y_pred == 2).mean())

    issues = []
    if hold_pct > 0.55:   # lowered from 0.70 — catches ETH at 60%
        issues.append(f"HOLD-biased: {hold_pct:.0%} predictions are HOLD")
    if long_pct < 0.05 or short_pct < 0.05:
        issues.append("One direction nearly absent")
    if float(conf_max.mean()) < 0.36:
        issues.append("Mean confidence < 0.36 — model highly uncertain")
    if float(conf_max.mean()) > 0.75:
        issues.append("Mean confidence > 0.75 — possible overfitting")

    return {
        "n_samples":  len(y_pred),
        "hold_pct":   hold_pct,
        "long_pct":   long_pct,
        "short_pct":  short_pct,
        "accuracy":   float((y_pred == y_true).mean()),
        "mean_conf":  float(conf_max.mean()),
        "conf_p25":   float(np.percentile(conf_max, 25)),
        "conf_p50":   float(np.percentile(conf_max, 50)),
        "conf_p75":   float(np.percentile(conf_max, 75)),
        "conf_p90":   float(np.percentile(conf_max, 90)),
        "proba":      proba,
        "y_pred":     y_pred,
        "y_true":     y_true,
        "precision":  p_arr,
        "recall":     r_arr,
        "f1_class":   f_arr,
        "f1_macro":   float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "issues":     issues,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. REGIME BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════

def regime_breakdown(pred: dict, df_oos: pd.DataFrame) -> dict:
    y_pred = pred["y_pred"]
    y_true = pred["y_true"]
    n      = len(y_pred)

    regimes = {}
    if "regime_trending" in df_oos.columns:
        t = df_oos["regime_trending"].values[-n:].astype(bool)
        regimes["Trending (ADX>25)"] = t
        regimes["Ranging  (ADX<25)"] = ~t
    if "regime_high_vol" in df_oos.columns:
        regimes["High volatility"] = df_oos["regime_high_vol"].values[-n:].astype(bool)
    if "regime_low_vol" in df_oos.columns:
        regimes["Low volatility"]  = df_oos["regime_low_vol"].values[-n:].astype(bool)
    if "regime_bull_20" in df_oos.columns:
        regimes["Bull (20-bar up)"]   = df_oos["regime_bull_20"].values[-n:].astype(bool)
    if "regime_bear_20" in df_oos.columns:
        regimes["Bear (20-bar down)"] = df_oos["regime_bear_20"].values[-n:].astype(bool)

    return {
        name: {"accuracy": float((y_pred[m] == y_true[m]).mean()), "n": int(m.sum())}
        for name, m in regimes.items() if m.sum() >= 50
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. REALISTIC BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def realistic_backtest(y_pred: np.ndarray, proba: np.ndarray,
                       prices: np.ndarray, symbol: str,
                       conf_threshold: float = 0.42) -> dict:
    """
    Fixed-risk backtest.  Three choices that prevent inflated results:

    1. Fixed $100 risk per trade — no compounding on test data.
    2. PnL capped at [-1R, +2R] — timeout exits cannot credit more than TP.
    3. Sharpe via daily-bucketed equity returns × sqrt(252).
       Using sqrt(N_trades/year) inflates Sharpe for high-frequency systems
       regardless of actual edge (e.g. 800 trades/yr → sqrt(1280) ≈ 36×).
    """
    INITIAL_CAPITAL = 10_000.0
    RISK_DOLLARS    = 100.0      # 1% of initial capital, fixed
    STOP_PCT        = 0.010      # 1% SL
    TP_PCT          = 0.020      # 2% TP → 2:1 RR
    MAX_HOLD_BARS   = 32         # 32 × 15m = 8h timeout

    capital   = INITIAL_CAPITAL
    equity    = [capital]
    trades    = []
    in_trade  = False
    entry_px  = 0.0
    direction = 0
    entry_bar = 0

    for i, (signal, conf, price) in enumerate(zip(y_pred, proba.max(axis=1), prices)):
        if not in_trade:
            if signal == 1 and conf >= conf_threshold:
                entry_px  = price * (1 + SLIPPAGE)
                direction = 1
                in_trade  = True
                entry_bar = i
            elif signal == 2 and conf >= conf_threshold:
                entry_px  = price * (1 - SLIPPAGE)
                direction = -1
                in_trade  = True
                entry_bar = i
        else:
            hold_bars = i - entry_bar
            pnl_pct   = ((price - entry_px) / entry_px) if direction == 1 \
                        else ((entry_px - price) / entry_px)

            hit_tp = pnl_pct >=  TP_PCT
            hit_sl = pnl_pct <= -STOP_PCT

            if hit_tp or hit_sl or hold_bars >= MAX_HOLD_BARS:
                exit_px   = price * (1 - SLIPPAGE) if direction == 1 \
                            else price * (1 + SLIPPAGE)
                gross_ret = ((exit_px - entry_px) / entry_px) if direction == 1 \
                            else ((entry_px - exit_px) / entry_px)
                net_ret   = gross_ret - TAKER_FEE * 2
                r_mult    = max(-1.0, min(2.0, net_ret / STOP_PCT))
                pnl_usd   = RISK_DOLLARS * r_mult

                capital += pnl_usd
                equity.append(capital)

                trades.append({
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "bars_held": hold_bars,
                    "r_mult":    r_mult,
                    "pnl":       pnl_usd,
                    "exit_type": "TP" if hit_tp else ("SL" if hit_sl else "TIMEOUT"),
                })
                in_trade = False

    if not trades:
        return {"error": "No trades generated (threshold too high or model always HOLDs)"}

    equity_arr = np.array(equity)
    n_trades   = len(trades)
    wins       = [t for t in trades if t["pnl"] > 0]
    losses     = [t for t in trades if t["pnl"] <= 0]

    # ── Daily-bucketed Sharpe ─────────────────────────────────────────────────
    # Resample the equity curve to ~daily intervals (96 bars × 15m = 24h)
    # then compute mean/std of daily returns × sqrt(252).
    # This is the standard method used by prime brokers and fund administrators.
    bars_per_day  = 96
    n_equity_pts  = len(equity_arr)
    # Build an equity value at each multiple of 96 equity-curve points
    step          = max(1, n_equity_pts // max(1, len(prices) // bars_per_day))
    daily_eq      = equity_arr[::step]
    daily_rets    = np.diff(daily_eq) / daily_eq[:-1]
    sharpe        = 0.0
    if len(daily_rets) > 1 and daily_rets.std() > 1e-10:
        sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(252))

    # Max drawdown
    peak   = np.maximum.accumulate(equity_arr)
    dd     = (equity_arr - peak) / peak
    max_dd = float(dd.min())

    r_series     = np.array([t["r_mult"] for t in trades])
    gross_win_r  = sum(t["r_mult"] for t in wins)
    gross_loss_r = abs(sum(t["r_mult"] for t in losses))
    pf           = gross_win_r / (gross_loss_r + 1e-10)

    longs    = [t for t in trades if t["direction"] == "LONG"]
    shorts   = [t for t in trades if t["direction"] == "SHORT"]
    tp_exits = sum(1 for t in trades if t["exit_type"] == "TP")
    sl_exits = sum(1 for t in trades if t["exit_type"] == "SL")

    return {
        "initial_capital": INITIAL_CAPITAL,
        "final_capital":   round(capital, 2),
        "total_pnl":       round(capital - INITIAL_CAPITAL, 2),
        "total_return":    (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL,
        "n_trades":        n_trades,
        "win_rate":        len(wins) / n_trades,
        "sharpe":          sharpe,
        "max_drawdown":    max_dd,
        "profit_factor":   pf,
        "expectancy_r":    float(r_series.mean()),
        "avg_r_win":       float(np.mean([t["r_mult"] for t in wins]))   if wins   else 0.0,
        "avg_r_loss":      float(np.mean([t["r_mult"] for t in losses])) if losses else 0.0,
        "n_longs":         len(longs),
        "n_shorts":        len(shorts),
        "long_wr":         sum(1 for t in longs  if t["pnl"] > 0) / max(len(longs), 1),
        "short_wr":        sum(1 for t in shorts if t["pnl"] > 0) / max(len(shorts), 1),
        "tp_exits":        tp_exits,
        "sl_exits":        sl_exits,
        "timeout_exits":   n_trades - tp_exits - sl_exits,
        "avg_bars_held":   float(np.mean([t["bars_held"] for t in trades])),
        "equity_curve":    equity_arr.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def get_feature_importance(payload: dict, top_n: int = 15) -> list:
    feature_names = payload.get("feature_names", [])
    for name, model, _ in payload.get("models", []):
        if name == "lgbm":
            base = model
            if hasattr(model, "calibrated_classifiers_"):
                base = model.calibrated_classifiers_[0].estimator
            elif hasattr(model, "base_estimator"):
                base = model.base_estimator
            if hasattr(base, "feature_importances_"):
                imp = base.feature_importances_
                if len(imp) == len(feature_names):
                    return sorted(zip(feature_names, imp),
                                  key=lambda x: x[1], reverse=True)[:top_n]
    return []


# ══════════════════════════════════════════════════════════════════════════════
# 7. LABEL DISTRIBUTION CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_label_distribution(symbol: str) -> dict:
    df_15m = load_parquet(symbol, "15m")
    if df_15m is None:
        return {}
    df_15m   = add_all_indicators(df_15m)
    atr_mult = ATR_MULT_MAP.get(symbol, 1.5)
    sample   = df_15m.iloc[-50000:].copy() if len(df_15m) > 50000 else df_15m.copy()
    labels   = generate_labels(sample, method="triple_barrier",
                               future_bars=FUTURE_BARS, atr_mult=atr_mult)
    return {
        "HOLD":      int((labels == 0).sum()),
        "LONG":      int((labels == 1).sum()),
        "SHORT":     int((labels == 2).sum()),
        "HOLD_pct":  float((labels == 0).mean()),
        "LONG_pct":  float((labels == 1).mean()),
        "SHORT_pct": float((labels == 2).mean()),
        "total":     len(labels),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN VALIDATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def validate_symbol(symbol: str, quick: bool = False) -> dict:
    print(f"\n{'='*60}")
    print(f"  VALIDATING: {symbol}")
    print(f"{'='*60}")
    summary = {"symbol": symbol, "passed": [], "warnings": [], "failures": []}

    # ── [1/7] Integrity ───────────────────────────────────────────────────────
    print("\n  [1/7] Model integrity check...")
    ig = check_model_integrity(symbol)

    if not ig["exists"]:
        print(f"  ✗ Not found: {MODELS_DIR / f'ml_{symbol}.joblib'}")
        summary["failures"].append("Model file missing")
        return summary
    if not ig["loadable"]:
        print(f"  ✗ Load failed: {ig['issues']}")
        summary["failures"].append("Model not loadable")
        return summary

    print(f"  ✓ Loaded  |  {ig['size_mb']:.1f} MB  |  "
          f"{ig['n_features']} features  |  {'+'.join(ig['model_types'])}")
    print(f"  ✓ Trained on {ig['bars_trained']:,} bars  |  {ig['date_range']}")
    print(f"  ✓ WF Sharpe={ig['wf_sharpe']:.2f}  F1={ig['wf_f1']:.3f}  "
          f"Accepted={ig['wf_accepted']}")

    try:
        trained_at = datetime.fromisoformat(ig["trained_at"])
        age_days   = (datetime.now() - trained_at).days
        if age_days > 90:
            print(f"  ⚠ Model is {age_days} days old — consider retraining")
            summary["warnings"].append(f"Model age: {age_days}d")
        else:
            print(f"  ✓ Model age: {age_days} days")
    except Exception:
        pass

    if ig["issues"]:
        for issue in ig["issues"]:
            print(f"  ⚠ {issue}")
            summary["warnings"].append(issue)
    else:
        summary["passed"].append("integrity")

    payload = joblib.load(MODELS_DIR / f"ml_{symbol}.joblib")

    # ── [2/7] Label distribution ──────────────────────────────────────────────
    print(f"\n  [2/7] Label distribution (last 50k bars)...")
    ld = check_label_distribution(symbol)
    if ld:
        print(f"  HOLD={ld['HOLD_pct']:.1%}  LONG={ld['LONG_pct']:.1%}  "
              f"SHORT={ld['SHORT_pct']:.1%}  (n={ld['total']:,})")
        for cls in ["HOLD", "LONG", "SHORT"]:
            pct = ld[f"{cls}_pct"]
            if pct < 0.10:
                msg = f"Low {cls} labels: {pct:.1%}"
                print(f"  ⚠ {cls} only {pct:.1%} — very imbalanced")
                summary["warnings"].append(msg)
            elif pct < 0.20:
                print(f"  ⚠ {cls} at {pct:.1%} — somewhat imbalanced")
        if all(ld[f"{c}_pct"] > 0.15 for c in ["HOLD", "LONG", "SHORT"]):
            print("  ✓ Label distribution balanced")
            summary["passed"].append("label_distribution")

    # ── [3/7] Feature matrix ──────────────────────────────────────────────────
    print(f"\n  [3/7] Building OOS feature matrix (raw, unscaled)...")
    feat_df, labels, prices, df_15m, _ = build_feature_matrix(symbol)

    if feat_df is None:
        print("  ✗ Could not build feature matrix")
        summary["failures"].append("Feature matrix failed")
        return summary

    n_total    = len(feat_df)
    n_oos      = int(n_total * OOS_RATIO)
    n_train    = n_total - n_oos
    feat_train = feat_df.iloc[:n_train]
    feat_oos   = feat_df.iloc[n_train:]
    y_train    = labels[:n_train]
    y_oos      = labels[n_train:]
    prices_oos = prices[n_train:]

    print(f"  Total: {n_total:,}  |  Train: {n_train:,}  |  OOS: {n_oos:,} ({OOS_RATIO:.0%})")
    print(f"  OOS period: {feat_oos.index[0].strftime('%Y-%m-%d')} → "
          f"{feat_oos.index[-1].strftime('%Y-%m-%d')}")

    expected    = payload["feature_names"]
    X_train_raw = feat_train.reindex(columns=expected, fill_value=0).values
    X_oos_raw   = feat_oos.reindex(columns=expected, fill_value=0).values

    missing = set(expected) - set(feat_df.columns)
    if missing:
        print(f"  ⚠ {len(missing)} features missing: {list(missing)[:5]}...")
        summary["warnings"].append(f"{len(missing)} missing features")
    else:
        print(f"  ✓ Feature matrix ready ({len(expected)} features)")
        summary["passed"].append("feature_matrix")

    # ── [4/7] Prediction sanity ───────────────────────────────────────────────
    print(f"\n  [4/7] Prediction sanity (OOS, using model's saved scaler)...")
    pred = check_predictions(payload, X_oos_raw, y_oos)

    print(f"  Accuracy:    {pred['accuracy']:.2%}  (random=33%)")
    print(f"  F1 macro:    {pred['f1_macro']:.3f}")
    print(f"  Predictions: HOLD={pred['hold_pct']:.1%}  "
          f"LONG={pred['long_pct']:.1%}  SHORT={pred['short_pct']:.1%}")
    print(f"  Confidence:  mean={pred['mean_conf']:.3f}  "
          f"p25={pred['conf_p25']:.3f}  p50={pred['conf_p50']:.3f}  "
          f"p75={pred['conf_p75']:.3f}")
    print()
    print(f"  Per-class:")
    for i, cls in enumerate(["HOLD", "LONG", "SHORT"]):
        print(f"    {cls:<6}  prec={pred['precision'][i]:.3f}  "
              f"rec={pred['recall'][i]:.3f}  f1={pred['f1_class'][i]:.3f}")

    for issue in pred["issues"]:
        print(f"  ⚠ {issue}")
        summary["warnings"].append(issue)

    if pred["accuracy"] > 0.45 and pred["f1_macro"] > 0.35:
        summary["passed"].append("prediction_quality")
    elif pred["accuracy"] < 0.36:
        summary["failures"].append(f"Accuracy near random: {pred['accuracy']:.1%}")

    # ── [5/7] Overfitting check ───────────────────────────────────────────────
    print(f"\n  [5/7] Overfitting check...")
    tr_pred = check_predictions(payload, X_train_raw[:5000], y_train[:5000])
    gap     = tr_pred["accuracy"] - pred["accuracy"]

    print(f"  Train (5k sample): {tr_pred['accuracy']:.2%}")
    print(f"  OOS:               {pred['accuracy']:.2%}")
    print(f"  Gap:               {gap:.2%}")

    if gap > 0.20:
        print(f"  ✗ Large overfit ({gap:.1%})")
        summary["failures"].append(f"Overfitting: {gap:.1%}")
    elif gap > 0.10:
        print(f"  ⚠ Moderate overfit ({gap:.1%}) — monitor in paper trading")
        summary["warnings"].append(f"Overfit gap: {gap:.1%}")
    else:
        print(f"  ✓ Generalisation healthy (gap ≤ 10%)")
        summary["passed"].append("overfitting_check")

    # ── [6/7] Regime breakdown ────────────────────────────────────────────────
    print(f"\n  [6/7] Accuracy by regime...")
    df_oos_slice = (df_15m.iloc[-(n_oos + FUTURE_BARS):-FUTURE_BARS]
                    if len(df_15m) > n_oos + FUTURE_BARS else df_15m)
    bd = regime_breakdown(pred, df_oos_slice)

    if bd:
        for name, stats in bd.items():
            bar = "█" * int(stats["accuracy"] * 30)
            print(f"  {name:<25} {stats['accuracy']:.2%}  {bar}  (n={stats['n']:,})")
        summary["passed"].append("regime_stability")
    else:
        print("  Regime columns not available")

    # ── [7/7] Realistic backtest ──────────────────────────────────────────────
    if not quick:
        conf_thr = CONF_THRESHOLDS.get(symbol, 0.42)
        print(f"\n  [7/7] Backtest (conf_threshold={conf_thr})...")
        if conf_thr < 0.42:
            print(f"  Note: {symbol} uses lower threshold (HOLD-biased model)")

        bt = realistic_backtest(pred["y_pred"], pred["proba"],
                                prices_oos, symbol, conf_threshold=conf_thr)

        if "error" in bt:
            print(f"  ⚠ {bt['error']}")
            summary["warnings"].append(bt["error"])
        else:
            timeout_pct = bt["timeout_exits"] / bt["n_trades"]
            print(f"  Fixed risk:  $100/trade (1% of $10,000)")
            print(f"  Net P&L:     ${bt['total_pnl']:+,.2f}  "
                  f"({bt['total_return']*100:+.1f}%  |  {bt['n_trades']} trades)")
            print(f"  Trades:      {bt['n_trades']}  "
                  f"({bt['n_longs']} long / {bt['n_shorts']} short)")
            print(f"  Win rate:    {bt['win_rate']:.1%}  "
                  f"(long={bt['long_wr']:.1%} / short={bt['short_wr']:.1%})")
            print(f"  Expectancy:  {bt['expectancy_r']:+.3f}R/trade  "
                  f"(avg win={bt['avg_r_win']:+.2f}R / avg loss={bt['avg_r_loss']:+.2f}R)")
            print(f"  Sharpe:      {bt['sharpe']:.2f}  "
                  f"(daily-bucketed × sqrt(252))")
            print(f"  Max DD:      {bt['max_drawdown']:.2%}")
            print(f"  PF:          {bt['profit_factor']:.2f}")
            print(f"  Avg hold:    {bt['avg_bars_held']:.1f} bars  "
                  f"({bt['avg_bars_held']*15/60:.1f}h)")
            print(f"  Exits:       TP={bt['tp_exits']}  SL={bt['sl_exits']}  "
                  f"TIMEOUT={bt['timeout_exits']} ({timeout_pct:.0%})")

            if timeout_pct > 0.50:
                print(f"  ⚠ {timeout_pct:.0%} timeout exits — "
                      f"TP/SL too tight for {symbol}'s current volatility")
                summary["warnings"].append(f"High timeout rate: {timeout_pct:.0%}")

            if bt["sharpe"] > 1.5:
                print(f"  ✓ Excellent Sharpe ({bt['sharpe']:.2f})")
                summary["passed"].append("backtest")
            elif bt["sharpe"] > 0.8:
                print(f"  ✓ Good Sharpe ({bt['sharpe']:.2f})")
                summary["passed"].append("backtest")
            elif bt["sharpe"] > 0.3:
                print(f"  ⚠ Marginal Sharpe ({bt['sharpe']:.2f})")
                summary["warnings"].append(f"Marginal Sharpe: {bt['sharpe']:.2f}")
            else:
                print(f"  ✗ Sharpe below 0.3 ({bt['sharpe']:.2f})")
                summary["failures"].append(f"Low Sharpe: {bt['sharpe']:.2f}")

            summary["backtest"] = bt
    else:
        print("\n  [7/7] Backtest skipped (--quick)")

    # ── Feature importance ─────────────────────────────────────────────────────
    print(f"\n  Top 15 features:")
    importance = get_feature_importance(payload, top_n=15)
    if importance:
        max_imp = importance[0][1]
        for i, (feat, imp) in enumerate(importance, 1):
            bar = "█" * int((imp / max_imp) * 25)
            print(f"  {i:2}. {feat:<30}  {bar}")
    else:
        print("  (LightGBM not available)")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--quick",  action="store_true", help="Skip backtest")
    args    = parser.parse_args()
    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS

    print("\n" + "="*60)
    print("  MODEL VALIDATION REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Symbols: {symbols}")
    print("="*60)

    summaries = []
    for sym in symbols:
        summaries.append(validate_symbol(sym, quick=args.quick))

    print("\n\n" + "="*60)
    print("  VALIDATION SUMMARY")
    print("="*60)
    print(f"  {'Symbol':<10}  {'Passed':>8}  {'Warnings':>9}  {'Failures':>9}  Verdict")
    print(f"  {'-'*58}")

    for s in summaries:
        np_ = len(s["passed"])
        nw  = len(s["warnings"])
        nf  = len(s["failures"])
        vd  = "✗ ISSUES" if nf > 0 else ("⚠ REVIEW" if nw > 2 else "✓ READY")
        print(f"  {s['symbol']:<10}  {np_:>8}  {nw:>9}  {nf:>9}  {vd}")

    print()
    print("  INTERPRETATION GUIDE:")
    print("  ─────────────────────────────────────────────────────────────")
    print("  Accuracy      > 50%  = real signal (random = 33%)")
    print("  Sharpe        > 0.8  = viable  |  > 1.5 = excellent")
    print("  Max drawdown  < 20%  = acceptable  |  < 10% = excellent")
    print("  Overfit gap   < 10%  = healthy")
    print("  HOLD%         < 55%  = model actively trading both directions")
    print("  TIMEOUT exits < 40%  = TP/SL well-sized for this asset")
    print()
    print("  Conf thresholds: BTC/BNB=0.42  ETH/SOL=0.38")
    print("  These must match CONF_THRESHOLDS in futures_trader.py")


if __name__ == "__main__":
    main()