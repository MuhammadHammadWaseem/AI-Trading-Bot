"""
models/training/trainer.py
───────────────────────────
Production ML training pipeline with walk-forward validation,
class balancing, and probability calibration.

Replaces the inline training code in core/models/ml_model.py.

Key improvements over original:
  1. ATR-adaptive labels (not fixed 0.4% threshold)
  2. XGBoost + LightGBM both get class balancing
  3. Walk-forward validation (not naive 80/20 split)
  4. CalibratedClassifierCV (isotonic) for reliable probabilities
  5. Proper metrics: precision, recall, F1, Sharpe per fold
  6. Returns structured TrainingResult (not FakeHistory)

Usage:
    from models.training.trainer import ModelTrainer

    trainer = ModelTrainer(symbol="BTCUSDT")
    result  = trainer.train(df)          # df must have add_all_indicators() applied
    if result.accepted:
        trainer.save(result)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score

from models.training.label_engine import generate_labels, label_distribution
from research.walk_forward import WalkForwardValidator
from data.indicators import get_feature_columns
from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    """Full output from a training run."""
    symbol:         str
    accepted:       bool          # passed walk-forward acceptance thresholds
    model:          object        # fitted ensemble (list of (name, model, weight))
    scaler:         RobustScaler
    feature_names:  List[str]
    wf_sharpe:      float         # mean walk-forward Sharpe
    wf_f1:          float         # mean walk-forward F1 macro
    wf_consistency: float         # fraction of folds with positive Sharpe
    val_accuracy:   float
    val_report:     str
    label_dist:     dict          # HOLD/LONG/SHORT counts


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ML feature matrix from indicator-enriched DataFrame.
    Uses pd.concat for all columns at once — avoids PerformanceWarning.
    """
    cols = {}

    # ── Base indicator columns ────────────────────────────────────────────
    for col in get_feature_columns():
        if col in df.columns:
            cols[col] = df[col]

    # ── Multi-period returns ───────────────────────────────────────────────
    for p in [1, 3, 5, 10, 20]:
        cols[f"return_{p}"]   = df["close"].pct_change(p)
        cols[f"log_ret_{p}"]  = np.log(df["close"] / (df["close"].shift(p) + 1e-10))
        cols[f"hl_range_{p}"] = (
            df["high"].rolling(p).max() - df["low"].rolling(p).min()
        ) / (df["close"] + 1e-10)

    # ── Rolling stats on oscillators ──────────────────────────────────────
    for col in ["rsi", "macd", "adx", "volume"]:
        if col not in df.columns:
            continue
        for w in [5, 10, 20]:
            rm = df[col].rolling(w).mean()
            rs = df[col].rolling(w).std()
            cols[f"{col}_ma{w}"]  = rm
            cols[f"{col}_std{w}"] = rs
            cols[f"{col}_z{w}"]   = (df[col] - rm) / (rs + 1e-10)

    # ── Indicator crossover signals ────────────────────────────────────────
    if "rsi" in df.columns:
        cols["rsi_oversold"]       = (df["rsi"] < 30).astype(int)
        cols["rsi_overbought"]     = (df["rsi"] > 70).astype(int)
        cols["rsi_mid_cross_up"]   = ((df["rsi"] > 50) & (df["rsi"].shift(1) <= 50)).astype(int)
        cols["rsi_mid_cross_down"] = ((df["rsi"] < 50) & (df["rsi"].shift(1) >= 50)).astype(int)

    if "macd" in df.columns and "macd_signal" in df.columns:
        cols["macd_cross_up"]    = ((df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))).astype(int)
        cols["macd_cross_down"]  = ((df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))).astype(int)
        cols["macd_positive"]    = (df["macd"] > 0).astype(int)
        cols["macd_hist_rising"] = (df["macd_hist"] > df["macd_hist"].shift(1)).astype(int)

    if "ema_9" in df.columns and "ema_21" in df.columns:
        cols["ema_9_21_cross_up"]   = ((df["ema_9"] > df["ema_21"]) & (df["ema_9"].shift(1) <= df["ema_21"].shift(1))).astype(int)
        cols["ema_9_21_cross_down"] = ((df["ema_9"] < df["ema_21"]) & (df["ema_9"].shift(1) >= df["ema_21"].shift(1))).astype(int)

    # ── BB squeeze ────────────────────────────────────────────────────────
    if "bb_width" in df.columns:
        cols["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(20).quantile(0.20)).astype(int)

    # ── ATR-normalized price features ─────────────────────────────────────
    if "atr" in df.columns:
        cols["atr_ratio"] = df["atr"] / (df["close"] + 1e-10)
        cols["atr_z20"]   = (df["atr"] - df["atr"].rolling(20).mean()) / (df["atr"].rolling(20).std() + 1e-10)

    # ── Volume × price momentum ────────────────────────────────────────────
    if "volume" in df.columns:
        cols["vol_price_momentum"] = df["close"].pct_change() * np.log1p(df["volume"])
        cols["vol_spike"]          = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)

    # ── Build DataFrame in one concat call (no fragmentation) ─────────────
    f = pd.concat(cols, axis=1)
    f = f.replace([np.inf, -np.inf], np.nan)
    return f


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Trains the LightGBM + XGBoost ensemble with proper ML practices.

    Flow:
      1. engineer_features(df)
      2. generate_labels(df, method="triple_barrier")
      3. WalkForwardValidator.run() → fold metrics
      4. Final fit on full data (if walk-forward accepted)
      5. CalibratedClassifierCV wrapping
      6. Save model + scaler
    """

    LABEL_METHOD   = "triple_barrier"
    FUTURE_BARS    = 8
    ATR_MULT       = 1.5
    N_WF_SPLITS    = 5
    WF_GAP         = 8      # = FUTURE_BARS to prevent label leakage

    def __init__(self, symbol: str = "BTCUSDT", model_dir: Path = None):
        self.symbol    = symbol.replace("/", "")
        self.model_dir = model_dir or settings.model.saved_models_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(self, df: pd.DataFrame) -> TrainingResult:
        """Full training pipeline. df must have add_all_indicators() applied."""
        logger.info(f"[TRAINER] Starting training for {self.symbol} | {len(df)} bars")

        # ── 1. Features ───────────────────────────────────────────────────
        feat_df = engineer_features(df)
        labels  = generate_labels(
            df,
            method      = self.LABEL_METHOD,
            future_bars = self.FUTURE_BARS,
            atr_mult    = self.ATR_MULT,
        )

        dist = label_distribution(labels)
        logger.info(
            f"[TRAINER] Labels: HOLD={dist['HOLD_pct']:.1f}% "
            f"LONG={dist['LONG_pct']:.1f}% SHORT={dist['SHORT_pct']:.1f}%"
        )

        # ── 2. Align and clean ────────────────────────────────────────────
        valid_mask = ~feat_df.isnull().any(axis=1)
        feat_df    = feat_df[valid_mask]
        labels     = labels[valid_mask.values]

        # Drop tail (no valid labels for last FUTURE_BARS bars)
        feat_df = feat_df.iloc[: -self.FUTURE_BARS]
        labels  = labels[: -self.FUTURE_BARS]

        feature_names = feat_df.columns.tolist()
        X = feat_df.values.astype(np.float32)
        y = labels

        logger.info(
            f"[TRAINER] Features={X.shape[1]}  Samples={len(X):,}  "
            f"HOLD={int((y==0).sum()):,} LONG={int((y==1).sum()):,} SHORT={int((y==2).sum()):,}"
        )

        if len(X) < 500:
            logger.warning("[TRAINER] Too few samples for reliable training (< 500)")

        # ── 3. Walk-Forward Validation ────────────────────────────────────
        prices = df["close"].values[valid_mask.values][: -self.FUTURE_BARS] \
                 if "close" in df.columns else None

        wfv = WalkForwardValidator(
            n_splits        = self.N_WF_SPLITS,
            gap             = self.WF_GAP,
            min_train_size  = 300,
            window_type     = "expanding",
        )

        wf_result = wfv.run(
            X        = X,
            y        = y,
            train_fn = self._build_and_train_fold,
            prices   = prices,
            verbose  = True,
        )

        accepted = wfv.should_accept_model(wf_result)
        logger.info(f"[TRAINER] Walk-forward verdict: {wf_result.recommendation}")

        # ── 4. Final fit on all data ──────────────────────────────────────
        scaler    = RobustScaler()
        X_scaled  = scaler.fit_transform(X)
        models    = self._build_final_ensemble(X_scaled, y)

        # ── 5. Validation report (last 20%) ───────────────────────────────
        split      = int(len(X) * 0.80)
        X_val_s    = scaler.transform(X[split:])
        y_val      = y[split:]
        final_probs = self._ensemble_proba(models, X_val_s)
        final_preds = np.argmax(final_probs, axis=1)
        val_acc     = float((final_preds == y_val).mean())
        val_report  = classification_report(
            y_val, final_preds,
            target_names=["HOLD", "LONG", "SHORT"],
            zero_division=0,
        )
        logger.info(f"[TRAINER] Val accuracy: {val_acc:.2%}")
        print(val_report)

        return TrainingResult(
            symbol        = self.symbol,
            accepted      = accepted,
            model         = models,
            scaler        = scaler,
            feature_names = feature_names,
            wf_sharpe     = wf_result.mean_sharpe,
            wf_f1         = wf_result.mean_f1_macro,
            wf_consistency = wf_result.consistency,
            val_accuracy  = val_acc,
            val_report    = val_report,
            label_dist    = dist,
        )

    def save(self, result: TrainingResult) -> Path:
        """Persist model artifacts to disk."""
        path = self.model_dir / f"ml_{self.symbol}.joblib"
        joblib.dump({
            "models":        result.model,
            "scaler":        result.scaler,
            "feature_names": result.feature_names,
            "wf_sharpe":     result.wf_sharpe,
            "wf_f1":         result.wf_f1,
            "label_dist":    result.label_dist,
            "wf_accepted":   result.accepted,   # used by MLModel._try_load()
        }, path)
        logger.info(f"[TRAINER] Saved: {path}")
        return path

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_and_train_fold(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train a calibrated ensemble on one walk-forward fold.
        Called by WalkForwardValidator.run() for each fold.
        Returns an object with .predict_proba(X_test).
        """
        scaler   = RobustScaler()
        X_scaled = scaler.fit_transform(X_train)

        # Compute class weights
        classes = np.array([0, 1, 2])
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        w_dict  = {0: weights[0], 1: weights[1], 2: weights[2]}
        sample_w = np.array([w_dict[y] for y in y_train])

        models = self._build_final_ensemble(X_scaled, y_train, sample_weights=sample_w)

        class FoldWrapper:
            """Wraps models + scaler to expose predict_proba."""
            def __init__(self, models_, scaler_):
                self._models = models_
                self._scaler = scaler_

            def predict_proba(self_, X):
                Xs = self_._scaler.transform(X)
                total_w = sum(w for _, _, w in self_._models)
                probs = None
                for _, m, w in self_._models:
                    p = m.predict_proba(Xs) * (w / total_w)
                    if p.shape[1] < 3:
                        full = np.zeros((len(p), 3))
                        for ci, cls in enumerate(m.classes_):
                            full[:, cls] = p[:, ci]
                        p = full
                    probs = p if probs is None else probs + p
                return probs

        return FoldWrapper(models, scaler)

    def _build_final_ensemble(
        self,
        X_scaled:       np.ndarray,
        y:              np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> list:
        """Build the LightGBM + XGBoost ensemble. Falls back to sklearn."""
        models = []

        # Compute class weights if not provided
        if sample_weights is None:
            classes = np.array([0, 1, 2])
            weights = compute_class_weight("balanced", classes=classes, y=y)
            w_dict  = {0: weights[0], 1: weights[1], 2: weights[2]}
            sample_weights = np.array([w_dict[yi] for yi in y])

        split = int(len(X_scaled) * 0.85)
        X_tr, y_tr = X_scaled[:split], y[:split]
        X_val, y_val = X_scaled[split:], y[split:]
        sw_tr = sample_weights[:split]

        # ── LightGBM ─────────────────────────────────────────────────────
        try:
            import lightgbm as lgbm
            lgb_model = lgbm.LGBMClassifier(
                n_estimators       = 500,
                learning_rate      = 0.05,
                max_depth          = 6,
                num_leaves         = 63,
                min_child_samples  = 30,
                subsample          = 0.8,
                colsample_bytree   = 0.8,
                class_weight       = "balanced",
                random_state       = 42,
                n_jobs             = -1,
                verbose            = -1,
            )
            lgb_model.fit(
                X_tr, y_tr,
                sample_weight  = sw_tr,
                eval_set       = [(X_val, y_val)],
                callbacks      = [
                    lgbm.early_stopping(50, verbose=False),
                    lgbm.log_evaluation(200),
                ],
            )
            # Calibrate probabilities
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lgb_cal = CalibratedClassifierCV(lgb_model, method="isotonic", cv="prefit")
                lgb_cal.fit(X_val, y_val)
            models.append(("lgbm", lgb_cal, 0.55))
            logger.info(f"[TRAINER] LightGBM trained + calibrated")
        except ImportError:
            logger.warning("[TRAINER] LightGBM not installed — skipping")
        except Exception as e:
            logger.warning(f"[TRAINER] LightGBM failed: {e}")

        # ── XGBoost ──────────────────────────────────────────────────────
        try:
            import xgboost as xgb
            xgb_model = xgb.XGBClassifier(
                n_estimators       = 500,
                learning_rate      = 0.05,
                max_depth          = 5,
                subsample          = 0.8,
                colsample_bytree   = 0.8,
                eval_metric        = "mlogloss",
                random_state       = 42,
                n_jobs             = -1,
                verbosity          = 0,
                use_label_encoder  = False,
            )
            xgb_model.fit(
                X_tr, y_tr,
                sample_weight       = sw_tr,
                eval_set            = [(X_val, y_val)],
                early_stopping_rounds = 50,
                verbose             = False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xgb_cal = CalibratedClassifierCV(xgb_model, method="isotonic", cv="prefit")
                xgb_cal.fit(X_val, y_val)
            models.append(("xgb", xgb_cal, 0.45))
            logger.info(f"[TRAINER] XGBoost trained + calibrated")
        except ImportError:
            logger.warning("[TRAINER] XGBoost not installed — skipping")
        except Exception as e:
            logger.warning(f"[TRAINER] XGBoost failed: {e}")

        # ── Fallback: RandomForest + GradientBoosting ─────────────────────
        if not models:
            logger.warning("[TRAINER] Using sklearn fallback (install lightgbm/xgboost)")
            from sklearn.ensemble import (
                GradientBoostingClassifier, RandomForestClassifier,
                ExtraTreesClassifier
            )
            rf = RandomForestClassifier(
                n_estimators=300, max_depth=10,
                class_weight="balanced", random_state=42, n_jobs=-1
            )
            gb = GradientBoostingClassifier(
                n_estimators=300, max_depth=5,
                learning_rate=0.05, random_state=42
            )
            et = ExtraTreesClassifier(
                n_estimators=200, max_depth=12,
                class_weight="balanced", random_state=42, n_jobs=-1
            )
            rf.fit(X_tr, y_tr, sample_weight=sw_tr)
            gb.fit(X_tr, y_tr, sample_weight=sw_tr)
            et.fit(X_tr, y_tr, sample_weight=sw_tr)

            for name, m, w in [("rf", rf, 0.4), ("gb", gb, 0.35), ("et", et, 0.25)]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cal = CalibratedClassifierCV(m, method="isotonic", cv="prefit")
                    cal.fit(X_val, y_val)
                models.append((name, cal, w))

            logger.info("[TRAINER] Sklearn fallback ensemble trained + calibrated")

        return models

    @staticmethod
    def _ensemble_proba(models: list, X_scaled: np.ndarray) -> np.ndarray:
        """Weighted soft-vote across all models."""
        total_w = sum(w for _, _, w in models)
        probs   = None
        for _, m, w in models:
            p = m.predict_proba(X_scaled) * (w / total_w)
            if p.shape[1] < 3:
                full = np.zeros((len(p), 3))
                for ci, cls in enumerate(m.classes_):
                    full[:, cls] = p[:, ci]
                p = full
            probs = p if probs is None else probs + p
        return probs