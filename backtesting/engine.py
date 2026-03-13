"""
backtesting/engine.py
──────────────────────
Vectorized backtesting engine with realistic market simulation.

Features:
  - Taker/maker fee differentiation
  - Configurable slippage model
  - Long and short futures support
  - Configurable leverage
  - ATR-based dynamic stop-loss / take-profit
  - Per-trade logging
  - Equity curve + drawdown tracking

Usage:
    from backtesting.engine import BacktestEngine
    from backtesting.metrics import compute_metrics

    engine = BacktestEngine(
        initial_capital = 10_000,
        leverage        = 10,
        risk_per_trade  = 0.01,   # 1% per trade
        taker_fee       = 0.0004,
        slippage        = 0.0002,
    )
    result = engine.run(df_with_signals)
    print(compute_metrics(result.equity_curve, result.trades))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """A single completed trade."""
    entry_bar:    int
    exit_bar:     int
    direction:    int           # +1 = long, -1 = short
    entry_price:  float
    exit_price:   float
    quantity:     float         # contracts
    leverage:     int
    pnl_gross:    float         # before fees
    pnl_net:      float         # after fees + slippage
    fee_paid:     float
    slippage_cost: float
    exit_reason:  str           # "TP" | "SL" | "SIGNAL_FLIP" | "EOD" | "END"
    confidence:   float = 0.0   # model confidence at entry


@dataclass
class BacktestResult:
    """Full backtest output."""
    equity_curve:     np.ndarray    # equity at each bar
    trades:           List[Trade]
    initial_capital:  float
    final_capital:    float
    total_return:     float
    signals:          np.ndarray    # 0=HOLD, 1=LONG, 2=SHORT
    prices:           np.ndarray
    drawdown_curve:   np.ndarray


# ──────────────────────────────────────────────────────────────────────────────
# Backtest Engine
# ──────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Event-driven backtesting engine for leveraged futures trading.

    Position Sizing:
        The engine uses risk-based position sizing (not fixed lot size):
        quantity = (equity × risk_per_trade) / (entry_price × stop_pct)

    Stop / Take Profit:
        If atr_column is provided, TP/SL are set as multiples of ATR.
        Otherwise, fixed percentage TP/SL are used.

    Parameters
    ----------
    initial_capital : starting USDT balance
    leverage        : futures leverage multiplier
    risk_per_trade  : fraction of equity risked per trade (0.01 = 1%)
    taker_fee       : taker fee rate (Binance = 0.0004)
    maker_fee       : maker fee rate (Binance = 0.0002)
    slippage        : one-way slippage rate (0.0002 = 0.02%)
    tp_atr_mult     : take-profit = entry ± tp_atr_mult × ATR
    sl_atr_mult     : stop-loss   = entry ∓ sl_atr_mult × ATR
    tp_pct          : fixed TP % (used if no ATR column)
    sl_pct          : fixed SL % (used if no ATR column)
    max_open_bars   : force-close position after this many bars
    min_confidence  : skip trades with model confidence below this
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        leverage:        int   = 10,
        risk_per_trade:  float = 0.01,
        taker_fee:       float = 0.0004,
        maker_fee:       float = 0.0002,
        slippage:        float = 0.0002,
        tp_atr_mult:     float = 2.0,
        sl_atr_mult:     float = 1.0,
        tp_pct:          float = 0.02,
        sl_pct:          float = 0.01,
        max_open_bars:   int   = 48,
        min_confidence:  float = 0.60,
    ):
        self.initial_capital = initial_capital
        self.leverage        = leverage
        self.risk_per_trade  = risk_per_trade
        self.taker_fee       = taker_fee
        self.maker_fee       = maker_fee
        self.slippage        = slippage
        self.tp_atr_mult     = tp_atr_mult
        self.sl_atr_mult     = sl_atr_mult
        self.tp_pct          = tp_pct
        self.sl_pct          = sl_pct
        self.max_open_bars   = max_open_bars
        self.min_confidence  = min_confidence

    def run(
        self,
        df:              pd.DataFrame,
        signal_col:      str = "signal",
        confidence_col:  Optional[str] = "confidence",
        atr_col:         Optional[str] = "atr",
    ) -> BacktestResult:
        """
        Run the backtest on a DataFrame.

        Required columns: open, high, low, close, {signal_col}
        Optional columns: {confidence_col}, {atr_col}

        signal_col values: 0=HOLD, 1=LONG, 2=SHORT
        """
        closes     = df["close"].values.astype(float)
        highs      = df["high"].values.astype(float)
        lows       = df["low"].values.astype(float)
        signals    = df[signal_col].values.astype(int)
        confidence = df[confidence_col].values.astype(float) if confidence_col and confidence_col in df.columns else np.ones(len(df))
        atr        = df[atr_col].values.astype(float)        if atr_col         and atr_col         in df.columns else None

        n             = len(df)
        equity        = np.full(n, self.initial_capital)
        trades: List[Trade] = []

        # Position state
        in_position   = False
        direction     = 0
        entry_price   = 0.0
        entry_bar     = 0
        quantity      = 0.0
        tp_price      = 0.0
        sl_price      = 0.0
        entry_conf    = 0.0

        for i in range(n):
            price = closes[i]
            high  = highs[i]
            low   = lows[i]
            sig   = signals[i]
            conf  = float(confidence[i])
            atr_v = float(atr[i]) if atr is not None and not np.isnan(atr[i]) else price * self.sl_pct

            # Update equity mark-to-market while in position
            if in_position:
                equity[i] = equity[i - 1] + direction * (price - entry_price) * quantity

                # Check TP / SL using intra-bar high/low
                tp_hit = (high >= tp_price) if direction == 1 else (low <= tp_price)
                sl_hit = (low  <= sl_price) if direction == 1 else (high >= sl_price)

                # Force close after max_open_bars
                time_hit = (i - entry_bar) >= self.max_open_bars

                if tp_hit or sl_hit or time_hit:
                    exit_reason = "TP" if tp_hit else ("SL" if sl_hit else "MAX_BARS")
                    exit_p = tp_price if tp_hit else (sl_price if sl_hit else price)
                    trade  = self._close(
                        entry_bar, i, direction, entry_price, exit_p,
                        quantity, entry_conf, exit_reason, equity[i - 1]
                    )
                    trades.append(trade)
                    equity[i]   = equity[i - 1] + trade.pnl_net
                    in_position = False
                    direction   = 0

            else:
                equity[i] = equity[i - 1] if i > 0 else self.initial_capital

            # Check for new entry signal
            if not in_position and sig in (1, 2) and conf >= self.min_confidence:
                dir_new  = 1 if sig == 1 else -1
                entry_p  = price * (1 + self.slippage * dir_new)

                # Risk-based position sizing
                stop_dist = atr_v * self.sl_atr_mult / price
                stop_dist = max(stop_dist, 0.001)   # minimum 0.1%
                notional  = equity[i] * self.risk_per_trade * self.leverage / stop_dist
                qty       = notional / entry_p

                if qty <= 0 or np.isnan(qty):
                    continue

                # Set TP/SL
                if atr is not None:
                    tp = entry_p + dir_new * self.tp_atr_mult * atr_v
                    sl = entry_p - dir_new * self.sl_atr_mult * atr_v
                else:
                    tp = entry_p * (1 + dir_new * self.tp_pct)
                    sl = entry_p * (1 - dir_new * self.sl_pct)

                in_position = True
                direction   = dir_new
                entry_price = entry_p
                entry_bar   = i
                quantity    = qty
                tp_price    = tp
                sl_price    = sl
                entry_conf  = conf

        # Close any open position at end of data
        if in_position:
            trade = self._close(
                entry_bar, n - 1, direction, entry_price, closes[-1],
                quantity, entry_conf, "END", equity[-2]
            )
            trades.append(trade)
            equity[-1] = equity[-2] + trade.pnl_net

        # Drawdown curve
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / (peak + 1e-10)

        return BacktestResult(
            equity_curve    = equity,
            trades          = trades,
            initial_capital = self.initial_capital,
            final_capital   = float(equity[-1]),
            total_return    = float((equity[-1] / self.initial_capital) - 1.0),
            signals         = signals,
            prices          = closes,
            drawdown_curve  = drawdown,
        )

    def _close(
        self,
        entry_bar:    int,
        exit_bar:     int,
        direction:    int,
        entry_price:  float,
        exit_price:   float,
        quantity:     float,
        confidence:   float,
        reason:       str,
        prev_equity:  float,
    ) -> Trade:
        """Compute net PnL for a closed trade."""
        exit_p_slip  = exit_price * (1 - self.slippage * direction)
        pnl_gross    = direction * (exit_p_slip - entry_price) * quantity
        fee_entry    = entry_price * quantity * self.taker_fee
        fee_exit     = exit_p_slip * quantity * self.taker_fee
        fee_total    = fee_entry + fee_exit
        slippage_cost = abs(exit_price - exit_p_slip) * quantity + abs(entry_price * self.slippage) * quantity
        pnl_net      = pnl_gross - fee_total

        return Trade(
            entry_bar     = entry_bar,
            exit_bar      = exit_bar,
            direction     = direction,
            entry_price   = entry_price,
            exit_price    = exit_p_slip,
            quantity      = quantity,
            leverage      = self.leverage,
            pnl_gross     = pnl_gross,
            pnl_net       = pnl_net,
            fee_paid      = fee_total,
            slippage_cost = slippage_cost,
            exit_reason   = reason,
            confidence    = confidence,
        )
