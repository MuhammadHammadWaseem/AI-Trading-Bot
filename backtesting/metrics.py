"""
backtesting/metrics.py
───────────────────────
Professional performance metrics for strategy evaluation.

All metrics are computed from an equity curve and/or trade list.
Used by both the backtester and the walk-forward validator.

Public API:
    compute_metrics(equity_curve, trades, periods_per_year) -> dict
    print_report(metrics_dict)
"""

from __future__ import annotations

import math
import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backtesting.engine import Trade


def compute_metrics(
    equity_curve:      np.ndarray,
    trades:            List["Trade"],
    periods_per_year:  int = 35040,   # 15m bars: 4/hr × 24 × 365
    risk_free_rate:    float = 0.05,  # 5% annualized
) -> dict:
    """
    Compute a comprehensive set of performance metrics.

    Parameters
    ----------
    equity_curve     : equity at each bar (output of BacktestEngine.run)
    trades           : list of Trade objects
    periods_per_year : number of bars per year (for annualization)
    risk_free_rate   : annualized risk-free rate for Sharpe computation

    Returns
    -------
    dict with all metrics (see keys below)
    """
    metrics = {}

    # ── Basic P&L ─────────────────────────────────────────────────────────
    initial   = float(equity_curve[0])
    final     = float(equity_curve[-1])
    total_ret = (final / initial) - 1.0

    metrics["initial_capital"]  = initial
    metrics["final_capital"]    = final
    metrics["total_return_pct"] = total_ret * 100.0
    metrics["total_return_x"]   = final / initial

    # ── Trade Statistics ──────────────────────────────────────────────────
    n_trades = len(trades)
    metrics["n_trades"] = n_trades

    if n_trades == 0:
        return _empty_metrics(metrics, initial, final)

    pnls     = np.array([t.pnl_net   for t in trades])
    wins     = pnls[pnls > 0]
    losses   = pnls[pnls < 0]

    metrics["n_wins"]              = len(wins)
    metrics["n_losses"]            = len(losses)
    metrics["win_rate_pct"]        = len(wins) / n_trades * 100.0
    metrics["avg_win"]             = float(wins.mean())  if len(wins)   > 0 else 0.0
    metrics["avg_loss"]            = float(losses.mean()) if len(losses) > 0 else 0.0
    metrics["largest_win"]         = float(wins.max())   if len(wins)   > 0 else 0.0
    metrics["largest_loss"]        = float(losses.min()) if len(losses) > 0 else 0.0
    metrics["avg_trade_pnl"]       = float(pnls.mean())

    # Profit factor = gross profit / gross loss
    gross_profit = float(wins.sum())   if len(wins)   > 0 else 0.0
    gross_loss   = float(-losses.sum()) if len(losses) > 0 else 1e-10
    metrics["profit_factor"]       = min(gross_profit / (gross_loss + 1e-10), 99.0)
    metrics["total_fees_paid"]     = float(sum(t.fee_paid for t in trades))

    # Expectancy per trade
    wr  = len(wins) / n_trades
    avg_win  = metrics["avg_win"]
    avg_loss = metrics["avg_loss"]
    metrics["expectancy"] = wr * avg_win + (1 - wr) * avg_loss

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    metrics["exit_reasons"] = exit_reasons

    # Direction split
    longs  = [t for t in trades if t.direction ==  1]
    shorts = [t for t in trades if t.direction == -1]
    metrics["n_longs"]  = len(longs)
    metrics["n_shorts"] = len(shorts)

    # ── Equity Curve / Drawdown ───────────────────────────────────────────
    peak     = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / (peak + 1e-10)

    metrics["max_drawdown_pct"]    = float(drawdown.min()) * 100.0
    metrics["max_drawdown_usdt"]   = float((equity_curve - peak).min())

    # Calmar ratio = annualized return / |max drawdown|
    n_bars   = len(equity_curve)
    years    = n_bars / periods_per_year
    ann_ret  = (final / initial) ** (1.0 / max(years, 1e-6)) - 1.0
    max_dd   = abs(metrics["max_drawdown_pct"] / 100.0)
    metrics["calmar_ratio"]        = ann_ret / (max_dd + 1e-10)
    metrics["annualized_return_pct"] = ann_ret * 100.0

    # Average + max drawdown duration
    in_dd        = drawdown < -0.001
    dd_durations = _drawdown_durations(in_dd)
    metrics["avg_drawdown_bars"]   = float(np.mean(dd_durations)) if dd_durations else 0.0
    metrics["max_drawdown_bars"]   = int(max(dd_durations)) if dd_durations else 0

    # ── Sharpe Ratio ──────────────────────────────────────────────────────
    bar_returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-10)
    rfr_per_bar = (1 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess      = bar_returns - rfr_per_bar

    if bar_returns.std() > 1e-10:
        sharpe = float(excess.mean() / excess.std() * np.sqrt(periods_per_year))
    else:
        sharpe = 0.0
    metrics["sharpe_ratio"] = sharpe

    # ── Sortino Ratio (downside deviation only) ───────────────────────────
    downside = excess[excess < 0]
    if len(downside) > 1:
        sortino = float(excess.mean() / (downside.std() + 1e-10) * np.sqrt(periods_per_year))
    else:
        sortino = 0.0
    metrics["sortino_ratio"] = sortino

    # ── Hold duration stats ───────────────────────────────────────────────
    durations = [t.exit_bar - t.entry_bar for t in trades]
    metrics["avg_hold_bars"]  = float(np.mean(durations)) if durations else 0.0
    metrics["min_hold_bars"]  = int(min(durations))       if durations else 0
    metrics["max_hold_bars"]  = int(max(durations))       if durations else 0

    return metrics


def print_report(metrics: dict) -> None:
    """Print a formatted performance report."""
    divider = "─" * 52
    print(f"\n{'='*52}")
    print(f"  BACKTEST PERFORMANCE REPORT")
    print(f"{'='*52}")

    print(f"\n  RETURNS")
    print(divider)
    print(f"  Initial Capital   : ${metrics.get('initial_capital', 0):>12,.2f}")
    print(f"  Final Capital     : ${metrics.get('final_capital', 0):>12,.2f}")
    print(f"  Total Return      : {metrics.get('total_return_pct', 0):>12.2f}%")
    print(f"  Ann. Return       : {metrics.get('annualized_return_pct', 0):>12.2f}%")

    print(f"\n  RISK-ADJUSTED")
    print(divider)
    print(f"  Sharpe Ratio      : {metrics.get('sharpe_ratio', 0):>12.3f}")
    print(f"  Sortino Ratio     : {metrics.get('sortino_ratio', 0):>12.3f}")
    print(f"  Calmar Ratio      : {metrics.get('calmar_ratio', 0):>12.3f}")
    print(f"  Max Drawdown      : {metrics.get('max_drawdown_pct', 0):>12.2f}%")
    print(f"  Max DD Duration   : {metrics.get('max_drawdown_bars', 0):>12} bars")

    print(f"\n  TRADE STATISTICS")
    print(divider)
    print(f"  Total Trades      : {metrics.get('n_trades', 0):>12}")
    print(f"  Win Rate          : {metrics.get('win_rate_pct', 0):>12.1f}%")
    print(f"  Profit Factor     : {metrics.get('profit_factor', 0):>12.3f}")
    print(f"  Expectancy        : ${metrics.get('expectancy', 0):>12.4f}")
    print(f"  Avg Win           : ${metrics.get('avg_win', 0):>12.4f}")
    print(f"  Avg Loss          : ${metrics.get('avg_loss', 0):>12.4f}")
    print(f"  Largest Win       : ${metrics.get('largest_win', 0):>12.4f}")
    print(f"  Largest Loss      : ${metrics.get('largest_loss', 0):>12.4f}")
    print(f"  Longs / Shorts    : {metrics.get('n_longs',0):>5} / {metrics.get('n_shorts',0):<5}")
    print(f"  Avg Hold (bars)   : {metrics.get('avg_hold_bars', 0):>12.1f}")
    print(f"  Total Fees Paid   : ${metrics.get('total_fees_paid', 0):>12.4f}")

    if "exit_reasons" in metrics:
        print(f"\n  EXIT REASONS")
        print(divider)
        for reason, count in sorted(metrics["exit_reasons"].items()):
            pct = count / max(metrics.get('n_trades', 1), 1) * 100
            print(f"  {reason:<20}: {count:>5}  ({pct:.1f}%)")

    print(f"\n{'='*52}\n")


def strategy_grade(metrics: dict) -> str:
    """
    Assign a letter grade based on combined metrics.
    Grade scale:
      A+ : Sharpe > 2.0,  PF > 2.0, WR > 60%, MaxDD < 15%
      A  : Sharpe > 1.5,  PF > 1.8, WR > 55%, MaxDD < 20%
      B  : Sharpe > 1.0,  PF > 1.5, WR > 50%, MaxDD < 25%
      C  : Sharpe > 0.5,  PF > 1.2, WR > 45%, MaxDD < 35%
      D  : Sharpe > 0.0,  PF > 1.0
      F  : otherwise (unprofitable / random)
    """
    sh  = metrics.get("sharpe_ratio",       0)
    pf  = metrics.get("profit_factor",      0)
    wr  = metrics.get("win_rate_pct",        0)
    mdd = abs(metrics.get("max_drawdown_pct", 100))

    if sh > 2.0 and pf > 2.0 and wr > 60 and mdd < 15:
        return "A+"
    if sh > 1.5 and pf > 1.8 and wr > 55 and mdd < 20:
        return "A"
    if sh > 1.0 and pf > 1.5 and wr > 50 and mdd < 25:
        return "B"
    if sh > 0.5 and pf > 1.2 and wr > 45 and mdd < 35:
        return "C"
    if sh > 0.0 and pf > 1.0:
        return "D"
    return "F"


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _empty_metrics(base: dict, initial: float, final: float) -> dict:
    """Return a zeroed metrics dict when there are no trades."""
    base.update({
        "n_trades": 0, "n_wins": 0, "n_losses": 0,
        "win_rate_pct": 0.0, "profit_factor": 0.0,
        "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
        "max_drawdown_pct": 0.0, "calmar_ratio": 0.0,
        "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
    })
    return base


def _drawdown_durations(in_dd_mask: np.ndarray) -> List[int]:
    """Count lengths of consecutive True runs in boolean array."""
    durations = []
    count = 0
    for v in in_dd_mask:
        if v:
            count += 1
        elif count > 0:
            durations.append(count)
            count = 0
    if count > 0:
        durations.append(count)
    return durations
