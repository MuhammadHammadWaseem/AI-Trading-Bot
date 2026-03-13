"""
backtesting/portfolio.py
─────────────────────────
Multi-symbol portfolio backtest aggregation.

Runs individual symbol backtests and combines them into a
portfolio-level equity curve with correlation and diversification metrics.

Usage:
    from backtesting.portfolio import PortfolioBacktest
    from backtesting.engine import BacktestEngine

    engine = BacktestEngine(initial_capital=10_000)
    portfolio = PortfolioBacktest(symbols=["BTCUSDT","ETHUSDT","SOLUSDT"])
    result = portfolio.run(dfs_dict, engine)
    portfolio.print_summary(result)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.metrics import compute_metrics, print_report, strategy_grade


@dataclass
class PortfolioResult:
    """Aggregated portfolio results across symbols."""
    symbol_results:      Dict[str, BacktestResult]
    symbol_metrics:      Dict[str, dict]
    combined_equity:     np.ndarray
    combined_metrics:    dict
    correlation_matrix:  Optional[np.ndarray]
    portfolio_grade:     str


class PortfolioBacktest:
    """
    Runs a BacktestEngine across multiple symbols and aggregates results.

    Capital allocation: equal-weight by default (capital / n_symbols per symbol).
    """

    def __init__(
        self,
        symbols:     List[str],
        allocation:  Optional[Dict[str, float]] = None,  # symbol -> fraction
    ):
        self.symbols    = symbols
        # Equal weight if no allocation specified
        if allocation:
            total = sum(allocation.values())
            self.allocation = {s: v / total for s, v in allocation.items()}
        else:
            w = 1.0 / len(symbols)
            self.allocation = {s: w for s in symbols}

    def run(
        self,
        dfs:     Dict[str, pd.DataFrame],
        engine:  BacktestEngine,
        verbose: bool = True,
    ) -> PortfolioResult:
        """
        Run backtest for each symbol and combine.

        Parameters
        ----------
        dfs    : dict mapping symbol → DataFrame (with 'signal' column)
        engine : configured BacktestEngine
        """
        symbol_results: Dict[str, BacktestResult] = {}
        symbol_metrics: Dict[str, dict]           = {}
        equity_curves:  Dict[str, np.ndarray]     = {}

        for sym in self.symbols:
            if sym not in dfs:
                print(f"  [WARN] {sym} not in data — skipping")
                continue

            df    = dfs[sym]
            alloc = self.allocation.get(sym, 1.0 / len(self.symbols))

            # Scale capital by allocation weight
            orig_cap = engine.initial_capital
            engine.initial_capital = orig_cap * alloc

            try:
                result = engine.run(df)
                metrics = compute_metrics(result.equity_curve, result.trades)
                grade   = strategy_grade(metrics)

                symbol_results[sym] = result
                symbol_metrics[sym] = metrics
                equity_curves[sym]  = result.equity_curve

                if verbose:
                    print(f"  {sym:12s} | Return={metrics['total_return_pct']:+6.1f}% | "
                          f"Sharpe={metrics['sharpe_ratio']:5.2f} | "
                          f"MaxDD={metrics['max_drawdown_pct']:5.1f}% | "
                          f"Trades={metrics['n_trades']:4d} | Grade={grade}")
            except Exception as e:
                print(f"  [ERROR] {sym}: {e}")
            finally:
                engine.initial_capital = orig_cap

        if not symbol_results:
            raise RuntimeError("All symbol backtests failed")

        # ── Combine equity curves ─────────────────────────────────────────
        # Align all curves to same length (trim to shortest)
        min_len = min(len(v) for v in equity_curves.values())
        combined = np.zeros(min_len)
        for sym, curve in equity_curves.items():
            alloc = self.allocation.get(sym, 0.0)
            combined += curve[:min_len] * alloc  # weighted sum

        combined_metrics = compute_metrics(
            combined,
            [t for r in symbol_results.values() for t in r.trades],
        )
        portfolio_grade  = strategy_grade(combined_metrics)

        # ── Correlation matrix ────────────────────────────────────────────
        corr_matrix = None
        if len(equity_curves) > 1:
            returns_df = pd.DataFrame({
                sym: np.diff(curve[:min_len]) / (curve[:min_len][:-1] + 1e-10)
                for sym, curve in equity_curves.items()
            })
            corr_matrix = returns_df.corr().values

        if verbose:
            print(f"\n  PORTFOLIO COMBINED")
            print(f"  Return={combined_metrics['total_return_pct']:+6.1f}% | "
                  f"Sharpe={combined_metrics['sharpe_ratio']:5.2f} | "
                  f"MaxDD={combined_metrics['max_drawdown_pct']:5.1f}% | "
                  f"Grade={portfolio_grade}")

        return PortfolioResult(
            symbol_results   = symbol_results,
            symbol_metrics   = symbol_metrics,
            combined_equity  = combined,
            combined_metrics = combined_metrics,
            correlation_matrix = corr_matrix,
            portfolio_grade  = portfolio_grade,
        )

    def print_summary(self, result: PortfolioResult) -> None:
        """Print full portfolio summary."""
        print(f"\n{'='*60}")
        print(f"  PORTFOLIO BACKTEST SUMMARY")
        print(f"{'='*60}")
        for sym in result.symbol_results:
            m = result.symbol_metrics[sym]
            print(f"  {sym}: Return={m['total_return_pct']:+.1f}%  "
                  f"Sharpe={m['sharpe_ratio']:.2f}  "
                  f"WinRate={m['win_rate_pct']:.0f}%  "
                  f"PF={m['profit_factor']:.2f}  "
                  f"MaxDD={m['max_drawdown_pct']:.1f}%")

        print(f"\n  COMBINED PORTFOLIO:")
        print(f"  Grade: {result.portfolio_grade}")
        print_report(result.combined_metrics)

        if result.correlation_matrix is not None:
            print(f"  RETURN CORRELATIONS:")
            syms = list(result.symbol_results.keys())
            header = "        " + "  ".join(f"{s[:6]:>7}" for s in syms)
            print(f"  {header}")
            for i, sym in enumerate(syms):
                row = "  ".join(f"{result.correlation_matrix[i,j]:7.3f}" for j in range(len(syms)))
                print(f"  {sym[:6]:>6}: {row}")
