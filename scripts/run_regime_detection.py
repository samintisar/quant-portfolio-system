#!/usr/bin/env python
"""
Script to run regime detection analysis and generate visualization.

Usage:
    python scripts/run_regime_detection.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.regime.detector import RegimeDetector
from portfolio.data.yahoo_service import YahooFinanceService
from portfolio.backtesting.walk_forward import WalkForwardBacktester


def main():
    """Run regime detection analysis."""
    
    print("="*80)
    print(" MARKET REGIME DETECTION ANALYSIS ".center(80))
    print("="*80)
    print()
    
    # Configuration
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN",
        "META", "NVDA", "NFLX", "TSLA",
        "JPM", "JNJ", "V", "MA"
    ]
    
    results_dir = os.path.join("data", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Load data
    print("Step 1: Loading market data...")
    service = YahooFinanceService(use_offline_data=True, offline_data_dir="data")
    
    spy_data = service.fetch_historical_data('SPY', period='10y')
    spy_prices = spy_data['Adj Close']
    spy_returns = spy_prices.pct_change().dropna()
    print(f"  ✓ Loaded {len(spy_returns)} SPY returns from {spy_returns.index[0].date()} to {spy_returns.index[-1].date()}")
    
    # Step 2: Fit regime detector
    print("\nStep 2: Training HMM regime detector...")
    detector = RegimeDetector(
        n_states=3,
        return_window=60,
        volatility_window=20,
        random_state=42
    )
    
    regimes = detector.fit_predict(spy_returns)
    print(f"  ✓ Detected {len(regimes)} regime observations")
    
    # Step 3: Analyze regimes
    print("\nStep 3: Regime Analysis")
    print("-" * 80)
    
    # Distribution
    print("\nRegime Distribution:")
    for regime in ['Bull', 'Bear', 'Sideways']:
        count = (regimes == regime).sum()
        pct = count / len(regimes) * 100
        print(f"  {regime:10s}: {count:5d} days ({pct:5.1f}%)")
    
    # Statistics
    print("\nRegime Statistics:")
    stats = detector.get_regime_statistics(regimes, spy_returns)
    print(stats.to_string())
    
    # Transitions
    print("\nRegime Transition Matrix:")
    transitions = detector.get_regime_transition_matrix(regimes)
    print(transitions.round(3).to_string())
    
    # Save statistics
    stats.to_csv(os.path.join(results_dir, "regime_statistics.csv"))
    transitions.to_csv(os.path.join(results_dir, "regime_transitions.csv"))
    print(f"\n  ✓ Saved statistics to {results_dir}")
    
    # Step 4: Run backtests
    print("\nStep 4: Running Backtests")
    print("-" * 80)
    
    backtester = WalkForwardBacktester(
        train_period="3y",
        test_period="3mo",
        transaction_cost=0.00075
    )
    
    # Get date range
    prices = service.fetch_price_data(symbols, period='10y')
    start_date = str(prices.index[0].date())
    end_date = str(prices.index[-1].date())
    
    print(f"\nBacktest Period: {start_date} to {end_date}")
    print(f"Symbols: {', '.join(symbols)}")
    
    # Regime-adaptive strategy
    print("\n[1/2] Running regime-adaptive strategy...")
    result_adaptive = backtester.run_regime_adaptive_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        spy_returns=spy_returns,
        detector=detector,
        weight_cap=0.20
    )
    
    # Fixed MVO strategy for comparison
    print("[2/2] Running fixed MVO strategy...")
    results_fixed = backtester.run_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
    result_mvo = results_fixed['optimized']
    
    # Step 5: Compare results
    print("\nStep 5: Performance Comparison")
    print("-" * 80)
    
    comparison = pd.DataFrame({
        'Regime-Adaptive': {
            'Sharpe': result_adaptive.metrics['sharpe_ratio'],
            'Return (%)': result_adaptive.metrics['annual_return'] * 100,
            'Volatility (%)': result_adaptive.metrics['annual_volatility'] * 100,
            'Max DD (%)': result_adaptive.metrics['max_drawdown'] * 100,
            'Sortino': result_adaptive.metrics.get('sortino_ratio', 0),
            'Calmar': result_adaptive.metrics.get('calmar_ratio', 0)
        },
        'Fixed MVO': {
            'Sharpe': result_mvo.metrics['sharpe_ratio'],
            'Return (%)': result_mvo.metrics['annual_return'] * 100,
            'Volatility (%)': result_mvo.metrics['annual_volatility'] * 100,
            'Max DD (%)': result_mvo.metrics['max_drawdown'] * 100,
            'Sortino': result_mvo.metrics.get('sortino_ratio', 0),
            'Calmar': result_mvo.metrics.get('calmar_ratio', 0)
        }
    }).T
    
    print("\n" + comparison.round(3).to_string())
    
    # Calculate improvement
    sharpe_improvement = (
        (result_adaptive.metrics['sharpe_ratio'] - result_mvo.metrics['sharpe_ratio']) 
        / abs(result_mvo.metrics['sharpe_ratio']) * 100
    )
    
    print(f"\n{'='*80}")
    print(f"Sharpe Ratio Improvement: {sharpe_improvement:+.1f}%")
    print(f"{'='*80}")
    
    # Save comparison
    comparison.to_csv(os.path.join(results_dir, "regime_adaptive_comparison.csv"))
    
    # Step 6: Create visualizations
    print("\nStep 6: Creating Visualizations")
    print("-" * 80)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Regime Timeline
    ax1 = fig.add_subplot(gs[0, :])
    colors = {'Bull': 'lightgreen', 'Bear': 'lightcoral', 'Sideways': 'lightyellow'}
    
    for regime, color in colors.items():
        mask = regimes == regime
        if mask.any():
            ax1.fill_between(
                range(len(regimes)),
                0, 1,
                where=(regimes == regime).values,
                color=color,
                alpha=0.6,
                label=regime
            )
    
    ax1.set_title('Market Regime Detection Timeline (HMM 3-State Model)', 
                  fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel('Regime', fontweight='bold')
    ax1.set_xlim(0, len(regimes))
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Equity Curves
    ax2 = fig.add_subplot(gs[1, :])
    
    equity_adaptive = result_adaptive.equity_curve
    equity_mvo = result_mvo.equity_curve
    
    ax2.plot(equity_adaptive.values, linewidth=2.5, color='darkblue', 
             label=f'Regime-Adaptive (Sharpe: {result_adaptive.metrics["sharpe_ratio"]:.2f})', alpha=0.9)
    ax2.plot(equity_mvo.values, linewidth=2, color='steelblue', 
             label=f'Fixed MVO (Sharpe: {result_mvo.metrics["sharpe_ratio"]:.2f})', alpha=0.7)
    
    ax2.set_title('Cumulative Returns: Regime-Adaptive vs Fixed MVO', 
                  fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Trading Days', fontweight='bold')
    ax2.set_ylabel('Portfolio Value ($1 invested)', fontweight='bold')
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Regime Statistics (Bar Chart)
    ax3 = fig.add_subplot(gs[2, 0])
    
    x = np.arange(len(stats))
    width = 0.25
    
    ax3.bar(x - width, stats['mean_return'] * 100, width, label='Annualized Return (%)', alpha=0.8)
    ax3.bar(x, stats['volatility'] * 100, width, label='Volatility (%)', alpha=0.8)
    ax3.bar(x + width, stats['sharpe'], width, label='Sharpe Ratio', alpha=0.8)
    
    ax3.set_title('Regime Characteristics', fontsize=12, fontweight='bold', pad=10)
    ax3.set_ylabel('Value', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(stats.index)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Performance Metrics Comparison
    ax4 = fig.add_subplot(gs[2, 1])
    
    metrics_to_plot = ['Sharpe', 'Sortino', 'Calmar']
    adaptive_vals = [comparison.loc['Regime-Adaptive', m] for m in metrics_to_plot]
    mvo_vals = [comparison.loc['Fixed MVO', m] for m in metrics_to_plot]
    
    x2 = np.arange(len(metrics_to_plot))
    width2 = 0.35
    
    ax4.bar(x2 - width2/2, adaptive_vals, width2, label='Regime-Adaptive', 
            color='darkblue', alpha=0.8)
    ax4.bar(x2 + width2/2, mvo_vals, width2, label='Fixed MVO', 
            color='steelblue', alpha=0.8)
    
    ax4.set_title('Risk-Adjusted Performance Metrics', fontsize=12, fontweight='bold', pad=10)
    ax4.set_ylabel('Ratio', fontweight='bold')
    ax4.set_xticks(x2)
    ax4.set_xticklabels(metrics_to_plot)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (a, m) in enumerate(zip(adaptive_vals, mvo_vals)):
        ax4.text(i - width2/2, a + 0.05, f'{a:.2f}', ha='center', va='bottom', fontsize=9)
        ax4.text(i + width2/2, m + 0.05, f'{m:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Save figure
    output_path = os.path.join(results_dir, 'regime_detection_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved visualization to {output_path}")
    
    # Show plot
    try:
        plt.show()
    except Exception:
        pass
    
    # Summary
    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE ".center(80))
    print("="*80)
    print("\nGenerated Files:")
    print(f"  • {os.path.join(results_dir, 'regime_statistics.csv')}")
    print(f"  • {os.path.join(results_dir, 'regime_transitions.csv')}")
    print(f"  • {os.path.join(results_dir, 'regime_adaptive_comparison.csv')}")
    print(f"  • {os.path.join(results_dir, 'regime_detection_analysis.png')}")
    
    print("\nKey Findings:")
    print(f"  • Detected {len(regimes.unique())} distinct market regimes")
    print(f"  • Regime-Adaptive Sharpe: {result_adaptive.metrics['sharpe_ratio']:.3f}")
    print(f"  • Fixed MVO Sharpe: {result_mvo.metrics['sharpe_ratio']:.3f}")
    print(f"  • Improvement: {sharpe_improvement:+.1f}%")
    
    if sharpe_improvement > 0:
        print("\n  ✓ Regime-adaptive strategy successfully outperformed fixed MVO!")
    
    print()


if __name__ == "__main__":
    main()
