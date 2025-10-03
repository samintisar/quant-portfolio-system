"""
Demonstration of market regime detection and adaptive strategy selection.

This script shows how to use HMM-based regime detection to switch between
optimization methods based on market conditions.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.regime.detector import RegimeDetector
from portfolio.data.yahoo_service import YahooFinanceService
from portfolio.backtesting.walk_forward import WalkForwardBacktester, BacktestConfig

def main():
    """Run regime-adaptive backtest demonstration."""
    
    print("="*80)
    print("MARKET REGIME DETECTION AND ADAPTIVE STRATEGY DEMONSTRATION")
    print("="*80)
    
    # Configuration
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "MA"]
    results_dir = os.path.join("data", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Load SPY data for regime detection
    print("\nStep 1: Loading SPY benchmark data...")
    service = YahooFinanceService(use_offline_data=True, offline_data_dir="data")
    
    spy_data = service.fetch_historical_data('SPY', period='10y')
    spy_prices = spy_data['Adj Close']
    spy_returns = spy_prices.pct_change().dropna()
    print(f"✓ Loaded {len(spy_returns)} SPY returns")
    
    # Step 2: Fit regime detector
    print("\nStep 2: Training HMM regime detector...")
    detector = RegimeDetector(
        n_states=3,
        return_window=60,
        volatility_window=20,
        random_state=42
    )
    
    regimes = detector.fit_predict(spy_returns)
    print(f"✓ Detected {len(regimes)} regime observations")
    print("\nRegime Distribution:")
    for regime, count in regimes.value_counts().items():
        pct = count / len(regimes) * 100
        print(f"  {regime}: {count} days ({pct:.1f}%)")
    
    # Step 3: Analyze regime statistics
    print("\nStep 3: Regime Statistics:")
    regime_stats = detector.get_regime_statistics(regimes, spy_returns)
    print(regime_stats.round(4).to_string())
    
    # Step 4: Run backtests
    print("\nStep 4: Running backtests...")
    
    config = BacktestConfig(
        train_years=3,
        test_quarters=1,
        transaction_cost_bps=7.5,
        rebalance_frequency='quarterly'
    )
    
    backtester = WalkForwardBacktester(config)
    
    # Get date range from prices
    prices = service.fetch_price_data(symbols, period='10y')
    start_date = str(prices.index[0].date())
    end_date = str(prices.index[-1].date())
    
    print(f"Backtest period: {start_date} to {end_date}")
    
    # Run regime-adaptive backtest
    print("\nRunning regime-adaptive strategy...")
    result_adaptive = backtester.run_regime_adaptive_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        spy_returns=spy_returns,
        detector=detector,
        weight_cap=0.20
    )
    
    print(f"✓ Regime-Adaptive Sharpe: {result_adaptive.metrics['sharpe_ratio']:.3f}")
    
    # Step 5: Display results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    metrics = result_adaptive.metrics
    print(f"\nRegime-Adaptive Strategy Performance:")
    print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.3f}")
    print(f"  Annual Return:     {metrics['annual_return']*100:.2f}%")
    print(f"  Annual Volatility: {metrics['annual_volatility']*100:.2f}%")
    print(f"  Max Drawdown:      {metrics['max_drawdown']*100:.2f}%")
    print(f"  Sortino Ratio:     {metrics['sortino_ratio']:.3f}")
    print(f"  Calmar Ratio:      {metrics['calmar_ratio']:.3f}")
    
    # Step 6: Visualize
    print("\nStep 6: Creating visualizations...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Regime timeline
    ax1 = axes[0]
    colors = {'Bull': 'lightgreen', 'Bear': 'lightcoral', 'Sideways': 'lightyellow'}
    for regime, color in colors.items():
        mask = regimes == regime
        if mask.any():
            ax1.fill_between(
                range(len(regimes)),
                0, 1,
                where=(regimes == regime).values,
                color=color,
                alpha=0.5,
                label=regime
            )
    
    ax1.set_title('Market Regime Detection Timeline', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Regime')
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.grid(True, alpha=0.3)
    
    # Equity curve
    ax2 = axes[1]
    equity = result_adaptive.equity_curve
    ax2.plot(equity.values, linewidth=2, color='darkblue', label='Regime-Adaptive')
    ax2.set_title('Portfolio Equity Curve', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Portfolio Value ($1 invested)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(results_dir, 'regime_adaptive_demo.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    
    plt.show()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
