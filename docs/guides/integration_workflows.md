# Integration Workflows and End-to-End Examples

This comprehensive guide provides complete end-to-end workflows for integrating various components of the Quantitative Trading System.

## 🔄 Complete System Integration

### Architecture Overview

```
Data Sources → Data Ingestion → Preprocessing → Feature Generation → Portfolio Optimization → Risk Management → Execution
```

### 1. Full Pipeline Integration

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantitativeTradingSystem:
    """Complete quantitative trading system integration"""

    def __init__(self, config):
        self.config = config
        self.data_sources = self._initialize_data_sources()
        self.preprocessing_pipeline = self._initialize_preprocessing()
        self.feature_generator = self._initialize_feature_generation()
        self.portfolio_optimizer = self._initialize_portfolio_optimization()
        self.risk_manager = self._initialize_risk_management()
        self.execution_engine = self._initialize_execution_engine()

    def _initialize_data_sources(self):
        """Initialize data sources"""
        from data.src.feeds.yahoo_finance_ingestion import YahooFinanceIngestion
        from data.src.feeds.alpha_vantage_ingestion import AlphaVantageIngestion

        return {
            'yahoo': YahooFinanceIngestion(),
            'alpha_vantage': AlphaVantageIngestion(api_key=self.config.get('alpha_vantage_key'))
        }

    def _initialize_preprocessing(self):
        """Initialize preprocessing pipeline"""
        from data.src.preprocessing import PreprocessingOrchestrator
        from data.src.config.pipeline_config import PipelineConfigManager

        config_manager = PipelineConfigManager()
        return PreprocessingOrchestrator(config_manager)

    def _initialize_feature_generation(self):
        """Initialize feature generation"""
        from services.feature_service import FeatureGenerator
        return FeatureGenerator()

    def _initialize_portfolio_optimization(self):
        """Initialize portfolio optimization"""
        from portfolio.src.optimization import PortfolioOptimizer
        return PortfolioOptimizer()

    def _initialize_risk_management(self):
        """Initialize risk management"""
        from portfolio.src.risk import RiskManager
        return RiskManager()

    def _initialize_execution_engine(self):
        """Initialize execution engine"""
        from portfolio.src.execution import ExecutionEngine
        return ExecutionEngine()

    def run_complete_workflow(self, symbols, start_date, end_date):
        """Run complete end-to-end workflow"""
        logger.info(f"Starting complete workflow for {len(symbols)} symbols")

        # Step 1: Data Ingestion
        logger.info("Step 1: Data Ingestion")
        raw_data = self._ingest_data(symbols, start_date, end_date)

        # Step 2: Data Preprocessing
        logger.info("Step 2: Data Preprocessing")
        processed_data = self._preprocess_data(raw_data)

        # Step 3: Feature Generation
        logger.info("Step 3: Feature Generation")
        features = self._generate_features(processed_data)

        # Step 4: Portfolio Optimization
        logger.info("Step 4: Portfolio Optimization")
        optimal_portfolio = self._optimize_portfolio(features)

        # Step 5: Risk Management
        logger.info("Step 5: Risk Management")
        risk_assessment = self._assess_risk(optimal_portfolio, processed_data)

        # Step 6: Execution
        logger.info("Step 6: Execution")
        execution_results = self._execute_trades(optimal_portfolio)

        return {
            'raw_data': raw_data,
            'processed_data': processed_data,
            'features': features,
            'optimal_portfolio': optimal_portfolio,
            'risk_assessment': risk_assessment,
            'execution_results': execution_results
        }

    def _ingest_data(self, symbols, start_date, end_date):
        """Ingest data from multiple sources"""
        all_data = {}

        for symbol in symbols:
            try:
                # Try Yahoo Finance first
                data = self.data_sources['yahoo'].download_data(
                    symbols=[symbol],
                    start_date=start_date,
                    end_date=end_date
                )
                all_data[symbol] = data[symbol]
                logger.info(f"Downloaded {symbol} from Yahoo Finance")

            except Exception as e:
                logger.warning(f"Failed to download {symbol} from Yahoo Finance: {e}")
                try:
                    # Fallback to Alpha Vantage
                    data = self.data_sources['alpha_vantage'].download_data(
                        symbols=[symbol],
                        start_date=start_date,
                        end_date=end_date
                    )
                    all_data[symbol] = data[symbol]
                    logger.info(f"Downloaded {symbol} from Alpha Vantage")

                except Exception as e2:
                    logger.error(f"Failed to download {symbol} from all sources: {e2}")

        return all_data

    def _preprocess_data(self, raw_data):
        """Preprocess raw data"""
        processed_data = {}

        for symbol, data in raw_data.items():
            try:
                # Create pipeline configuration
                pipeline_config = {
                    'pipeline_id': f'{symbol}_preprocessing',
                    'description': f'Preprocessing pipeline for {symbol}',
                    'asset_classes': ['equity'],
                    'rules': [
                        {
                            'type': 'validation',
                            'conditions': [
                                {'field': 'close', 'operator': 'greater_than', 'value': 0},
                                {'field': 'volume', 'operator': 'greater_than', 'value': 0}
                            ],
                            'actions': [{'type': 'flag', 'severity': 'warning'}]
                        },
                        {
                            'type': 'cleaning',
                            'conditions': [{'field': 'close', 'operator': 'is_null'}],
                            'actions': [{'type': 'interpolate', 'method': 'linear'}]
                        }
                    ],
                    'quality_thresholds': {
                        'completeness': 0.95,
                        'accuracy': 0.90
                    }
                }

                # Process data
                result = self.preprocessing_pipeline.preprocess_data(
                    data=data,
                    pipeline_config=pipeline_config
                )

                processed_data[symbol] = result['processed_data']

            except Exception as e:
                logger.error(f"Failed to preprocess {symbol}: {e}")

        return processed_data

    def _generate_features(self, processed_data):
        """Generate features from processed data"""
        all_features = {}

        for symbol, data in processed_data.items():
            try:
                # Create feature generation configuration
                from services.feature_service import FeatureGenerationConfig
                feature_config = FeatureGenerationConfig(
                    return_periods=[1, 5, 21, 63],      # Daily, weekly, monthly, quarterly
                    volatility_windows=[5, 21, 63],    # Weekly, monthly, quarterly
                    momentum_periods=[5, 14, 21],     # Short, medium, long momentum
                    volume_periods=[5, 21],           # Volume analysis
                    price_levels=[10, 20, 50]         # Support/resistance levels
                )

                # Generate features
                features = self.feature_generator.generate_features(
                    price_data=data,
                    custom_config=feature_config
                )

                all_features[symbol] = features

            except Exception as e:
                logger.error(f"Failed to generate features for {symbol}: {e}")

        return all_features

    def _optimize_portfolio(self, features):
        """Optimize portfolio using features"""
        try:
            # Extract returns from features
            returns_data = {}
            for symbol, feature_set in features.items():
                if feature_set.has_feature('returns_1', symbol):
                    returns_data[symbol] = feature_set.get_feature('returns_1', symbol)

            returns_df = pd.DataFrame(returns_data).dropna()

            # Create optimization configuration
            from portfolio.src.optimization import OptimizationConfig
            opt_config = OptimizationConfig(
                objective='sharpe_ratio',
                risk_free_rate=0.02,
                constraints={
                    'min_weight': 0.02,
                    'max_weight': 0.20,
                    'min_positions': 5,
                    'max_positions': 15,
                    'max_volatility': 0.20,
                    'max_drawdown': 0.15,
                    'sector_limits': {
                        'Technology': 0.30,
                        'Healthcare': 0.20,
                        'Finance': 0.25,
                        'Consumer': 0.25
                    }
                },
                transaction_costs=0.001,
                risk_model='factor'
            )

            # Optimize portfolio
            optimal_portfolio = self.portfolio_optimizer.optimize_portfolio(
                returns=returns_df,
                config=opt_config
            )

            return optimal_portfolio

        except Exception as e:
            logger.error(f"Failed to optimize portfolio: {e}")
            return None

    def _assess_risk(self, portfolio, processed_data):
        """Assess portfolio risk"""
        try:
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(portfolio, processed_data)

            # Risk assessment
            risk_metrics = self.risk_manager.calculate_risk_metrics(
                returns=portfolio_returns,
                weights=portfolio.weights
            )

            # Stress testing
            stress_scenarios = {
                'market_crash': {'shock': -0.30, 'duration': 21},
                'volatility_spike': {'volatility_multiplier': 2.0, 'duration': 10},
                'liquidity_crisis': {'liquidity_shock': -0.50, 'duration': 5}
            }

            stress_results = self.risk_manager.run_stress_tests(
                portfolio=portfolio,
                returns=portfolio_returns,
                scenarios=stress_scenarios
            )

            return {
                'risk_metrics': risk_metrics,
                'stress_results': stress_results,
                'risk_level': self._determine_risk_level(risk_metrics)
            }

        except Exception as e:
            logger.error(f"Failed to assess risk: {e}")
            return None

    def _execute_trades(self, portfolio):
        """Execute trades for the portfolio"""
        try:
            # Get current positions (simplified)
            current_positions = self._get_current_positions()

            # Calculate required trades
            trades = self._calculate_trades(portfolio.weights, current_positions)

            # Execute trades
            execution_results = self.execution_engine.execute_trades(trades)

            return execution_results

        except Exception as e:
            logger.error(f"Failed to execute trades: {e}")
            return None

    def _calculate_portfolio_returns(self, portfolio, processed_data):
        """Calculate portfolio returns"""
        # Simplified implementation
        returns = []
        weights_dict = portfolio.weights

        for symbol, weight in weights_dict.items():
            if symbol in processed_data:
                symbol_returns = processed_data[symbol]['close'].pct_change().dropna()
                weighted_returns = symbol_returns * weight
                returns.append(weighted_returns)

        if returns:
            return pd.concat(returns, axis=1).sum(axis=1)
        else:
            return pd.Series()

    def _determine_risk_level(self, risk_metrics):
        """Determine risk level based on metrics"""
        volatility = risk_metrics.get('volatility', 0)
        var_95 = risk_metrics.get('var_95', 0)
        max_drawdown = risk_metrics.get('max_drawdown', 0)

        if volatility < 0.10 and var_95 > -0.15 and max_drawdown < 0.20:
            return 'low'
        elif volatility < 0.20 and var_95 > -0.25 and max_drawdown < 0.30:
            return 'medium'
        else:
            return 'high'

    def _get_current_positions(self):
        """Get current portfolio positions (simplified)"""
        # In a real implementation, this would query your broker or portfolio database
        return {}

    def _calculate_trades(self, target_weights, current_positions):
        """Calculate required trades to reach target weights"""
        trades = []

        for symbol, target_weight in target_weights.items():
            current_weight = current_positions.get(symbol, 0)
            weight_change = target_weight - current_weight

            if abs(weight_change) > 0.01:  # Only trade if change > 1%
                trades.append({
                    'symbol': symbol,
                    'action': 'buy' if weight_change > 0 else 'sell',
                    'weight_change': weight_change,
                    'priority': abs(weight_change)
                })

        # Sort by priority (largest changes first)
        trades.sort(key=lambda x: x['priority'], reverse=True)
        return trades


# Example Usage
if __name__ == "__main__":
    # Configuration
    config = {
        'alpha_vantage_key': 'your_alpha_vantage_key_here',
        'risk_profile': 'moderate',
        'rebalancing_frequency': 'monthly'
    }

    # Initialize system
    system = QuantitativeTradingSystem(config)

    # Define portfolio universe
    symbols = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META',
        'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
        'PG', 'UNH', 'HD', 'MA', 'PYPL'
    ]

    # Define time period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years

    # Run complete workflow
    results = system.run_complete_workflow(symbols, start_date, end_date)

    # Generate comprehensive report
    report = generate_comprehensive_report(results)
    print(report)
```

### 2. Automated Rebalancing Workflow

```python
import schedule
import time
from datetime import datetime, timedelta
import json
import logging

class AutomatedRebalancingSystem:
    """Automated portfolio rebalancing system"""

    def __init__(self, config):
        self.config = config
        self.trading_system = QuantitativeTradingSystem(config)
        self.rebalancing_schedule = config.get('rebalancing_schedule', 'monthly')
        self.performance_threshold = config.get('performance_threshold', 0.05)  # 5%
        self.setup_schedule()

    def setup_schedule(self):
        """Setup automated rebalancing schedule"""
        if self.rebalancing_schedule == 'daily':
            schedule.every().day.at("09:30").do(self.rebalance_portfolio)
        elif self.rebalancing_schedule == 'weekly':
            schedule.every().monday.at("09:30").do(self.rebalance_portfolio)
        elif self.rebalancing_schedule == 'monthly':
            schedule.every().month.at("09:30").do(self.rebalance_portfolio)
        elif self.rebalancing_schedule == 'quarterly':
            schedule.every(3).months.at("09:30").do(self.rebalance_portfolio)

    def rebalance_portfolio(self):
        """Execute portfolio rebalancing"""
        try:
            logger.info("Starting automated portfolio rebalancing")

            # Get current portfolio state
            current_state = self.get_current_portfolio_state()

            # Check if rebalancing is needed
            if self._needs_rebalancing(current_state):
                logger.info("Rebalancing triggered")

                # Run optimization
                optimal_portfolio = self._optimize_current_portfolio()

                # Execute rebalancing trades
                execution_results = self._execute_rebalancing(optimal_portfolio)

                # Update portfolio state
                self._update_portfolio_state(optimal_portfolio)

                # Generate rebalancing report
                report = self._generate_rebalancing_report(
                    current_state, optimal_portfolio, execution_results
                )

                # Send notifications
                self._send_notifications(report)

                logger.info("Portfolio rebalancing completed successfully")
            else:
                logger.info("No rebalancing needed")

        except Exception as e:
            logger.error(f"Portfolio rebalancing failed: {e}")
            self._send_error_alert(e)

    def get_current_portfolio_state(self):
        """Get current portfolio state"""
        # This would typically query your portfolio database
        return {
            'positions': self._get_current_positions(),
            'cash_balance': self._get_cash_balance(),
            'total_value': self._get_portfolio_value(),
            'last_rebalanced': self._get_last_rebalance_date(),
            'performance_since_rebalance': self._calculate_performance_since_rebalance()
        }

    def _needs_rebalancing(self, current_state):
        """Determine if rebalancing is needed"""
        # Check time-based rebalancing
        if self._time_for_rebalancing(current_state['last_rebalanced']):
            return True

        # Check performance-based rebalancing
        if abs(current_state['performance_since_rebalance']) > self.performance_threshold:
            return True

        # Check weight deviation
        current_weights = self._calculate_current_weights(current_state['positions'])
        optimal_weights = self._get_optimal_weights()

        weight_deviation = self._calculate_weight_deviation(current_weights, optimal_weights)
        if weight_deviation > 0.05:  # 5% threshold
            return True

        return False

    def _optimize_current_portfolio(self):
        """Optimize current portfolio"""
        # Get current portfolio universe
        symbols = self._get_portfolio_universe()

        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 years

        # Run optimization
        results = self.trading_system.run_complete_workflow(symbols, start_date, end_date)
        return results['optimal_portfolio']

    def _execute_rebalancing(self, optimal_portfolio):
        """Execute rebalancing trades"""
        # Calculate required trades
        current_positions = self._get_current_positions()
        trades = self._calculate_rebalancing_trades(
            optimal_portfolio.weights, current_positions
        )

        # Execute trades with risk management
        execution_results = []
        for trade in trades:
            try:
                # Apply risk management rules
                if self._passes_risk_checks(trade):
                    result = self.trading_system.execution_engine.execute_trade(trade)
                    execution_results.append(result)
                else:
                    logger.warning(f"Trade {trade} failed risk checks")
            except Exception as e:
                logger.error(f"Failed to execute trade {trade}: {e}")

        return execution_results

    def _generate_rebalancing_report(self, current_state, optimal_portfolio, execution_results):
        """Generate comprehensive rebalancing report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': current_state['total_value'],
            'previous_weights': self._calculate_current_weights(current_state['positions']),
            'new_weights': optimal_portfolio.weights,
            'trades_executed': execution_results,
            'estimated_transaction_costs': self._calculate_transaction_costs(execution_results),
            'expected_performance_improvement': self._estimate_performance_improvement(
                current_state, optimal_portfolio
            )
        }
        return report

    def _send_notifications(self, report):
        """Send rebalancing notifications"""
        # Email notification
        if self.config.get('email_notifications'):
            self._send_email_notification(report)

        # Slack notification
        if self.config.get('slack_notifications'):
            self._send_slack_notification(report)

        # Webhook notification
        if self.config.get('webhook_url'):
            self._send_webhook_notification(report)

    def run_scheduler(self):
        """Run the rebalancing scheduler"""
        logger.info(f"Starting automated rebalancing scheduler ({self.rebalancing_schedule})")
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


# Real-time Market Data Integration
class RealTimeDataIntegration:
    """Real-time market data integration system"""

    def __init__(self, config):
        self.config = config
        self.data_sources = self._initialize_real_time_sources()
        self.message_queue = []
        self.subscribers = []
        self.is_running = False

    def _initialize_real_time_sources(self):
        """Initialize real-time data sources"""
        sources = {}

        # WebSocket connection for real-time data
        if self.config.get('websocket_url'):
            sources['websocket'] = self._setup_websocket_connection()

        # Alpha Vantage real-time API
        if self.config.get('alpha_vantage_key'):
            sources['alpha_vantage'] = self._setup_alpha_vantage_realtime()

        # Yahoo Finance real-time
        sources['yahoo'] = self._setup_yahoo_realtime()

        return sources

    def start_real_time_feed(self):
        """Start real-time data feed"""
        self.is_running = True

        # Start each data source
        for source_name, source in self.data_sources.items():
            try:
                source.start()
                logger.info(f"Started real-time feed from {source_name}")
            except Exception as e:
                logger.error(f"Failed to start {source_name}: {e}")

        # Start processing loop
        self._start_processing_loop()

    def stop_real_time_feed(self):
        """Stop real-time data feed"""
        self.is_running = False

        # Stop each data source
        for source_name, source in self.data_sources.items():
            try:
                source.stop()
                logger.info(f"Stopped real-time feed from {source_name}")
            except Exception as e:
                logger.error(f"Failed to stop {source_name}: {e}")

    def subscribe_to_updates(self, callback):
        """Subscribe to real-time data updates"""
        self.subscribers.append(callback)

    def _start_processing_loop(self):
        """Start processing real-time data"""
        while self.is_running:
            try:
                # Get data from sources
                for source_name, source in self.data_sources.items():
                    data = source.get_latest_data()
                    if data:
                        self._process_real_time_data(source_name, data)

                # Process messages
                self._process_message_queue()

                # Sleep briefly
                time.sleep(0.001)  # 1ms

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")

    def _process_real_time_data(self, source_name, data):
        """Process real-time data from a source"""
        # Validate and normalize data
        normalized_data = self._normalize_real_time_data(source_name, data)

        # Add to message queue
        self.message_queue.append({
            'source': source_name,
            'data': normalized_data,
            'timestamp': datetime.now().isoformat()
        })

    def _process_message_queue(self):
        """Process messages in the queue"""
        while self.message_queue:
            message = self.message_queue.pop(0)

            # Notify subscribers
            for callback in self.subscribers:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")

    def _normalize_real_time_data(self, source_name, data):
        """Normalize real-time data from different sources"""
        # Convert to standard format
        normalized = {
            'symbol': data.get('symbol'),
            'price': data.get('price'),
            'volume': data.get('volume'),
            'timestamp': data.get('timestamp'),
            'source': source_name,
            'data_type': 'tick'
        }

        # Add calculated fields
        if 'bid' in data and 'ask' in data:
            normalized['spread'] = data['ask'] - data['bid']
            normalized['mid_price'] = (data['bid'] + data['ask']) / 2

        return normalized


# Example: Complete Automated Trading System
class AutomatedTradingSystem:
    """Complete automated trading system integration"""

    def __init__(self, config):
        self.config = config
        self.rebalancing_system = AutomatedRebalancingSystem(config)
        self.real_time_system = RealTimeDataIntegration(config)
        self.risk_monitoring = RiskMonitoringSystem(config)
        self.performance_tracking = PerformanceTrackingSystem(config)

    def start(self):
        """Start the automated trading system"""
        logger.info("Starting automated trading system")

        # Start real-time data feed
        self.real_time_system.start_real_time_feed()

        # Subscribe to real-time updates
        self.real_time_system.subscribe_to_updates(self._handle_real_time_update)

        # Start rebalancing system
        self.rebalancing_system.run_scheduler()

        # Start risk monitoring
        self.risk_monitoring.start_monitoring()

        # Start performance tracking
        self.performance_tracking.start_tracking()

        logger.info("Automated trading system started successfully")

    def stop(self):
        """Stop the automated trading system"""
        logger.info("Stopping automated trading system")

        # Stop all systems
        self.real_time_system.stop_real_time_feed()
        self.risk_monitoring.stop_monitoring()
        self.performance_tracking.stop_tracking()

        logger.info("Automated trading system stopped")

    def _handle_real_time_update(self, message):
        """Handle real-time market updates"""
        # Process real-time data
        processed_data = self._process_real_time_message(message)

        # Update risk monitoring
        self.risk_monitoring.update_with_real_time_data(processed_data)

        # Update performance tracking
        self.performance_tracking.update_with_real_time_data(processed_data)

        # Check for trading signals
        signals = self._generate_trading_signals(processed_data)

        # Execute trades if signals detected
        if signals:
            self._execute_trading_signals(signals)

    def _generate_trading_signals(self, real_time_data):
        """Generate trading signals from real-time data"""
        signals = []

        # Check for price movements
        if self._detect_significant_price_movement(real_time_data):
            signals.append({
                'type': 'price_movement',
                'symbol': real_time_data['symbol'],
                'action': 'review',
                'urgency': 'medium'
            })

        # Check for volume spikes
        if self._detect_volume_spike(real_time_data):
            signals.append({
                'type': 'volume_spike',
                'symbol': real_time_data['symbol'],
                'action': 'review',
                'urgency': 'high'
            })

        # Check for technical indicators
        if self._detect_technical_signal(real_time_data):
            signals.append({
                'type': 'technical_signal',
                'symbol': real_time_data['symbol'],
                'action': 'consider_trade',
                'urgency': 'medium'
            })

        return signals


# Helper Functions
def generate_comprehensive_report(results):
    """Generate comprehensive workflow report"""
    report = f"""
Comprehensive Quantitative Trading System Report
================================================

Workflow Summary:
- Symbols Processed: {len(results.get('raw_data', {}))}
- Successfully Preprocessed: {len(results.get('processed_data', {}))}
- Features Generated: {len(results.get('features', {}))}
- Portfolio Optimized: {results.get('optimal_portfolio') is not None}
- Risk Assessed: {results.get('risk_assessment') is not None}
- Trades Executed: {len(results.get('execution_results', []))}

Optimal Portfolio:
"""

    if results.get('optimal_portfolio'):
        for symbol, weight in results['optimal_portfolio'].weights.items():
            report += f"- {symbol}: {weight:.2%}\n"

    if results.get('risk_assessment'):
        risk = results['risk_assessment']
        report += f"""
Risk Assessment:
- Risk Level: {risk.get('risk_level', 'unknown')}
- Portfolio Volatility: {risk.get('risk_metrics', {}).get('volatility', 0):.2%}
- Value at Risk (95%): {risk.get('risk_metrics', {}).get('var_95', 0):.2%}
- Maximum Drawdown: {risk.get('risk_metrics', {}).get('max_drawdown', 0):.2%}
"""

    report += f"""
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    return report


# Example Configuration
SYSTEM_CONFIG = {
    'alpha_vantage_key': 'your_alpha_vantage_key_here',
    'rebalancing_schedule': 'monthly',
    'performance_threshold': 0.05,
    'risk_limits': {
        'max_position_size': 0.20,
        'max_portfolio_volatility': 0.20,
        'max_drawdown': 0.15
    },
    'notifications': {
        'email': True,
        'slack': True,
        'webhook_url': 'https://your-webhook-url.com'
    },
    'trading': {
        'broker_api_key': 'your_broker_key',
        'paper_trading': True,  # Set to False for live trading
        'max_trades_per_day': 10
    }
}

if __name__ == "__main__":
    # Example: Run complete workflow
    config = SYSTEM_CONFIG

    # Initialize and run trading system
    trading_system = AutomatedTradingSystem(config)

    try:
        trading_system.start()
        # Keep running indefinitely
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down trading system...")
        trading_system.stop()
        logger.info("Trading system stopped")
```

## 🔄 API Integration Workflows

### 1. REST API Integration

```python
import requests
import json
from datetime import datetime

class APIIntegrationWorkflow:
    """REST API integration workflow"""

    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })

    def complete_api_workflow(self, symbols):
        """Complete workflow using REST APIs"""
        workflow_id = self._start_workflow(symbols)
        self._monitor_workflow(workflow_id)
        results = self._get_workflow_results(workflow_id)
        return results

    def _start_workflow(self, symbols):
        """Start new workflow via API"""
        url = f"{self.base_url}/workflows"
        payload = {
            'symbols': symbols,
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'config': {
                'risk_profile': 'moderate',
                'rebalancing_frequency': 'monthly'
            }
        }

        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()['workflow_id']

    def _monitor_workflow(self, workflow_id):
        """Monitor workflow progress"""
        url = f"{self.base_url}/workflows/{workflow_id}/status"

        while True:
            response = self.session.get(url)
            status = response.json()

            if status['status'] in ['completed', 'failed']:
                break

            print(f"Workflow progress: {status['progress']:.1%}")
            time.sleep(5)

    def _get_workflow_results(self, workflow_id):
        """Get workflow results"""
        url = f"{self.base_url}/workflows/{workflow_id}/results"
        response = self.session.get(url)
        return response.json()
```

### 2. WebSocket Integration

```python
import asyncio
import websockets
import json

class WebSocketIntegration:
    """WebSocket integration for real-time updates"""

    def __init__(self, ws_url, api_key):
        self.ws_url = ws_url
        self.api_key = api_key
        self.websocket = None

    async def connect(self):
        """Connect to WebSocket"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        self.websocket = await websockets.connect(self.ws_url, extra_headers=headers)

    async def subscribe_to_market_data(self, symbols):
        """Subscribe to market data for symbols"""
        subscribe_message = {
            'action': 'subscribe',
            'symbols': symbols,
            'data_type': 'market_data'
        }
        await self.websocket.send(json.dumps(subscribe_message))

    async def listen_for_updates(self, callback):
        """Listen for real-time updates"""
        async for message in self.websocket:
            data = json.loads(message)
            await callback(data)

    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
```

## 📊 Monitoring and Alerting

### 1. System Monitoring Dashboard

```python
class SystemMonitor:
    """System monitoring and alerting"""

    def __init__(self, config):
        self.config = config
        self.metrics = {}
        self.alerts = []
        self.alert_thresholds = config.get('alert_thresholds', {})

    def monitor_system_health(self):
        """Monitor overall system health"""
        health_metrics = {
            'data_processing': self._monitor_data_processing(),
            'feature_generation': self._monitor_feature_generation(),
            'portfolio_optimization': self._monitor_portfolio_optimization(),
            'risk_management': self._monitor_risk_management(),
            'execution_engine': self._monitor_execution_engine()
        }

        self._check_health_alerts(health_metrics)
        return health_metrics

    def _monitor_data_processing(self):
        """Monitor data processing health"""
        return {
            'status': 'healthy',
            'processing_time_avg': 2.5,
            'error_rate': 0.02,
            'queue_size': 150,
            'throughput': 1000  # records per second
        }

    def _check_health_alerts(self, metrics):
        """Check for health alerts"""
        for component, metrics in metrics.items():
            if metrics['status'] != 'healthy':
                self._create_alert(
                    component, 'health_check',
                    f"{component} is not healthy: {metrics['status']}",
                    'critical'
                )

            # Check specific thresholds
            if 'error_rate' in metrics and metrics['error_rate'] > 0.1:
                self._create_alert(
                    component, 'high_error_rate',
                    f"High error rate in {component}: {metrics['error_rate']:.2%}",
                    'warning'
                )
```

This comprehensive integration workflow guide provides complete end-to-end examples for building and deploying automated quantitative trading systems. The examples cover everything from basic data processing to advanced automated trading with real-time monitoring and risk management.

---

*Last Updated: 2024-01-15*