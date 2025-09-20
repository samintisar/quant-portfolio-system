"""
Extreme event monitoring and alerting system.

Implements comprehensive extreme event detection and alerting including:
- Real-time market extreme event detection
- Multi-indicator event correlation
- Automated alert generation and notification
- Event impact analysis and risk assessment
- Historical event pattern analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
import asyncio
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import json
from collections import deque, defaultdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import time
from scipy import stats


class EventSeverity(Enum):
    """Event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(Enum):
    """Types of extreme events."""
    PRICE_SPIKE = "price_spike"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    MARKET_CRASH = "market_crash"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    REGIME_SHIFT = "regime_shift"
    BLACK_SWAN = "black_swan"


class AlertChannel(Enum):
    """Alert notification channels."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    SMS = "sms"
    CONSOLE = "console"


@dataclass
class ExtremeEvent:
    """Extreme event detection result."""
    event_id: str
    event_type: EventType
    severity: EventSeverity
    timestamp: str
    symbol: str
    market_indicators: Dict[str, float]
    statistical_significance: float
    impact_score: float
    context: Dict[str, Any]
    affected_assets: List[str]
    recommended_actions: List[str]


@dataclass
class AlertNotification:
    """Alert notification configuration."""
    channel: AlertChannel
    recipients: List[str]
    enabled: bool
    severity_threshold: EventSeverity
    message_template: str


@dataclass
class EventPattern:
    """Historical event pattern for comparison."""
    pattern_id: str
    event_type: EventType
    characteristics: Dict[str, Any]
    typical_duration: int
    typical_impact: float
    historical_frequency: float


class ExtremeEventDetector(ABC):
    """Abstract base class for extreme event detectors."""

    @abstractmethod
    def detect_events(self, data: pd.DataFrame, window_size: int = 30) -> List[ExtremeEvent]:
        """Detect extreme events in data."""
        pass


class StatisticalEventDetector(ExtremeEventDetector):
    """Statistical extreme event detection using z-scores and percentiles."""

    def __init__(self, z_threshold: float = 3.0, percentile_threshold: float = 99.0):
        self.z_threshold = z_threshold
        self.percentile_threshold = percentile_threshold
        self.logger = logging.getLogger(__name__)

    def detect_events(self, data: pd.DataFrame, window_size: int = 30) -> List[ExtremeEvent]:
        """Detect events using statistical methods."""
        events = []

        if 'Close' not in data.columns:
            return events

        # Calculate returns
        returns = data['Close'].pct_change().dropna()

        # Rolling statistics
        rolling_mean = returns.rolling(window=window_size).mean()
        rolling_std = returns.rolling(window=window_size).std()

        # Percentile-based detection
        rolling_percentile = returns.rolling(window=window_size).quantile(self.percentile_threshold / 100)

        for i in range(window_size, len(returns)):
            date = returns.index[i]
            current_return = returns.iloc[i]
            current_z = (current_return - rolling_mean.iloc[i]) / rolling_std.iloc[i]

            # Check thresholds
            if abs(current_z) > self.z_threshold or abs(current_return) > abs(rolling_percentile.iloc[i]):
                event = self._create_event_from_stats(
                    date, current_return, current_z, EventType.PRICE_SPIKE, data
                )
                if event:
                    events.append(event)

        return events

    def _create_event_from_stats(self,
                                date: datetime,
                                return_value: float,
                                z_score: float,
                                event_type: EventType,
                                data: pd.DataFrame) -> Optional[ExtremeEvent]:
        """Create event from statistical analysis."""
        try:
            # Determine severity
            if abs(z_score) > 5:
                severity = EventSeverity.CRITICAL
            elif abs(z_score) > 4:
                severity = EventSeverity.HIGH
            elif abs(z_score) > 3:
                severity = EventSeverity.MEDIUM
            else:
                severity = EventSeverity.LOW

            # Calculate impact score
            impact_score = min(1.0, abs(z_score) / 5.0)

            # Get market context
            market_indicators = self._get_market_indicators(data, date)

            return ExtremeEvent(
                event_id=f"stat_{int(time.time())}_{hash(str(date)) % 10000}",
                event_type=event_type,
                severity=severity,
                timestamp=date.isoformat(),
                symbol=data.get('symbol', 'UNKNOWN'),
                market_indicators=market_indicators,
                statistical_significance=min(1.0, abs(z_score) / 10.0),
                impact_score=impact_score,
                context={
                    'detection_method': 'statistical',
                    'z_score': z_score,
                    'return_value': return_value,
                    'window_size': 30
                },
                affected_assets=[data.get('symbol', 'UNKNOWN')],
                recommended_actions=self._get_recommendations(severity, event_type)
            )

        except Exception as e:
            self.logger.error(f"Failed to create event from stats: {e}")
            return None

    def _get_market_indicators(self, data: pd.DataFrame, date: datetime) -> Dict[str, float]:
        """Get market indicators for the event date."""
        try:
            date_idx = data.index.get_loc(date)
            start_idx = max(0, date_idx - 5)
            end_idx = min(len(data), date_idx + 5)

            window_data = data.iloc[start_idx:end_idx]

            indicators = {
                'price_change': (window_data['Close'].iloc[-1] - window_data['Close'].iloc[0]) / window_data['Close'].iloc[0],
                'volatility': window_data['Close'].pct_change().std(),
                'volume_change': 0.0
            }

            if 'Volume' in window_data.columns:
                avg_volume = window_data['Volume'].iloc[:-1].mean()
                current_volume = window_data['Volume'].iloc[-1]
                indicators['volume_change'] = (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0.0

            return indicators

        except Exception:
            return {}

    def _get_recommendations(self, severity: EventSeverity, event_type: EventType) -> List[str]:
        """Get recommended actions for the event."""
        base_recommendations = [
            "Monitor market conditions closely",
            "Review portfolio exposure",
            "Assess risk management measures"
        ]

        if severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]:
            base_recommendations.extend([
                "Consider reducing position sizes",
                "Implement protective measures",
                "Alert risk management team"
            ])

        if event_type == EventType.MARKET_CRASH:
            base_recommendations.extend([
                "Activate crisis protocols",
                "Consider hedging strategies",
                "Prepare for volatility"
            ])

        return base_recommendations


class CorrelationEventDetector(ExtremeEventDetector):
    """Detect correlation breakdown events."""

    def __init__(self, correlation_threshold: float = 0.3, window_size: int = 20):
        self.correlation_threshold = correlation_threshold
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)

    def detect_events(self, data: pd.DataFrame, window_size: int = 30) -> List[ExtremeEvent]:
        """Detect correlation breakdown events."""
        events = []

        if len(data.columns) < 2:
            return events

        # Calculate rolling correlation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return events

        # Use first two numeric columns for correlation analysis
        col1, col2 = numeric_cols[0], numeric_cols[1]

        rolling_correlation = data[col1].rolling(self.window_size).corr(data[col2])

        # Detect correlation breakdowns
        for i in range(self.window_size, len(rolling_correlation)):
            current_corr = rolling_correlation.iloc[i]
            prev_corr = rolling_correlation.iloc[i-1]

            # Sudden change in correlation
            if abs(current_corr - prev_corr) > self.correlation_threshold:
                event = ExtremeEvent(
                    event_id=f"corr_{int(time.time())}_{i}",
                    event_type=EventType.CORRELATION_BREAKDOWN,
                    severity=EventSeverity.MEDIUM,
                    timestamp=data.index[i].isoformat(),
                    symbol="MARKET",
                    market_indicators={
                        'correlation_change': abs(current_corr - prev_corr),
                        'current_correlation': current_corr,
                        'previous_correlation': prev_corr
                    },
                    statistical_significance=min(1.0, abs(current_corr - prev_corr) * 2),
                    impact_score=min(1.0, abs(current_corr - prev_corr) * 3),
                    context={
                        'detection_method': 'correlation',
                        'asset1': col1,
                        'asset2': col2,
                        'window_size': self.window_size
                    },
                    affected_assets=[col1, col2],
                    recommended_actions=[
                        "Review correlation assumptions",
                        "Check for market regime changes",
                        "Update hedging strategies"
                    ]
                )
                events.append(event)

        return events


class ExtremeEventMonitoringService:
    """Main extreme event monitoring and alerting service."""

    def __init__(self,
                 alert_config: Optional[Dict[str, AlertNotification]] = None,
                 monitoring_interval: float = 60.0):
        self.alert_config = alert_config or self._default_alert_config()
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(__name__)

        # Detectors
        self.detectors = [
            StatisticalEventDetector(),
            CorrelationEventDetector()
        ]

        # Event storage
        self.event_history = deque(maxlen=10000)
        self.active_events = {}
        self.event_patterns = self._load_historical_patterns()

        # Monitoring state
        self.monitoring_active = False
        self.last_check_time = None

    def _default_alert_config(self) -> Dict[str, AlertNotification]:
        """Default alert configuration."""
        return {
            'console': AlertNotification(
                channel=AlertChannel.CONSOLE,
                recipients=['*'],
                enabled=True,
                severity_threshold=EventSeverity.MEDIUM,
                message_template="EXTREME EVENT: {event_type} - {severity} - {symbol}"
            ),
            'email': AlertNotification(
                channel=AlertChannel.EMAIL,
                recipients=['alerts@example.com'],
                enabled=False,
                severity_threshold=EventSeverity.HIGH,
                message_template="Extreme Event Alert: {event_type} detected in {symbol}"
            )
        }

    def _load_historical_patterns(self) -> List[EventPattern]:
        """Load historical event patterns for comparison."""
        # In real implementation, this would load from database
        return [
            EventPattern(
                pattern_id="market_crash_2008",
                event_type=EventType.MARKET_CRASH,
                characteristics={
                    'typical_decline': -0.20,
                    'volatility_increase': 3.0,
                    'duration_days': 30
                },
                typical_duration=30,
                typical_impact=0.95,
                historical_frequency=0.001
            ),
            EventPattern(
                pattern_id="flash_crash_2010",
                event_type=EventType.PRICE_SPIKE,
                characteristics={
                    'typical_decline': -0.10,
                    'recovery_time_hours': 1,
                    'liquidity_dryup': True
                },
                typical_duration=1,
                typical_impact=0.60,
                historical_frequency=0.01
            )
        ]

    async def start_monitoring(self):
        """Start continuous monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.logger.info("Starting extreme event monitoring")

        while self.monitoring_active:
            try:
                await self._monitoring_cycle()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Monitoring cycle failed: {e}")
                await asyncio.sleep(30)  # Wait before retrying

    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        self.logger.info("Extreme event monitoring stopped")

    async def _monitoring_cycle(self):
        """Execute one monitoring cycle."""
        # This would fetch current market data
        # For now, this is a placeholder for the actual monitoring implementation
        pass

    async def analyze_data_for_events(self,
                                    data: pd.DataFrame,
                                    symbol: str = "UNKNOWN") -> List[ExtremeEvent]:
        """
        Analyze data for extreme events.

        Args:
            data: Market data to analyze
            symbol: Symbol identifier

        Returns:
            List of detected extreme events
        """
        try:
            all_events = []

            # Add symbol to data
            data = data.copy()
            data['symbol'] = symbol

            # Run all detectors
            for detector in self.detectors:
                events = detector.detect_events(data)
                all_events.extend(events)

            # Post-process events
            processed_events = await self._process_events(all_events)

            # Store events
            for event in processed_events:
                self.event_history.append(event)
                self.active_events[event.event_id] = event

            # Send alerts
            for event in processed_events:
                await self._send_alerts(event)

            return processed_events

        except Exception as e:
            self.logger.error(f"Event analysis failed: {e}")
            return []

    async def _process_events(self, events: List[ExtremeEvent]) -> List[ExtremeEvent]:
        """Post-process detected events."""
        processed_events = []

        for event in events:
            # Compare with historical patterns
            pattern_match = await self._compare_with_patterns(event)

            # Update event with pattern information
            if pattern_match:
                event.context['pattern_match'] = pattern_match.pattern_id
                event.context['pattern_similarity'] = pattern_match.typical_impact

                # Adjust impact score based on pattern
                event.impact_score = min(1.0, event.impact_score * pattern_match.typical_impact)

            # Check for event clusters
            cluster_info = await self._analyze_event_clusters(event)
            if cluster_info:
                event.context['cluster_info'] = cluster_info

            processed_events.append(event)

        return processed_events

    async def _compare_with_patterns(self, event: ExtremeEvent) -> Optional[EventPattern]:
        """Compare event with historical patterns."""
        best_match = None
        best_similarity = 0.0

        for pattern in self.event_patterns:
            if pattern.event_type == event.event_type:
                similarity = self._calculate_pattern_similarity(event, pattern)
                if similarity > best_similarity and similarity > 0.5:
                    best_match = pattern
                    best_similarity = similarity

        return best_match

    def _calculate_pattern_similarity(self, event: ExtremeEvent, pattern: EventPattern) -> float:
        """Calculate similarity between event and pattern."""
        similarity = 0.0
        factors = 0

        # Compare impact scores
        if pattern.typical_impact > 0:
            impact_similarity = 1.0 - abs(event.impact_score - pattern.typical_impact)
            similarity += impact_similarity
            factors += 1

        # Compare characteristics if available
        for key, expected_value in pattern.characteristics.items():
            if key in event.market_indicators:
                actual_value = event.market_indicators[key]
                characteristic_similarity = 1.0 - min(1.0, abs(actual_value - expected_value) / abs(expected_value))
                similarity += characteristic_similarity
                factors += 1

        return similarity / factors if factors > 0 else 0.0

    async def _analyze_event_clusters(self, event: ExtremeEvent) -> Optional[Dict[str, Any]]:
        """Analyze if event is part of a cluster."""
        # Look for recent similar events
        recent_events = [e for e in self.event_history if
                         (datetime.fromisoformat(e.timestamp) > datetime.fromisoformat(event.timestamp) - timedelta(hours=1)) and
                         e.event_type == event.event_type]

        if len(recent_events) >= 3:  # Cluster threshold
            return {
                'cluster_size': len(recent_events) + 1,
                'cluster_duration_minutes': int(
                    (datetime.fromisoformat(event.timestamp) -
                     datetime.fromisoformat(recent_events[0].timestamp)).total_seconds() / 60
                ),
                'affected_symbols': list(set(e.symbol for e in recent_events + [event]))
            }

        return None

    async def _send_alerts(self, event: ExtremeEvent):
        """Send alerts for the event."""
        for config_name, alert_config in self.alert_config.items():
            if not alert_config.enabled:
                continue

            if self._should_send_alert(event, alert_config):
                try:
                    if alert_config.channel == AlertChannel.CONSOLE:
                        await self._send_console_alert(event, alert_config)
                    elif alert_config.channel == AlertChannel.EMAIL:
                        await self._send_email_alert(event, alert_config)
                    elif alert_config.channel == AlertChannel.WEBHOOK:
                        await self._send_webhook_alert(event, alert_config)

                except Exception as e:
                    self.logger.error(f"Failed to send {config_name} alert: {e}")

    def _should_send_alert(self, event: ExtremeEvent, alert_config: AlertNotification) -> bool:
        """Determine if alert should be sent."""
        severity_levels = {
            EventSeverity.LOW: 1,
            EventSeverity.MEDIUM: 2,
            EventSeverity.HIGH: 3,
            EventSeverity.CRITICAL: 4
        }

        threshold_level = severity_levels.get(alert_config.severity_threshold, 2)
        event_level = severity_levels.get(event.severity, 1)

        return event_level >= threshold_level

    async def _send_console_alert(self, event: ExtremeEvent, alert_config: AlertNotification):
        """Send console alert."""
        message = alert_config.message_template.format(
            event_type=event.event_type.value,
            severity=event.severity.value,
            symbol=event.symbol,
            timestamp=event.timestamp
        )

        self.logger.warning(f"ALERT: {message}")

    async def _send_email_alert(self, event: ExtremeEvent, alert_config: AlertNotification):
        """Send email alert (placeholder implementation)."""
        message = alert_config.message_template.format(
            event_type=event.event_type.value,
            severity=event.severity.value,
            symbol=event.symbol,
            timestamp=event.timestamp
        )

        # In real implementation, this would send actual email
        self.logger.info(f"EMAIL alert would be sent to {alert_config.recipients}: {message}")

    async def _send_webhook_alert(self, event: ExtremeEvent, alert_config: AlertNotification):
        """Send webhook alert (placeholder implementation)."""
        payload = {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'severity': event.severity.value,
            'symbol': event.symbol,
            'timestamp': event.timestamp,
            'impact_score': event.impact_score,
            'market_indicators': event.market_indicators
        }

        # In real implementation, this would make HTTP request
        self.logger.info(f"WEBHOOK alert would be sent: {payload}")

    async def get_event_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive event monitoring dashboard."""
        try:
            # Recent events (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_events = [
                asdict(e) for e in self.event_history
                if datetime.fromisoformat(e.timestamp) > cutoff_time
            ]

            # Event statistics
            event_stats = self._calculate_event_statistics()

            # Active alerts
            active_alerts = [
                asdict(e) for e in self.active_events.values()
                if datetime.fromisoformat(e.timestamp) > datetime.now() - timedelta(hours=6)
            ]

            # Pattern analysis
            pattern_analysis = self._analyze_pattern_activity()

            return {
                'dashboard_time': datetime.now().isoformat(),
                'recent_events': recent_events[-50:],  # Last 50 events
                'event_statistics': event_stats,
                'active_alerts': active_alerts,
                'pattern_analysis': pattern_analysis,
                'monitoring_status': {
                    'monitoring_active': self.monitoring_active,
                    'total_events_detected': len(self.event_history),
                    'active_event_count': len(self.active_events),
                    'detectors_active': len(self.detectors),
                    'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to generate event dashboard: {e}")
            return {'error': str(e)}

    def _calculate_event_statistics(self) -> Dict[str, Any]:
        """Calculate event statistics."""
        if not self.event_history:
            return {'message': 'No events detected yet'}

        events_list = list(self.event_history)

        # By type
        by_type = defaultdict(int)
        for event in events_list:
            by_type[event.event_type.value] += 1

        # By severity
        by_severity = defaultdict(int)
        for event in events_list:
            by_severity[event.severity.value] += 1

        # Recent activity (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_events = [e for e in events_list if datetime.fromisoformat(e.timestamp) > week_ago]

        return {
            'total_events': len(events_list),
            'events_by_type': dict(by_type),
            'events_by_severity': dict(by_severity),
            'events_last_week': len(recent_events),
            'average_impact_score': np.mean([e.impact_score for e in events_list]),
            'high_severity_events': len([e for e in events_list if e.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]])
        }

    def _analyze_pattern_activity(self) -> Dict[str, Any]:
        """Analyze pattern matching activity."""
        pattern_matches = 0
        unique_patterns = set()

        for event in self.event_history:
            if 'pattern_match' in event.context:
                pattern_matches += 1
                unique_patterns.add(event.context['pattern_match'])

        return {
            'pattern_matches': pattern_matches,
            'unique_patterns_matched': len(unique_patterns),
            'match_rate': pattern_matches / len(self.event_history) if self.event_history else 0,
            'most_active_pattern': max(unique_patterns, key=lambda p: sum(1 for e in self.event_history if e.context.get('pattern_match') == p), default=None)
        }

    async def get_event_analysis(self, event_id: str) -> Dict[str, Any]:
        """Get detailed analysis for a specific event."""
        try:
            event = next((e for e in self.event_history if e.event_id == event_id), None)
            if not event:
                return {'error': f'Event {event_id} not found'}

            # Find similar historical events
            similar_events = await self._find_similar_events(event)

            # Impact analysis
            impact_analysis = self._analyze_event_impact(event)

            return {
                'event_details': asdict(event),
                'similar_historical_events': similar_events,
                'impact_analysis': impact_analysis,
                'risk_assessment': self._assess_event_risk(event)
            }

        except Exception as e:
            self.logger.error(f"Failed to analyze event {event_id}: {e}")
            return {'error': str(e)}

    async def _find_similar_events(self, event: ExtremeEvent) -> List[Dict[str, Any]]:
        """Find similar historical events."""
        similar_events = []

        for historical_event in self.event_history:
            if historical_event.event_id == event.event_id:
                continue

            # Check for similarity
            if (historical_event.event_type == event.event_type and
                historical_event.severity == event.severity):

                similarity_score = self._calculate_event_similarity(event, historical_event)
                if similarity_score > 0.6:
                    similar_events.append({
                        'event_id': historical_event.event_id,
                        'timestamp': historical_event.timestamp,
                        'similarity_score': similarity_score,
                        'impact_score': historical_event.impact_score
                    })

        return sorted(similar_events, key=lambda x: x['similarity_score'], reverse=True)[:5]

    def _calculate_event_similarity(self, event1: ExtremeEvent, event2: ExtremeEvent) -> float:
        """Calculate similarity between two events."""
        similarity_factors = []

        # Impact similarity
        impact_similarity = 1.0 - abs(event1.impact_score - event2.impact_score)
        similarity_factors.append(impact_similarity)

        # Market indicators similarity
        common_indicators = set(event1.market_indicators.keys()) & set(event2.market_indicators.keys())
        if common_indicators:
            indicator_similarities = []
            for indicator in common_indicators:
                val1 = event1.market_indicators[indicator]
                val2 = event2.market_indicators[indicator]
                if abs(val2) > 0:
                    similarity = 1.0 - min(1.0, abs(val1 - val2) / abs(val2))
                    indicator_similarities.append(similarity)

            if indicator_similarities:
                similarity_factors.append(np.mean(indicator_similarities))

        return np.mean(similarity_factors) if similarity_factors else 0.0

    def _analyze_event_impact(self, event: ExtremeEvent) -> Dict[str, Any]:
        """Analyze the impact of an event."""
        return {
            'market_impact': event.impact_score,
            'statistical_significance': event.statistical_significance,
            'affected_assets_count': len(event.affected_assets),
            'severity_level': event.severity.value,
            'event_type_category': event.event_type.value,
            'time_since_event': (datetime.now() - datetime.fromisoformat(event.timestamp)).total_seconds() / 3600  # hours
        }

    def _assess_event_risk(self, event: ExtremeEvent) -> Dict[str, Any]:
        """Assess risk level of an event."""
        risk_factors = []

        # High severity
        if event.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]:
            risk_factors.append("high_severity")

        # High impact
        if event.impact_score > 0.8:
            risk_factors.append("high_impact")

        # Statistical significance
        if event.statistical_significance > 0.95:
            risk_factors.append("statistically_significant")

        # Pattern match to high-risk patterns
        if event.context.get('pattern_match') in ['market_crash_2008']:
            risk_factors.append("high_risk_pattern")

        risk_level = len(risk_factors) / 4.0  # Normalize to 0-1

        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommended_actions': event.recommended_actions,
            'monitoring_priority': 'high' if risk_level > 0.6 else 'medium'
        }