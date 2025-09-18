# Trading Strategy Specification: [FEATURE NAME]

**Feature Branch**: `[###-feature-name]`
**Created**: [DATE]
**Status**: Draft
**Input**: Trading strategy description: "$ARGUMENTS"

## Execution Flow (main)
```
1. Parse trading strategy description from Input
   → If empty: ERROR "No strategy description provided"
2. Extract key trading concepts from description
   → Identify: assets, timeframes, indicators, risk rules
3. For each unclear aspect:
   → Mark with [NEEDS CLARIFICATION: specific question]
4. Fill Trading Scenarios & Backtesting section
   → If no clear trading logic: ERROR "Cannot determine trading scenarios"
5. Generate Functional Requirements
   → Each requirement must be testable via backtesting
   → Mark ambiguous requirements
6. Identify Key Trading Entities (if data involved)
7. Run Review Checklist
   → If any [NEEDS CLARIFICATION]: WARN "Strategy has uncertainties"
   → If implementation details found: ERROR "Remove tech details"
8. Return: SUCCESS (strategy ready for implementation planning)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT trading strategy does and WHY it should generate alpha
- ❌ Avoid HOW to implement (no algorithms, APIs, code structure)
- 👥 Written for quantitative analysts, not developers

### Section Requirements
- **Mandatory sections**: Must be completed for every trading strategy
- **Optional sections**: Include only when relevant to the strategy
- When a section doesn't apply, remove it entirely (don't leave as "N/A")

### For AI Generation
When creating this strategy spec from a user prompt:
1. **Mark all ambiguities**: Use [NEEDS CLARIFICATION: specific question] for any assumption you'd need to make
2. **Don't guess**: If the prompt doesn't specify something (e.g., "momentum strategy" without timeframe), mark it
3. **Think like a quant**: Every vague requirement should fail the "testable and unambiguous" checklist item
4. **Common underspecified areas**:
   - Asset classes and instruments
   - Timeframes and holding periods
   - Risk parameters and position sizing
   - Entry/exit criteria
   - Market regime considerations
   - Performance benchmarks and targets

---

## Trading Scenarios & Backtesting *(mandatory)*

### Primary Trading Strategy
[Describe the main trading logic in quantitative terms]

### Market Regime Analysis
1. **Bull Market**: [How strategy performs in upward trending markets]
2. **Bear Market**: [How strategy performs in downward trending markets]
3. **Sideways Market**: [How strategy performs in range-bound markets]
4. **High Volatility**: [How strategy handles volatile conditions]

### Backtest Scenarios
- **Given** [market conditions], **When** [signal triggers], **Then** [expected trade action]
- **Given** [market conditions], **When** [risk limit breached], **Then** [expected risk response]

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: Strategy MUST [specific capability, e.g., "detect momentum signals using moving average crossover"]
- **FR-002**: System MUST [specific capability, e.g., "calculate position size based on volatility"]
- **FR-003**: Strategy MUST be able to [key trading action, e.g., "enter long positions when RSI < 30"]
- **FR-004**: System MUST [data requirement, e.g., "maintain historical price data for 252 days"]
- **FR-005**: Strategy MUST [risk management, e.g., "implement stop-loss at 2% below entry"]

*Example of marking unclear requirements:*
- **FR-006**: Strategy MUST trade [NEEDS CLARIFICATION: asset class not specified - equities, forex, commodities?]
- **FR-007**: System MUST achieve [NEEDS CLARIFICATION: performance target not specified - Sharpe ratio, CAGR, max drawdown?]

### Key Trading Entities *(include if strategy involves data)*
- **[Entity 1]**: [What it represents, key attributes without implementation]
- **[Entity 2]**: [What it represents, relationships to other entities]

### Risk Parameters *(mandatory for trading strategies)*
- **Maximum Position Size**: [e.g., 5% of portfolio or NEEDS CLARIFICATION]
- **Stop Loss**: [e.g., 2% below entry or NEEDS CLARIFICATION]
- **Take Profit**: [e.g., 3% above entry or NEEDS CLARIFICATION]
- **Maximum Drawdown**: [e.g., 10% from peak or NEEDS CLARIFICATION]

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [ ] No implementation details (algorithms, frameworks, APIs)
- [ ] Focused on trading logic and alpha generation
- [ ] Written for quantitative analysts
- [ ] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain
- [ ] Requirements are testable via backtesting
- [ ] Success criteria are measurable
- [ ] Trading scope is clearly bounded
- [ ] Risk parameters are specified

### Trading Strategy Validation
- [ ] Entry and exit rules are clearly defined
- [ ] Risk management parameters are specified
- [ ] Performance metrics are defined
- [ ] Market regime considerations are included

---

## Execution Status
*Updated by main() during processing*

- [ ] Trading strategy description parsed
- [ ] Key trading concepts extracted
- [ ] Ambiguities marked
- [ ] Trading scenarios defined
- [ ] Requirements generated
- [ ] Trading entities identified
- [ ] Review checklist passed

---
