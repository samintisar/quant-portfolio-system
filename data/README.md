# Data Storage Directory

This directory contains offline market data for the portfolio optimization system.

## Directory Structure

```
data/
├── raw/           # Raw data downloaded from Yahoo Finance
├── processed/     # Cleaned and validated data
├── features/       # ML feature matrices
└── README.md       # This file
```

## Data Sources

- **Yahoo Finance API**: Primary source for historical price data
- **Update Frequency**: Manual updates as needed (weekly/monthly recommended)
- **Symbols**: 15-20 large cap US stocks
- **Time Period**: 3-5 years of historical data

## File Naming Convention

- Raw data: `{symbol}_{period}_raw.csv`
- Processed data: `{symbol}_{period}_processed.csv`
- Feature data: `{symbol}_{period}_features.csv`

## Data Management

### Fetching New Data
```bash
python scripts/fetch_market_data.py
```

### Data Refresh
Run the fetch script periodically to update market data with recent prices.

### Data Quality
- Minimum 95% data coverage required
- No extended gaps (>5 consecutive trading days)
- Validated price and volume data

## Notes

- Data is stored offline for reliable access during demos
- No caching logic - simple file-based storage
- Files can be committed to version control for reproducibility
- Large files should be added to .gitignore if needed