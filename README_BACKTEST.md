# Intelligent Trading Bot - Backtesting Framework

## Overview

This is a **rule-based intelligent trading system** with comprehensive backtesting capabilities. The bot uses multi-layer signal confirmation and adaptive risk management for improved decision-making.

## Files

### 1. `intelligent_trading_core.py`
Core trading logic with 4 main components:

- **EntryConfluenceChecker**: Validates entries with 8 rules
  - EMA Crossover
  - Supertrend Direction
  - RSI Momentum
  - Volume Strength
  - Price Action Patterns
  - Volatility Filter
  - Market Regime
  - MACD Signal

- **AdaptiveRiskManager**: Dynamic position sizing
  - Win rate adjustment (reduce after losses, increase after wins)
  - Drawdown brake (cut size in recovery)
  - Volatility scaling (smaller size in high volatility)

- **IntelligentTradeManager**: Smart exit rules
  - Take profit / Stop loss
  - Trend reversal detection
  - Time-based exits
  - Volatility expansion exits
  - Counter-trend exits
  - Partial closes

- **TradeFilter**: Pre-entry validation
  - Time filter (avoid illiquid hours)
  - Technical anomaly check

### 2. `backtest_framework.py`
Backtesting engine with performance metrics:

- Trade-by-trade simulation
- Position management
- P&L calculation
- Performance metrics:
  - Win rate, Profit factor
  - Max drawdown, Sharpe ratio
  - Consecutive wins/losses
  - ROI

### 3. `data_preparation.py`
Data handling and technical indicators:

- ATR (Average True Range)
- EMA (Exponential Moving Averages)
- Supertrend
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- CSV loading
- Synthetic data generation

### 4. `run_backtest.py`
Main execution script:

```bash
python3 run_backtest.py
```

Outputs:
- Console results summary
- JSON file with detailed trade logs

## Quick Start

### Test with Synthetic Data
```bash
python3 run_backtest.py
```

### Test with Your CSV Data

1. Prepare CSV with columns: `open, high, low, close, tick_volume`
2. Edit `run_backtest.py` last lines:
```python
metrics = run_backtest('your_data.csv')
```

3. Run:
```bash
python3 run_backtest.py
```

## How the Rules Work

### Entry Rules

The bot requires **at least 3 out of 8 confirmations** before entering:

```
1. EMA Crossover      (weight: 1.5x) - fast > slow
2. Supertrend         (weight: 1.5x) - direction
3. RSI                (weight: 1.0x) - momentum
4. Volume             (weight: 0.8x) - strength
5. Price Action       (weight: 1.2x) - patterns
6. Volatility Filter  (weight: 1.0x) - normal range
7. Market Regime      (weight: 0.5x) - trending
8. MACD               (weight: 0.8x) - momentum
```

Each rule scores -10 to +10, weighted and combined for final signal strength (0-10).

### Exit Rules

The bot exits when:

1. **Take Profit Hit**: Target reached
2. **Stop Loss Hit**: Max loss reached
3. **Trend Reversal**: EMA crossover against position
4. **Time Timeout**: Open > 4 hours with small profit
5. **Volatility Spike**: ATR > 1.8x average
6. **Counter-trend**: 3 consecutive candles against trade
7. **Partial Close**: At 50% of profit target

### Position Sizing

Lot size adjusts by:

```
Base Risk 1.0% × Win Rate Factor × Drawdown Brake × Volatility Factor

Win Rate Factor:
  < 35% win rate → 0.5x (reduce 50%)
  35-45%         → 0.75x
  > 60%          → 1.2x (increase 20%)

Drawdown Brake:
  In recovery    → 0.5x to 0.75x

Volatility Factor:
  High (>1.3x)   → 0.7x
  Normal         → 1.0x
```

## Results Interpretation

### Good Backtest Results

✅ **Win Rate > 55%** - System wins more than it loses
✅ **Profit Factor > 1.5** - Gross profit 1.5x gross loss
✅ **Max Drawdown < 20%** - Account doesn't drop > 20%
✅ **Sharpe Ratio > 0.5** - Risk-adjusted returns
✅ **Consecutive Losses < 5** - Doesn't go on long losing streaks

### Red Flags

🚩 **Win Rate < 40%** - Too many losses
🚩 **Profit Factor < 1.0** - Losing money overall
🚩 **Max Drawdown > 50%** - Too much risk
🚩 **Consecutive Losses > 10** - Uncontrolled losses

## Customization

### Adjust Minimum Confirmations

In `run_backtest.py`:
```python
backtester = RuleBacktester(initial_balance=10000)
backtester.confluence = EntryConfluenceChecker(min_confirmations=4)  # Require 4/8
```

### Adjust Risk Parameters

In `intelligent_trading_core.py`, `AdaptiveRiskManager`:
```python
base_risk_percent = 1.0  # Change to 0.5 for lower risk
max_dd_percent = 5.0     # Change max acceptable drawdown
```

### Adjust Exit Thresholds

In `intelligent_trading_core.py`, `IntelligentTradeManager`:
```python
if time_open > 240 and 0 < profit_pips < 2:  # Change 240 to different hours
```

## Next Steps

1. **Test with Real Data**: Load your XAUUSD historical data
2. **Optimize Parameters**: Find best min_confirmations, risk %, etc.
3. **Forward Test**: Run on recent data not used in backtest
4. **Live Paper Trading**: Test on demo account
5. **Monitor Performance**: Compare backtest vs live results

## Integration with Live Trading

To integrate with your MT5 bot:

1. Replace signal generation with `EntryConfluenceChecker`
2. Replace position sizing with `AdaptiveRiskManager`
3. Replace exit logic with `IntelligentTradeManager`
4. Use same SL/TP calculation (ATR-based)

Example:
```python
# In your main MT5 bot
from intelligent_trading_core import EntryConfluenceChecker

confluence = EntryConfluenceChecker(min_confirmations=3)
result = confluence.check_all_rules(df)

if result['min_met'] and result['signal_strength'] > 5.0:
    # Place order
    pass
```

## Troubleshooting

### "No trades executed"
- Min confirmations too high
- Data missing required indicators
- Filters too strict

### All trades losing
- Rules not matching market conditions
- SL/TP distances wrong
- Data quality issues

### Extreme drawdown
- Initial balance too small
- Position sizing too aggressive
- SL placement too far

## Support

For issues or questions:
1. Check the indicator calculations (ATR, EMA, etc.)
2. Verify data format (OHLCV required)
3. Review rule scoring in `EntryConfluenceChecker`

---

**Last Updated**: 2026-06-01
