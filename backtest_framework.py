# ============================================
# BACKTESTING FRAMEWORK
# ============================================
# Tests rules against historical OHLCV data

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import itertools
from intelligent_trading_core import (
    EntryConfluenceChecker,
    AdaptiveRiskManager,
    IntelligentTradeManager,
    TradeFilter
)


class RuleBacktester:
    """Backtest rule-based trading strategy"""
    
    def __init__(self, initial_balance=10000, commission=0.0001):
        self.initial_balance = initial_balance
        self.commission = commission
        
        self.confluence = EntryConfluenceChecker(min_confirmations=3)
        self.risk_manager = AdaptiveRiskManager()
        self.trade_manager = IntelligentTradeManager()
        self.trade_filter = TradeFilter()
    
    def backtest(self, df, verbose=False):
        """
        Run backtest on OHLCV dataframe
        
        df must contain: close, high, low, open, tick_volume, ATR, ema_fast, ema_slow, ST_dir
        
        Returns: dict with metrics
        """
        
        trades = []
        position = None
        balance = self.initial_balance
        equity_curve = [balance]
        recent_trades_pnl = []
        
        print(f"\n🔄 Backtesting {len(df)} candles...")
        
        for i in range(50, len(df)):
            current_df = df.iloc[max(0, i-50):i+1].copy()
            current_price = float(df.iloc[i]['close'])
            
            # ===== NO POSITION: LOOK FOR ENTRY =====
            if position is None:
                
                # Check confluence
                confluence = self.confluence.check_all_rules(current_df)
                
                # Determine direction
                ema_fast = float(df.iloc[i]['ema_fast'])
                ema_slow = float(df.iloc[i]['ema_slow'])
                direction = 'BUY' if ema_fast > ema_slow else 'SELL'
                
                # Check if signal is strong enough
                if confluence['min_met'] and confluence['signal_strength'] > 5.0:
                    
                    # Apply filters
                    filter_result = self.trade_filter.validate_entry(current_df, direction)
                    
                    if filter_result['valid']:
                        
                        # Calculate SL/TP
                        atr = float(df.iloc[i]['ATR'])
                        sl_distance = 1.5 * atr
                        tp_distance = 2.0 * atr
                        
                        if direction == 'BUY':
                            sl_price = current_price - sl_distance
                            tp_price = current_price + tp_distance
                        else:
                            sl_price = current_price + sl_distance
                            tp_price = current_price - tp_distance
                        
                        # Calculate lot size
                        atr_avg = float(df.iloc[max(0, i-20):i+1]['ATR'].mean())
                        lot_calc = self.risk_manager.calculate_smart_lot(
                            'XAUUSD', current_price, sl_price,
                            balance, recent_trades_pnl, atr, atr_avg
                        )
                        
                        # Entry trade
                        position = {
                            'entry_price': current_price,
                            'direction': direction,
                            'sl_price': sl_price,
                            'tp_price': tp_price,
                            'sl_pips': sl_distance,
                            'tp_pips': tp_distance,
                            'lot_size': lot_calc['lot_size'],
                            'entry_bar': i,
                            'entry_confluence': confluence['signal_strength'],
                            'partial_closed': False
                        }
                        
                        if verbose:
                            print(f"[{i}] 📈 ENTRY {direction} @ {current_price:.2f} | "
                                  f"SL:{sl_price:.2f} TP:{tp_price:.2f} | "
                                  f"Strength:{confluence['signal_strength']:.1f} | "
                                  f"Lot:{lot_calc['lot_size']}")
            
            # ===== HAS POSITION: MANAGE IT =====
            else:
                
                # Calculate current P&L in pips
                if position['direction'] == 'BUY':
                    profit_pips = current_price - position['entry_price']
                else:
                    profit_pips = position['entry_price'] - current_price
                
                position['time_open_bars'] = i - position['entry_bar']
                
                # Check exit rules
                exit_signal = self.trade_manager.should_close_trade(
                    position, current_df, profit_pips
                )
                
                # Check Hard SL/TP
                if position['direction'] == 'BUY':
                    if current_price >= position['tp_price']:
                        exit_signal = {'should_close': True, 'reason': 'TP_HIT', 'priority': 10}
                    elif current_price <= position['sl_price']:
                        exit_signal = {'should_close': True, 'reason': 'SL_HIT', 'priority': 10}
                else:
                    if current_price <= position['tp_price']:
                        exit_signal = {'should_close': True, 'reason': 'TP_HIT', 'priority': 10}
                    elif current_price >= position['sl_price']:
                        exit_signal = {'should_close': True, 'reason': 'SL_HIT', 'priority': 10}
                
                # Exit trade
                if exit_signal['should_close']:
                    
                    # Calculate P&L
                    if position['direction'] == 'BUY':
                        exit_price = current_price
                        pnl = (current_price - position['entry_price']) * position['lot_size']
                    else:
                        exit_price = current_price
                        pnl = (position['entry_price'] - current_price) * position['lot_size']
                    
                    pnl -= abs(pnl) * self.commission  # Commission
                    
                    # Update balance
                    balance += pnl
                    
                    # Log trade
                    trade_record = {
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'direction': position['direction'],
                        'pnl': pnl,
                        'pnl_pips': profit_pips,
                        'lot_size': position['lot_size'],
                        'duration_bars': position['time_open_bars'],
                        'entry_bar': position['entry_bar'],
                        'exit_bar': i,
                        'exit_reason': exit_signal['reason'],
                        'win': pnl > 0
                    }
                    trades.append(trade_record)
                    recent_trades_pnl.append(pnl)
                    
                    if verbose:
                        win_emoji = '✅' if pnl > 0 else '❌'
                        print(f"[{i}] {win_emoji} EXIT {position['direction']} @ {exit_price:.2f} | "
                              f"PnL: {pnl:.2f} ({profit_pips:.2f}p) | "
                              f"Reason: {exit_signal['reason']}")
                    
                    position = None
                    equity_curve.append(balance)
        
        # Close any open position at last price
        if position is not None:
            last_price = float(df.iloc[-1]['close'])
            if position['direction'] == 'BUY':
                pnl = (last_price - position['entry_price']) * position['lot_size']
            else:
                pnl = (position['entry_price'] - last_price) * position['lot_size']
            
            pnl -= abs(pnl) * self.commission
            balance += pnl
            
            trades.append({
                'entry_price': position['entry_price'],
                'exit_price': last_price,
                'direction': position['direction'],
                'pnl': pnl,
                'pnl_pips': (last_price - position['entry_price']) if position['direction'] == 'BUY' else (position['entry_price'] - last_price),
                'lot_size': position['lot_size'],
                'duration_bars': len(df) - position['entry_bar'],
                'entry_bar': position['entry_bar'],
                'exit_bar': len(df) - 1,
                'exit_reason': 'END_OF_DATA',
                'win': pnl > 0
            })
            recent_trades_pnl.append(pnl)
            equity_curve.append(balance)
        
        # Calculate metrics
        return self._calculate_metrics(trades, equity_curve, balance)
    
    def _calculate_metrics(self, trades, equity_curve, final_balance):
        """Calculate performance metrics"""
        
        if not trades:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'error': 'No trades executed'
            }
        
        pnls = [t['pnl'] for t in trades]
        pnl_pips = [t['pnl_pips'] for t in trades]
        wins = [t for t in trades if t['win']]
        losses = [t for t in trades if not t['win']]
        
        total_pnl = sum(pnls)
        win_rate = len(wins) / len(trades) if trades else 0
        
        # Profit Factor: Gross profit / Gross loss
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Max Drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (running_max - equity_array) / running_max
        max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0
        
        # Sharpe Ratio (simplified)
        if len(pnls) > 1:
            returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        # Win streaks
        win_streaks = [len(list(g)) for k, g in itertools.groupby([t['win'] for t in trades]) if k]
        max_win_streak = max(win_streaks) if win_streaks else 0
        
        loss_streaks = [len(list(g)) for k, g in itertools.groupby([not t['win'] for t in trades]) if k]
        max_loss_streak = max(loss_streaks) if loss_streaks else 0
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': round(win_rate * 100, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_pnl_per_trade': round(total_pnl / len(trades), 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(np.mean([t['pnl'] for t in wins]), 2) if wins else 0,
            'avg_loss': round(np.mean([t['pnl'] for t in losses]), 2) if losses else 0,
            'best_trade': round(max(pnls), 2),
            'worst_trade': round(min(pnls), 2),
            'max_drawdown': round(max_drawdown * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_consecutive_wins': max_win_streak,
            'max_consecutive_losses': max_loss_streak,
            'final_balance': round(final_balance, 2),
            'roi': round((final_balance - self.initial_balance) / self.initial_balance * 100, 2),
            'trades': trades
        }


print("✅ Backtest Framework Loaded")
