# ============================================
# INTELLIGENT RULE-BASED TRADING CORE
# ============================================
# Multi-confirmation entry system + Dynamic risk management

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. ENTRY CONFLUENCE CHECKER
# ============================================

class EntryConfluenceChecker:
    """Validates entry signal with multiple rules before executing"""
    
    def __init__(self, min_confirmations=3):
        self.min_confirmations = min_confirmations
    
    def check_all_rules(self, df, direction=None):
        """
        Score entry signal (0-10 scale)
        direction: 'BUY' or 'SELL' or None (auto-detect)
        """
        if len(df) < 50:
            return {'signal_strength': 0, 'confirmations': 0, 'min_met': False, 'breakdown': {}}
        
        results = {}
        
        # RULE 1: EMA Crossover (weight: 1.5x)
        results['ema'] = self._ema_crossover(df)
        
        # RULE 2: Supertrend Direction (weight: 1.5x)
        results['supertrend'] = self._supertrend_trend(df)
        
        # RULE 3: RSI Confirmation (weight: 1.0x)
        results['rsi'] = self._rsi_momentum(df, period=14)
        
        # RULE 4: Volume Confirmation (weight: 0.8x)
        results['volume'] = self._volume_strength(df)
        
        # RULE 5: Price Action Pattern (weight: 1.2x)
        results['price_action'] = self._price_action_pattern(df)
        
        # RULE 6: Volatility Filter (weight: 1.0x) - Boolean
        results['volatility_filter'] = self._volatility_filter(df)
        
        # RULE 7: Market Regime (weight: 0.5x)
        results['regime'] = self._market_regime(df, period=50)
        
        # RULE 8: MACD Momentum (weight: 0.8x)
        results['macd'] = self._macd_signal(df)
        
        # Count confirmations
        confirmations = sum([
            results['ema'] > 0,
            results['supertrend'] > 0,
            results['rsi'] > 0,
            results['volume'] > 0,
            results['price_action'] > 0,
            results['volatility_filter'],
            results['regime'] > 0,
            results['macd'] > 0
        ])
        
        # Calculate weighted score
        weights = {
            'ema': 1.5,
            'supertrend': 1.5,
            'rsi': 1.0,
            'volume': 0.8,
            'price_action': 1.2,
            'volatility_filter': 1.0,
            'regime': 0.5,
            'macd': 0.8
        }
        
        total_weight = sum(weights.values())
        weighted_sum = sum(
            results[rule] * weights[rule] 
            for rule in results.keys() if rule != 'volatility_filter'
        )
        weighted_sum += (10 if results['volatility_filter'] else -10) * weights['volatility_filter']
        
        signal_strength = weighted_sum / total_weight / 10.0  # Normalize to 0-1
        signal_strength = max(0, min(1, signal_strength))  # Clamp 0-1
        
        return {
            'signal_strength': signal_strength * 10,  # 0-10 scale
            'confirmations': confirmations,
            'min_met': confirmations >= self.min_confirmations,
            'breakeven_distance': self._calculate_breakeven_distance(df),
            'breakdown': results
        }
    
    def _ema_crossover(self, df):
        """EMA: Fast > Slow = Bullish (+10), Fast < Slow = Bearish (-10)"""
        if len(df) < 2 or 'ema_fast' not in df.columns:
            return 0
        
        try:
            ema_fast_now = float(df.iloc[-1]['ema_fast'])
            ema_fast_prev = float(df.iloc[-2]['ema_fast'])
            ema_slow = float(df.iloc[-1]['ema_slow'])
            ema_slow_prev = float(df.iloc[-2]['ema_slow'])
            
            # Bullish crossover: fast crosses above slow
            if ema_fast_prev <= ema_slow_prev and ema_fast_now > ema_slow:
                return 10  # Strong bullish
            # Bearish crossover
            elif ema_fast_prev >= ema_slow_prev and ema_fast_now < ema_slow:
                return -10  # Strong bearish
            # Already in trend
            elif ema_fast_now > ema_slow:
                return 5  # Weak bullish
            else:
                return -5  # Weak bearish
        except:
            return 0
    
    def _supertrend_trend(self, df):
        """Supertrend direction: Uptrend vs Downtrend"""
        if 'ST_dir' not in df.columns or len(df) < 1:
            return 0
        try:
            st_dir = bool(df.iloc[-1]['ST_dir'])
            return 8 if st_dir else -8
        except:
            return 0
    
    def _rsi_momentum(self, df, period=14):
        """RSI: Overbought/oversold + momentum"""
        if len(df) < period + 1:
            return 0
        
        try:
            close = df['close'].values
            deltas = np.diff(close)
            
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            rsi = 100 - (100 / (1 + rs)) if rs >= 0 else 0
            
            # Scoring
            if rsi > 70:
                return -5  # Overbought
            elif rsi < 30:
                return 5   # Oversold (bullish bounce)
            elif rsi > 50:
                return 3   # Bullish bias
            else:
                return -3  # Bearish bias
        except:
            return 0
    
    def _volume_strength(self, df):
        """Volume confirmation: is volume above average?"""
        if len(df) < 20 or 'tick_volume' not in df.columns:
            return 0
        
        try:
            current_vol = float(df.iloc[-1]['tick_volume'])
            avg_vol = float(df['tick_volume'].tail(20).mean())
            
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            if vol_ratio > 1.3:
                return 7  # Strong confirmation
            elif vol_ratio > 1.0:
                return 3  # Weak confirmation
            else:
                return -4  # Weak signal
        except:
            return 0
    
    def _price_action_pattern(self, df):
        """Detect higher lows (bullish) or lower highs (bearish)"""
        if len(df) < 3:
            return 0
        
        try:
            close_3 = float(df.iloc[-3]['close'])
            close_2 = float(df.iloc[-2]['close'])
            close_1 = float(df.iloc[-1]['close'])
            
            # Higher Low: bullish
            higher_low = (close_2 > close_3) and (close_1 > close_2)
            if higher_low:
                return 6
            
            # Lower High: bearish
            lower_high = (close_2 < close_3) and (close_1 < close_2)
            if lower_high:
                return -6
            
            return 0
        except:
            return 0
    
    def _volatility_filter(self, df, atr_period=14):
        """Filter: Only trade in NORMAL volatility zones"""
        if len(df) < atr_period or 'ATR' not in df.columns:
            return True
        
        try:
            atr = float(df.iloc[-1]['ATR'])
            atr_avg = float(df['ATR'].tail(20).mean())
            
            # Reject if too volatile (1.5x average)
            if atr > atr_avg * 1.5:
                return False
            
            # Reject if too quiet (0.5x average)
            if atr < atr_avg * 0.5:
                return False
            
            return True
        except:
            return True
    
    def _market_regime(self, df, period=50):
        """Determine trending vs ranging market"""
        if len(df) < period:
            return 0
        
        try:
            closes = df['close'].tail(period).values
            highs = df['high'].tail(period).values
            lows = df['low'].tail(period).values
            
            # ADX proxy
            current_range = abs(closes[-1] - closes[-period])
            avg_range = np.mean([h - l for h, l in zip(highs, lows)])
            
            trend_strength = current_range / avg_range if avg_range > 0 else 0
            
            if trend_strength > 1.2:
                return 7  # Strong trend
            elif trend_strength > 0.8:
                return 4  # Mild trend
            else:
                return -3  # Ranging
        except:
            return 0
    
    def _macd_signal(self, df, fast=12, slow=26, signal=9):
        """MACD: histogram > 0 = bullish, < 0 = bearish"""
        if len(df) < slow + 1:
            return 0
        
        try:
            close = df['close'].values
            ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
            ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
            
            macd_line = ema_fast - ema_slow
            signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
            histogram = macd_line - signal_line
            
            current_hist = histogram[-1]
            prev_hist = histogram[-2]
            
            # Crossover: histogram crosses above zero
            if prev_hist <= 0 and current_hist > 0:
                return 8  # Bullish crossover
            elif prev_hist >= 0 and current_hist < 0:
                return -8  # Bearish crossover
            elif current_hist > 0:
                return 3  # Bullish
            else:
                return -3  # Bearish
        except:
            return 0
    
    def _calculate_breakeven_distance(self, df):
        """How many pips/points to breakeven? (for position sizing)"""
        if len(df) < 14 or 'ATR' not in df.columns:
            return 1.5
        
        try:
            atr = float(df.iloc[-1]['ATR'])
            return atr
        except:
            return 1.5


# ============================================
# 2. ADAPTIVE RISK MANAGER
# ============================================

class AdaptiveRiskManager:
    """Adjusts position size based on market conditions"""
    
    def calculate_smart_lot(self, symbol, entry_price, sl_price, 
                            account_balance, recent_trades_list, atr_val, atr_avg):
        """
        Position sizing formula with multiple factors
        recent_trades_list: list of recent PnL values
        """
        
        base_risk_percent = 1.0  # Default 1% per trade
        
        # ===== FACTOR 1: Win Rate Adjustment =====
        win_rate = self._calculate_win_rate(recent_trades_list)
        
        if win_rate < 0.35:
            reduce_factor = 0.5  # Reduce 50% after bad stretch
        elif win_rate < 0.45:
            reduce_factor = 0.75
        elif win_rate > 0.60:
            reduce_factor = 1.2  # Increase 20% when hot
        else:
            reduce_factor = 1.0
        
        # ===== FACTOR 2: Drawdown Brake =====
        recent_dd = self._calculate_recent_drawdown(recent_trades_list)
        max_dd_percent = 5.0
        
        if recent_dd > max_dd_percent:
            dd_brake = 0.5  # Cut risk in half
        elif recent_dd > max_dd_percent * 0.7:
            dd_brake = 0.75
        else:
            dd_brake = 1.0
        
        # ===== FACTOR 3: Volatility Scaling =====
        if atr_val is not None and atr_avg is not None:
            if atr_val > atr_avg * 1.3:
                vol_factor = 0.7  # Reduce 30% in high vol
            elif atr_val > atr_avg:
                vol_factor = 0.85
            else:
                vol_factor = 1.0
        else:
            vol_factor = 1.0
        
        # Combine factors
        effective_risk = base_risk_percent * reduce_factor * dd_brake * vol_factor
        effective_risk = max(0.25, min(effective_risk, 2.5))  # Cap: 0.25% - 2.5%
        
        # Calculate lot based on effective risk
        try:
            risk_amount = account_balance * (effective_risk / 100.0)
            sl_points = abs(entry_price - sl_price) / 0.01  # Assuming 0.01 is point for XAUUSD
            vpp = 10.0  # Value per point approximation
            
            lot = risk_amount / (sl_points * vpp) if sl_points > 0 else 0.01
            lot = max(0.01, min(lot, 10.0))  # Cap lot size
            
        except:
            lot = 0.01
        
        return {
            'lot_size': round(lot, 2),
            'effective_risk_percent': round(effective_risk, 2),
            'breakdown': {
                'base_risk': base_risk_percent,
                'win_rate_factor': reduce_factor,
                'dd_brake_factor': dd_brake,
                'volatility_factor': vol_factor,
                'win_rate': round(win_rate * 100, 1)
            }
        }
    
    def _calculate_win_rate(self, recent_trades):
        """Win rate from last N trades"""
        if not recent_trades:
            return 0.5
        
        recent = recent_trades[-20:] if len(recent_trades) > 20 else recent_trades
        if not recent:
            return 0.5
        
        wins = sum(1 for p in recent if p > 0)
        return wins / len(recent)
    
    def _calculate_recent_drawdown(self, recent_trades):
        """Max drawdown from recent trades"""
        if not recent_trades:
            return 0.0
        
        cumsum = np.cumsum(recent_trades)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = running_max - cumsum
        
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0


# ============================================
# 3. INTELLIGENT TRADE MANAGER
# ============================================

class IntelligentTradeManager:
    """Manages open trades with dynamic exit rules"""
    
    def should_close_trade(self, position, df, profit_pips):
        """
        Decides whether to close based on multiple rules
        Returns: {'should_close': bool, 'reason': str, 'priority': int}
        """
        
        if len(df) < 3:
            return {'should_close': False, 'reason': 'INSUFFICIENT_DATA', 'priority': 0}
        
        try:
            # RULE 1: Take Profit Hit
            if profit_pips >= position['tp_pips']:
                return {'should_close': True, 'reason': 'TAKE_PROFIT_HIT', 'priority': 10}
            
            # RULE 2: Stop Loss Hit
            if profit_pips <= -position['sl_pips']:
                return {'should_close': True, 'reason': 'STOP_LOSS_HIT', 'priority': 10}
            
            # RULE 3: Trend Reversal (exit with profit)
            if self._detect_trend_reversal(df, position) and profit_pips > 0:
                return {'should_close': True, 'reason': 'TREND_REVERSAL', 'priority': 8}
            
            # RULE 4: Time-Based Exit
            time_open = position.get('time_open_bars', 0)
            if time_open > 240 and 0 < profit_pips < 2:  # 240 bars = ~4 hours on M5
                return {'should_close': True, 'reason': 'TIME_TIMEOUT', 'priority': 6}
            
            # RULE 5: Volatility Expansion Exit
            atr = float(df.iloc[-1]['ATR'])
            atr_avg = float(df['ATR'].tail(20).mean())
            if atr > atr_avg * 1.8 and profit_pips > 1:
                return {'should_close': True, 'reason': 'VOLATILITY_SPIKE', 'priority': 7}
            
            # RULE 6: Price Moving Against Trend
            if self._price_moving_against_trend(df, position) and profit_pips > 0.5:
                return {'should_close': True, 'reason': 'COUNTER_TREND_CLOSE', 'priority': 7}
            
            # RULE 7: Partial Close at 50% of TP
            if profit_pips > position['tp_pips'] * 0.5 and not position.get('partial_closed', False):
                return {'should_close': True, 'reason': 'PARTIAL_CLOSE_50PCT', 'priority': 5, 'partial': True}
            
        except Exception as e:
            print(f"[trade_manager] Error: {e}")
        
        return {'should_close': False, 'reason': 'HOLD', 'priority': 0}
    
    def _detect_trend_reversal(self, df, position):
        """Has the trend reversed?"""
        if len(df) < 2:
            return False
        
        try:
            is_buy = position['direction'] == 'BUY'
            ema_f_now = float(df.iloc[-1]['ema_fast'])
            ema_s_now = float(df.iloc[-1]['ema_slow'])
            ema_f_prev = float(df.iloc[-2]['ema_fast'])
            ema_s_prev = float(df.iloc[-2]['ema_slow'])
            
            if is_buy:
                # Crossover down = reversal
                if ema_f_prev > ema_s_prev and ema_f_now <= ema_s_now:
                    return True
            else:
                # Crossover up = reversal
                if ema_f_prev < ema_s_prev and ema_f_now >= ema_s_now:
                    return True
        except:
            pass
        
        return False
    
    def _price_moving_against_trend(self, df, position):
        """3 consecutive candles against position?"""
        if len(df) < 3:
            return False
        
        try:
            is_buy = position['direction'] == 'BUY'
            closes = df['close'].tail(3).values
            
            if is_buy:
                # 3 red candles = exit
                return all(closes[i] > closes[i+1] for i in range(len(closes)-1))
            else:
                # 3 green candles = exit
                return all(closes[i] < closes[i+1] for i in range(len(closes)-1))
        except:
            pass
        
        return False


# ============================================
# 4. TRADE FILTER
# ============================================

class TradeFilter:
    """Rejects bad setups before entry"""
    
    def validate_entry(self, df, signal_direction):
        """
        Returns: {'valid': bool, 'reason': str}
        """
        
        filters = {
            'time_filter': self._check_optimal_time(),
            'technical_anomaly': self._check_technical_anomaly(df),
        }
        
        # REJECT entry if ANY filter fails
        failed_filters = [k for k, v in filters.items() if not v]
        
        if failed_filters:
            return {'valid': False, 'reason': f'FILTER_FAILED: {failed_filters}'}
        
        return {'valid': True, 'reason': 'ALL_FILTERS_PASSED'}
    
    def _check_optimal_time(self):
        """Only trade during liquid hours"""
        current_hour = datetime.now().hour
        
        # Avoid Asian session for XAUUSD (0-8 UTC)
        if 0 <= current_hour < 8:
            return False
        
        return True
    
    def _check_technical_anomaly(self, df):
        """Reject if in extreme volatility"""
        if len(df) < 20 or 'ATR' not in df.columns:
            return True
        
        try:
            atr = float(df.iloc[-1]['ATR'])
            atr_avg = float(df['ATR'].tail(20).mean())
            
            # Extreme volatility = skip
            if atr > atr_avg * 2.0:
                return False
            
            return True
        except:
            return True


print("✅ Intelligent Trading Core Loaded")
