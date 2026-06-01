# ============================================
# DATA PREPARATION FOR BACKTESTING
# ============================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests


class DataPreparator:
    """Prepares OHLCV data with technical indicators"""
    
    @staticmethod
    def add_technical_indicators(df):
        """
        Add all technical indicators needed for backtesting
        Input df must have: open, high, low, close, tick_volume
        """
        
        df = df.copy()
        
        # ===== ATR =====
        df['TR'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # ===== EMA (Fast & Slow) =====
        df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # ===== SUPERTREND =====
        df = DataPreparator._add_supertrend(df, period=14, multiplier=3.0)
        
        # ===== RSI =====
        df['RSI'] = DataPreparator._calculate_rsi(df, period=14)
        
        # ===== MACD =====
        df = DataPreparator._add_macd(df, fast=12, slow=26, signal=9)
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    @staticmethod
    def _add_supertrend(df, period=14, multiplier=3.0):
        """Calculate Supertrend"""
        
        df['hl2'] = (df['high'] + df['low']) / 2
        df['matr'] = df['ATR'] * multiplier
        
        df['basic_ub'] = df['hl2'] + df['matr']
        df['basic_lb'] = df['hl2'] - df['matr']
        
        df['final_ub'] = 0.0
        df['final_lb'] = 0.0
        df['ST_dir'] = True
        
        for i in range(1, len(df)):
            # Final upper band
            if df['basic_ub'].iloc[i] < df['final_ub'].iloc[i-1] or df['close'].iloc[i-1] > df['final_ub'].iloc[i-1]:
                df.loc[df.index[i], 'final_ub'] = df['basic_ub'].iloc[i]
            else:
                df.loc[df.index[i], 'final_ub'] = df['final_ub'].iloc[i-1]
            
            # Final lower band
            if df['basic_lb'].iloc[i] > df['final_lb'].iloc[i-1] or df['close'].iloc[i-1] < df['final_lb'].iloc[i-1]:
                df.loc[df.index[i], 'final_lb'] = df['basic_lb'].iloc[i]
            else:
                df.loc[df.index[i], 'final_lb'] = df['final_lb'].iloc[i-1]
            
            # Direction
            if df['ST_dir'].iloc[i-1] and df['close'].iloc[i] <= df['final_ub'].iloc[i]:
                df.loc[df.index[i], 'ST_dir'] = False
            elif not df['ST_dir'].iloc[i-1] and df['close'].iloc[i] >= df['final_lb'].iloc[i]:
                df.loc[df.index[i], 'ST_dir'] = True
            else:
                df.loc[df.index[i], 'ST_dir'] = df['ST_dir'].iloc[i-1]
        
        df = df.drop(['hl2', 'matr', 'basic_ub', 'basic_lb', 'final_ub', 'final_lb'], axis=1)
        return df
    
    @staticmethod
    def _calculate_rsi(df, period=14):
        """Calculate RSI"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _add_macd(df, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['MACD'] = ema_fast - ema_slow
        df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        return df
    
    @staticmethod
    def create_sample_data(num_candles=500):
        """
        Create synthetic OHLCV data for testing
        In production, replace with real MT5 data
        """
        
        dates = pd.date_range(end=datetime.now(), periods=num_candles, freq='5T')
        
        # Generate realistic XAUUSD-like price data
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.005, num_candles)
        close_prices = 2000 + np.cumsum(returns)
        
        df = pd.DataFrame({
            'time': dates,
            'open': close_prices + np.random.uniform(-0.2, 0.2, num_candles),
            'high': close_prices + np.random.uniform(0.0, 0.5, num_candles),
            'low': close_prices - np.random.uniform(0.0, 0.5, num_candles),
            'close': close_prices,
            'tick_volume': np.random.uniform(100, 1000, num_candles).astype(int),
        })
        
        return df
    
    @staticmethod
    def load_csv_data(filepath):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            # Ensure required columns
            required = ['open', 'high', 'low', 'close', 'tick_volume']
            if not all(col in df.columns for col in required):
                print(f"❌ CSV missing required columns: {required}")
                return None
            return df
        except Exception as e:
            print(f"❌ Failed to load CSV: {e}")
            return None


print("✅ Data Preparation Loaded")
