import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

# ---------------- CONFIG ---------------- #
SYMBOLS = ["XAUUSD.s", "XAGUSD.s", "NAS100.s"]
LOT = 0.05
DEVIATION = 10
BARS = 300
MAX_PYRAMID_POSITIONS = 3
SLEEP_SECONDS = 60

# ATR/SL/TP multipliers
ATR_PERIOD = 10
SL_ATR_MULTIPLIER = 1.5
TP_ATR_MULTIPLIER = 3.0
TRAILING_SL_ATR_MULTIPLIER = 1.0
TRAILING_TP_ATR_MULTIPLIER = 1.0

# Strategy timeframes
TIMEFRAMES = {
    "scalper": mt5.TIMEFRAME_M5,
    "mean_reversion": mt5.TIMEFRAME_M15,
    "breakout": mt5.TIMEFRAME_M30,
    "vwap": mt5.TIMEFRAME_M15,
    "pair_trade": mt5.TIMEFRAME_M5
}

# Telegram
TELEGRAM_TOKEN = "8299338855:AAGPO7keJwwIkglNghehqkSgTvIyhpa3fQg"
CHAT_ID = "-4912158984"

# Log file
LOG_FILE = "GoldFlashNexus_log.txt"

# ---------------- LOGGING ---------------- #
def log_status(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    print(full_msg)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")

# ---------------- TELEGRAM ---------------- #
def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"})
        resp.raise_for_status()
    except Exception as e:
        log_status(f"[telegram error] {e}")

# ---------------- MT5 INIT ---------------- #
def initialize_mt5():
    if not mt5.initialize():
        log_status(f"❌ MT5 init failed: {mt5.last_error()}")
        return False
    for sym in SYMBOLS:
        if mt5.symbol_select(sym, True):
            log_status(f"✅ {sym} ready")
        else:
            log_status(f"❌ {sym} not available")
            mt5.shutdown()
            return False
    return True

# ---------------- FETCH DATA ---------------- #
def fetch_data(symbol, timeframe, bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        log_status(f"❌ No data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['volume'] = df['tick_volume'] if 'tick_volume' in df.columns else 1
    return df

# ---------------- INDICATORS ---------------- #
def calculate_atr(df, period=ATR_PERIOD):
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df

def calculate_ema(df, period):
    return df['close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    df['RSI'] = 100 - (100 / (1 + ma_up/ma_down))
    return df

def calculate_bollinger(df, period=20, std=2):
    df['MA'] = df['close'].rolling(period).mean()
    df['BB_up'] = df['MA'] + std*df['close'].rolling(period).std()
    df['BB_low'] = df['MA'] - std*df['close'].rolling(period).std()
    return df

# ---------------- ORDER FUNCTIONS ---------------- #
def place_order(symbol, order_type, volume, price, sl, tp):
    type_map = {"BUY": mt5.ORDER_TYPE_BUY, "SELL": mt5.ORDER_TYPE_SELL}
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": type_map[order_type],
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": 7654321,
        "comment": "GoldFlash Nexus",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK
    }
    res = mt5.order_send(request)
    return res

def modify_sl_tp(position, new_sl, new_tp):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position.ticket,
        "sl": new_sl,
        "tp": new_tp,
        "symbol": position.symbol
    }
    res = mt5.order_send(request)
    return res

def close_position(position):
    tick = mt5.symbol_info_tick(position.symbol)
    price = tick.bid if position.type == mt5.POSITION_TYPE_BUY else tick.ask
    order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position.ticket,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": order_type,
        "price": price,
        "deviation": DEVIATION,
        "magic": 7654321,
        "comment": "Close Position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK
    }
    res = mt5.order_send(request)
    if res.retcode == 10009:
        log_status(f"✂️ Closed position #{position.ticket} for {position.symbol} at {price}")
    return res

# ---------------- STRATEGIES ---------------- #
def scalper_strategy(df):
    df['EMA_fast'] = calculate_ema(df, 9)
    df['EMA_slow'] = calculate_ema(df, 21)
    df = calculate_rsi(df)
    last = df.iloc[-1]
    if last['EMA_fast'] > last['EMA_slow'] and last['RSI'] > 55:
        return "BUY"
    elif last['EMA_fast'] < last['EMA_slow'] and last['RSI'] < 45:
        return "SELL"
    else:
        return "HOLD"

def mean_reversion_strategy(df):
    df = calculate_bollinger(df)
    last = df.iloc[-1]
    if last['close'] < last['BB_low']:
        return "BUY"
    elif last['close'] > last['BB_up']:
        return "SELL"
    else:
        return "HOLD"

def breakout_strategy(df, lookback=20):
    last = df.iloc[-1]
    high = df['high'].rolling(lookback).max().iloc[-2]
    low = df['low'].rolling(lookback).min().iloc[-2]
    if last['close'] > high:
        return "BUY"
    elif last['close'] < low:
        return "SELL"
    else:
        return "HOLD"

def vwap_strategy(df):
    if 'volume' not in df.columns or df['volume'].isnull().all():
        return "HOLD"
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    last = df.iloc[-1]
    if last['close'] > vwap.iloc[-1]:
        return "BUY"
    elif last['close'] < vwap.iloc[-1]:
        return "SELL"
    else:
        return "HOLD"

# ---------------- MAIN LOOP ---------------- #
def main():
    if not initialize_mt5():
        return
    active_positions = {}  # ticket -> {symbol, strategy, signal}
    
    while True:
        try:
            now = datetime.now()
            for symbol in SYMBOLS:
                df = fetch_data(symbol, TIMEFRAMES['scalper'], BARS)
                if df is None:
                    continue
                df = calculate_atr(df)
                atr = df['ATR'].iloc[-1]
                last_close = df['close'].iloc[-1]
                digits = mt5.symbol_info(symbol).digits

                # --- Strategies --- #
                signals = {
                    "Scalper": scalper_strategy(df),
                    "MeanReversion": mean_reversion_strategy(df),
                    "Breakout": breakout_strategy(df),
                    "VWAP": vwap_strategy(df)
                }
                log_status(f"{symbol} signals: {signals}")

                # --- Place order if no existing position OR pyramid allowed --- #
                current_positions = [p for p in mt5.positions_get(symbol=symbol) or []]
                if len(current_positions) < MAX_PYRAMID_POSITIONS:
                    for strategy, signal in signals.items():
                        if signal in ["BUY", "SELL"]:
                            # Check already placed by this strategy
                            already_open = [p for p in current_positions if strategy in p.comment]
                            if already_open:
                                continue
                            sl = round(last_close - SL_ATR_MULTIPLIER*atr if signal=="BUY" else last_close + SL_ATR_MULTIPLIER*atr, digits)
                            tp = round(last_close + TP_ATR_MULTIPLIER*atr if signal=="BUY" else last_close - TP_ATR_MULTIPLIER*atr, digits)
                            res = place_order(symbol, signal, LOT, last_close, sl, tp)
                            if res.retcode == 10009:
                                ticket = res.order
                                active_positions[ticket] = {"symbol":symbol,"strategy":strategy,"signal":signal,"sl":sl,"tp":tp}
                                log_status(f"🚀 Order placed #{ticket} | {symbol} | {strategy} | {signal} | Entry: {last_close} | SL: {sl} | TP: {tp}")
                                send_telegram(f"""
⚡ GoldFlash Nexus - ({strategy})
📊 {symbol} ({TIMEFRAMES['scalper']})
🕒 Time: {now.strftime('%Y-%m-%d %H:%M:%S')}
⚡ Signal: {signal}
💳 Order #: {ticket}

✅ Entry: {last_close}
💵 TP: {tp} - {abs(tp-last_close)*10:.1f} Pips
🛑 SL: {sl} - {abs(last_close-sl)*10:.1f} Pips
""")

                # --- Trailing SL & TP and profit exit --- #
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    for pos in positions:
                        ticket = pos.ticket
                        data = active_positions.get(ticket, {})
                        current_price = mt5.symbol_info_tick(symbol).last
                        new_sl, new_tp = pos.sl, pos.tp
                        # Trailing SL
                        if pos.type == mt5.POSITION_TYPE_BUY:
                            trail_sl = round(current_price - TRAILING_SL_ATR_MULTIPLIER*atr, digits)
                            if trail_sl > pos.sl:
                                new_sl = trail_sl
                            trail_tp = round(current_price + TRAILING_TP_ATR_MULTIPLIER*atr, digits)
                            if trail_tp > pos.tp:
                                new_tp = trail_tp
                        else:
                            trail_sl = round(current_price + TRAILING_SL_ATR_MULTIPLIER*atr, digits)
                            if trail_sl < pos.sl:
                                new_sl = trail_sl
                            trail_tp = round(current_price - TRAILING_TP_ATR_MULTIPLIER*atr, digits)
                            if trail_tp < pos.tp:
                                new_tp = trail_tp

                        # Update SL/TP if changed
                        if new_sl != pos.sl or new_tp != pos.tp:
                            modify_sl_tp(pos, new_sl, new_tp)
                            log_status(f"🔧 Trailing SL/TP updated for Order #{ticket} | New SL: {new_sl} | New TP: {new_tp}")
                            send_telegram(f"🔧 Trailing SL/TP updated for Order #{ticket}\n🛑 SL: {new_sl}\n💵 TP: {new_tp}")

                        # Close profitable positions automatically
                        profit = (current_price - pos.price_open) * 10 if pos.type == mt5.POSITION_TYPE_BUY else (pos.price_open - current_price) * 10
                        if profit >= abs(pos.tp - pos.price_open)*10:
                            close_position(pos)
                            if ticket in active_positions:
                                active_positions.pop(ticket)

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            log_status(f"[main loop error] {e}")
            time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()
