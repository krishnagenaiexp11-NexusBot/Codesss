import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

# ---------------- CONFIG ----------------
SYMBOL = "XAUUSD.s"
TIMEFRAME = mt5.TIMEFRAME_M1
HIGHER_TIMEFRAME = mt5.TIMEFRAME_M5  # For trend confirmation
CANDLES = 200

LOT = 0.01
DEVIATION = 10
CONFIDENCE_THRESHOLD = 25

STANDARD_TP_PIPS = 80
STANDARD_SL_PIPS = 60
STOPS_LEVEL = 50

VOLUME_MULTIPLIER_THRESHOLD = 1.2
VOLATILITY_ATR_PERIOD = 14

TELEGRAM_TOKEN = "8299338855:AAGPO7keJwwIkglNghehqkSgTvIyhpa3fQg"
CHAT_ID = "-4912158984"

SLEEP_SECONDS = 60

recent_results = []

def send_telegram(text):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, json={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"})
        resp.raise_for_status()
        print("[telegram] sent")
    except Exception as e:
        print("[telegram error]", e)

def initialize_mt5():
    if not mt5.initialize():
        print("❌ MT5 initialization failed:", mt5.last_error())
        return False
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None or not mt5.symbol_select(SYMBOL, True):
        print(f"❌ Symbol {SYMBOL} not found or cannot select")
        mt5.shutdown()
        return False
    print(f"✅ MT5 initialized. Symbol: {SYMBOL}, Point: {symbol_info.point}, Digits: {symbol_info.digits}")
    print(f"Broker min stop: {STOPS_LEVEL} points ({STOPS_LEVEL * symbol_info.point:.2f} USD)")
    return True

def fetch_mt5_data(timeframe=TIMEFRAME, bars=CANDLES):
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        print(f"❌ No data for {SYMBOL} on TF {timeframe}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calculate_indicators(df):
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['momentum'] = df['close'] - df['close'].shift(10)
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=VOLATILITY_ATR_PERIOD).mean()
    df['avg_volume'] = df['real_volume'].rolling(20).mean()
    return df

def update_trade_results(order_result):
    global recent_results
    if order_result is not None:
        trade_won = order_result.retcode == 10009  # Simplified assumption
        recent_results.append(trade_won)
        if len(recent_results) > 10:
            recent_results.pop(0)

def get_dynamic_sl_tp_multipliers(atr):
    if len(recent_results) < 3:
        return 1.0, 1.0
    losing_streak = 0
    for res in reversed(recent_results):
        if res is False:
            losing_streak += 1
        else:
            break
    if losing_streak >= 2:
        sl_mult = max(1.5, atr / (STANDARD_SL_PIPS * 0.01))
        tp_mult = max(1.5, atr / (STANDARD_TP_PIPS * 0.01))
    else:
        sl_mult, tp_mult = 1.0, 1.0
    return sl_mult, tp_mult

def generate_signal(df, higher_df):
    latest = df.iloc[-1]
    ema9, ema20, rsi, price = latest['EMA9'], latest['EMA20'], latest['RSI'], latest['close']
    momentum = latest['momentum'] if 'momentum' in latest else 0
    avg_vol = latest['avg_volume'] if 'avg_volume' in latest else 0
    current_vol = latest['real_volume']
    atr = latest['ATR'] if 'ATR' in latest else 0

    ht_ema9 = higher_df['EMA9'].iloc[-1]
    ht_ema20 = higher_df['EMA20'].iloc[-1]

    # Debug print indicator values
    print(f"DEBUG - EMA9: {ema9:.2f}, EMA20: {ema20:.2f}, RSI: {rsi:.2f}, Momentum: {momentum:.2f}, Current Vol: {current_vol}, Avg Vol: {avg_vol:.2f}, ATR: {atr:.4f}")

    signal, confidence, reason = "HOLD", 0, "Neutral"

    # Volume filter temporarily disabled for debugging
    # if avg_vol == 0 or current_vol < VOLUME_MULTIPLIER_THRESHOLD * avg_vol:
    #     reason = "Volume below threshold"
    #     return "HOLD", 0, reason, ema9, ema20, rsi, price, atr

    is_high_volatility = atr > 1.5 * (STANDARD_SL_PIPS * 0.01)

    if not is_high_volatility:
        if ema9 > ema20 and rsi > 45 and momentum > 0 and ht_ema9 > ht_ema20:
            signal = "BUY"
            confidence = min(100, (rsi - 45) * 3 + (ema9 - ema20) * 10 + momentum * 10)
            reason = "Normal Vol: EMA up, RSI 45+, Pos Momentum"
        elif ema9 < ema20 and rsi < 55 and momentum < 0 and ht_ema9 < ht_ema20:
            signal = "SELL"
            confidence = min(100, (55 - rsi) * 3 + (ema20 - ema9) * 10 + abs(momentum) * 10)
            reason = "Normal Vol: EMA down, RSI 55-, Neg Momentum"
    else:
        if ema9 > ema20 and rsi > 40 and ht_ema9 > ht_ema20:
            signal = "BUY"
            confidence = min(100, (rsi - 40) * 1.5 + (ema9 - ema20) * 10 + momentum * 5 + atr * 15)
            reason = "High Vol: EMA up, RSI >40, momentum & ATR confirm"
        elif ema9 < ema20 and rsi < 60 and ht_ema9 < ht_ema20:
            signal = "SELL"
            confidence = min(100, (60 - rsi) * 1.5 + (ema20 - ema9) * 10 + abs(momentum) * 5 + atr * 15)
            reason = "High Vol: EMA down, RSI <60, momentum & ATR confirm"

    return signal, confidence, reason, ema9, ema20, rsi, price, atr


def place_order(signal, price, point, digits, atr):
    sl_mult, tp_mult = get_dynamic_sl_tp_multipliers(atr)

    tp_distance = max(STANDARD_TP_PIPS * tp_mult, STOPS_LEVEL) * point
    sl_distance = max(STANDARD_SL_PIPS * sl_mult, STOPS_LEVEL) * point

    if signal == "BUY":
        tp = round(price + tp_distance, digits)
        sl = round(price - sl_distance, digits)
        order_type = mt5.ORDER_TYPE_BUY
    elif signal == "SELL":
        tp = round(price - tp_distance, digits)
        sl = round(price + sl_distance, digits)
        order_type = mt5.ORDER_TYPE_SELL
    else:
        return None, None, None

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT,
        "type": order_type,
        "price": round(price, digits),
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": 234000,
        "comment": "GoldFlash Scalper",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK
    }

    result = mt5.order_send(request)
    print(f"[Order send] price={price} TP={tp} SL={sl} RT={result.retcode if result else ''}")
    return result, tp, sl


def main_loop():
    if not initialize_mt5():
        return

    symbol_info = mt5.symbol_info(SYMBOL)
    point = symbol_info.point
    digits = symbol_info.digits
    last_trade_bar = None

    print(f"🚀 GoldFlash Scalper running on {SYMBOL} (1min TF) with debug prints & relaxed thresholds ...")

    while True:
        try:
            df = fetch_mt5_data(TIMEFRAME)
            if df is None or len(df) < 30:
                time.sleep(SLEEP_SECONDS)
                continue

            higher_df = fetch_mt5_data(HIGHER_TIMEFRAME)
            if higher_df is None or len(higher_df) < 30:
                time.sleep(SLEEP_SECONDS)
                continue

            df = calculate_indicators(df)
            higher_df = calculate_indicators(higher_df)

            latest_bar_time = df['time'].iloc[-1]

            if last_trade_bar == latest_bar_time:
                time.sleep(SLEEP_SECONDS)
                continue

            signal, conf, reason, ema9, ema20, rsi, price, atr = generate_signal(df, higher_df)

            if conf < CONFIDENCE_THRESHOLD or signal == "HOLD":
                print(f"[{datetime.now()}] Skipping - Confidence {conf:.1f}% or No valid signal: {reason}")
                last_trade_bar = latest_bar_time
                continue

            order_result, tp, sl = place_order(signal, price, point, digits, atr)

            update_trade_results(order_result)

            pip_tp = abs(tp - price) / point if tp else 0
            pip_sl = abs(price - sl) / point if sl else 0
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            emoji = "🔵" if signal == "BUY" else ("🔴" if signal == "SELL" else "⚪")

            msg = (
                f"{emoji} *GoldFlash Scalper*\n"
                f"📊 {SYMBOL} (1min with debug)\n"
                f"🕒 {now}\n"
                f"⚡ Signal: *{signal}*\n"
                f"Reason: {reason}\n"
                f"EMA Fast: {ema9:.2f} | EMA Slow: {ema20:.2f} | RSI: {rsi:.1f}\n"
                f"💰 Price: {price:.2f}\n"
                f"ATR: {atr:.4f}\n"
                f"Confidence: {conf:.1f}%\n"
                f"✅ Entry: {price:.2f}\n"
                f"💵 TP: {tp:.2f} ({pip_tp:.0f}p)\n"
                f"🛑 SL: {sl:.2f} ({pip_sl:.0f}p)\n"
            )
            print(msg.replace("\n", " | "))

            if order_result and order_result.retcode == 10009:
                send_telegram(msg)
                print(f"✅ Order placed successfully at {price}")
            else:
                print(f"⚠️ Order failed. Retcode: {order_result.retcode if order_result else 'N/A'}")

            last_trade_bar = latest_bar_time
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print("[main loop error]", e)
            time.sleep(60)

if __name__ == "__main__":
    main_loop()
