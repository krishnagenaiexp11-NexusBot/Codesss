# NexusBot_RiskStrategy.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
import time
import os
import math
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
BARS = 500

# Risk / sizing / margin caps
RISK_PERCENT = 1.0                   # percent of balance risked per trade
MAX_TOTAL_RISK_PERCENT = 100      # total simultaneous risk across open trades (% balance)
MAX_MARGIN_USAGE_PERCENT = 30.0      # max margin usage (% balance)
MIN_FREE_MARGIN = 50.0               # USD free margin buffer
VOLATILITY_ATR_PERIOD = 14

# ATR multipliers
SL_ATR_MULTIPLIER = 1.5
TP_ATR_MULTIPLIER = 3.0
TRAIL_ATR_MULTIPLIER_MOVE = 1.0      # move SL to breakeven when profit > 1*ATR, then trail by 1*ATR

# Strategy indicators
EMA_FAST = 9
EMA_SLOW = 20
SUPERTREND_MULTIPLIER = 3.0

# Logging / Telegram
LOG_DIR = r"C:\Users\krish\Documents\XAUUSD\CSV_analysis"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "trades_log.csv")

# Telegram credentials (as requested - keep this file private)
TELEGRAM_TOKEN = "8299338855:AAGPO7keJwwIkglNghehqkSgTvIyhpa3fQg"
CHAT_ID = "-4912158984"

# Bot behavior
SLEEP_SECONDS = 60
MAGIC = 12345678
COMMENT = "RiskStrategy"

# ---------------- UTIL / TELEGRAM / LOGGING ----------------
def send_telegram_html(text):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=5)
    except Exception as e:
        print("[telegram error]", e)

def ensure_log_file():
    cols = [
        "strategy_name","symbol","direction","entry_time","entry_price","sl","tp",
        "atr_value","ema_fast","ema_slow","supertrend_dir","order_ticket",
        "status","exit_time","exit_price","profit","exit_reason"
    ]
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=cols).to_csv(LOG_FILE, index=False)

def log_trade_open(strategy, symbol, direction, entry_price, sl, tp, metrics, order_ticket):
    ensure_log_file()
    df = pd.read_csv(LOG_FILE)
    entry_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    # prevent duplicate record for same ticket
    if int(order_ticket) in df["order_ticket"].fillna(-1).astype(float).astype(int).tolist():
        return
    new_row = {
        "strategy_name": strategy,
        "symbol": symbol,
        "direction": direction,
        "entry_time": entry_time,
        "entry_price": float(entry_price),
        "sl": float(sl),
        "tp": float(tp),
        "atr_value": float(metrics.get("atr_value", 0)),
        "ema_fast": float(metrics.get("ema_fast", 0)),
        "ema_slow": float(metrics.get("ema_slow", 0)),
        "supertrend_dir": metrics.get("supertrend_dir", "N/A"),
        "order_ticket": int(order_ticket),
        "status": "OPEN"
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

def update_trade_close(order_ticket, exit_price, profit, reason):
    ensure_log_file()
    df = pd.read_csv(LOG_FILE)
    if "order_ticket" not in df.columns:
        return
    matches = df.index[df["order_ticket"] == int(order_ticket)].tolist()
    if not matches:
        # No matching open entry found: append as closed (fallback)
        row = {
            "strategy_name":"RiskStrategy","symbol":SYMBOL,"direction":"",
            "entry_time":"","entry_price":np.nan,"sl":np.nan,"tp":np.nan,
            "atr_value":np.nan,"ema_fast":np.nan,"ema_slow":np.nan,"supertrend_dir":"",
            "order_ticket":int(order_ticket),"status":"CLOSED",
            "exit_time":datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "exit_price":float(exit_price),"profit":float(profit),"exit_reason":reason
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        i = matches[0]
        df.at[i, "exit_time"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        df.at[i, "exit_price"] = float(exit_price)
        df.at[i, "profit"] = float(profit)
        df.at[i, "exit_reason"] = reason
        df.at[i, "status"] = "CLOSED"
    df.to_csv(LOG_FILE, index=False)

def send_entry_alert(strategy, symbol, signal, entry_price, sl, tp, metrics, order_ticket):
    msg = (
        f"⚡ GoldFlash Nexus Bot _ {strategy}\n"
        f"📊 {symbol}\n"
        f"🕒 Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"⚡ Signal: {signal}\n"
        f"✅ Entry: {entry_price:.2f}\n"
        f"🛑 SL: {sl:.2f}\n"
        f"💵 TP: {tp:.2f}\n"
        f"🆔 Order#: {order_ticket}\n\n"
        f"<b>Details</b>\n"
        f"ATR: {metrics.get('atr_value',0):.4f} | "
        f"EMA Fast: {metrics.get('ema_fast',0):.4f} | "
        f"EMA Slow: {metrics.get('ema_slow',0):.4f} | "
        f"SuperTrend Dir: {metrics.get('supertrend_dir','N/A')}"
    )
    send_telegram_html(msg)

def send_close_alert_profit(symbol, order_ticket, profit, duration_seconds):
    dur = str(timedelta(seconds=int(duration_seconds)))
    msg = f"🎯 Nexus 2 booked profit on {symbol} Order: {order_ticket} Profit: {profit:.2f}\nDuration: {dur}"
    send_telegram_html(msg)

def send_close_alert_loss(symbol, order_ticket, profit, duration_seconds):
    dur = str(timedelta(seconds=int(duration_seconds)))
    msg = f"⚡ Nexus 2 cut loss on {symbol} Order: {order_ticket} Loss: {profit:.2f}\nDuration: {dur}"
    send_telegram_html(msg)

# ---------------- MT5 / INDICATORS / SIZING HELPERS ----------------
def initialize_mt5():
    if not mt5.initialize():
        print("MT5 init failed:", mt5.last_error())
        return False
    ok = mt5.symbol_select(SYMBOL, True)
    if not ok:
        print("symbol_select failed for", SYMBOL)
        return False
    return True

def fetch_rates(symbol, timeframe, bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates)==0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calc_atr(df, period=VOLATILITY_ATR_PERIOD):
    df = df.copy()
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = (df['high'] - df['close'].shift(1)).abs()
    df['L-PC'] = (df['low'] - df['close'].shift(1)).abs()
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period, min_periods=1).mean()
    return df

def calc_ema(df, fast=EMA_FAST, slow=EMA_SLOW):
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    return df

def calc_supertrend(df, period=VOLATILITY_ATR_PERIOD, multiplier=SUPERTREND_MULTIPLIER):
    df = df.copy()
    df = calc_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    basic_ub = hl2 + (multiplier * df['ATR'])
    basic_lb = hl2 - (multiplier * df['ATR'])
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    super = pd.Series(index=df.index, dtype='float64')
    dir_bool = pd.Series(index=df.index, dtype='bool')
    super.iloc[0] = np.nan
    dir_bool.iloc[0] = True
    for i in range(1, len(df)):
        if basic_ub.iloc[i] < final_ub.iloc[i-1] or df['close'].iloc[i-1] > final_ub.iloc[i-1]:
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            final_ub.iloc[i] = final_ub.iloc[i-1]
        if basic_lb.iloc[i] > final_lb.iloc[i-1] or df['close'].iloc[i-1] < final_lb.iloc[i-1]:
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            final_lb.iloc[i] = final_lb.iloc[i-1]

        if dir_bool.iloc[i-1] and df['close'].iloc[i] <= final_ub.iloc[i]:
            dir_bool.iloc[i] = False
        elif (not dir_bool.iloc[i-1]) and df['close'].iloc[i] >= final_lb.iloc[i]:
            dir_bool.iloc[i] = True
        else:
            dir_bool.iloc[i] = dir_bool.iloc[i-1]

    df['ST_dir'] = dir_bool
    return df

def get_account_info():
    ai = mt5.account_info()
    if ai is None:
        raise RuntimeError("mt5.account_info() returns None")
    return ai

def estimate_value_per_point(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        return 1.0
    # prefer trade_tick_value/trade_tick_size if present
    try:
        if getattr(info, "trade_tick_value", None) and getattr(info, "trade_tick_size", None):
            return info.trade_tick_value / info.trade_tick_size
        if getattr(info, "tick_value", None):
            return info.tick_value
    except Exception:
        pass
    # fallback: contract_size * point
    contract_size = getattr(info, "trade_contract_size", None) or getattr(info, "contract_size", None) or 100
    return max(0.0001, float(contract_size) * info.point)

def estimate_margin_per_lot(symbol, price, lot=1.0):
    info = mt5.symbol_info(symbol)
    ai = get_account_info()
    leverage = getattr(ai, "leverage", None) or 100
    try:
        if getattr(info, "margin_initial", None):
            return float(info.margin_initial) * lot
    except Exception:
        pass
    contract_size = getattr(info, "trade_contract_size", None) or getattr(info, "contract_size", None) or 100
    est = (price * float(contract_size) * lot) / float(leverage)
    return max(0.0, est)

def compute_lot_for_risk(symbol, entry_price, sl_price, risk_percent=RISK_PERCENT):
    ai = get_account_info()
    balance = float(ai.balance)
    risk_amount = balance * (risk_percent / 100.0)

    info = mt5.symbol_info(symbol)
    if info is None:
        return 0.01
    point = info.point
    sl_points = abs(entry_price - sl_price) / point
    if sl_points <= 0:
        return info.volume_min if hasattr(info, "volume_min") else 0.01

    vpp = estimate_value_per_point(symbol)  # $ per point for 1 lot
    raw_lot = risk_amount / (sl_points * vpp)
    vol_step = getattr(info, "volume_step", 0.01) or 0.01
    vol_min = getattr(info, "volume_min", 0.01) or 0.01
    vol_max = getattr(info, "volume_max", 100.0) or 100.0

    # round down to nearest step
    lot = math.floor(raw_lot / vol_step) * vol_step
    lot = max(vol_min, min(lot, vol_max))
    if lot <= 0:
        lot = vol_min
    return round(lot, 2)

def allowed_to_open_new(symbol, new_trade_margin, new_trade_risk_amount):
    ai = get_account_info()
    balance = float(ai.balance)
    free_margin = float(getattr(ai, "margin_free", balance))
    open_positions = mt5.positions_get()
    total_committed_risk = 0.0
    total_margin_used = 0.0
    if open_positions:
        for p in open_positions:
            try:
                if p.sl and p.sl != 0.0:
                    sl_points = abs(float(p.price_open) - float(p.sl)) / mt5.symbol_info(p.symbol).point
                    vpp = estimate_value_per_point(p.symbol)
                    total_committed_risk += float(p.volume) * sl_points * vpp
                total_margin_used += estimate_margin_per_lot(p.symbol, float(p.price_open), lot=float(p.volume))
            except Exception:
                pass

    total_committed_risk += new_trade_risk_amount
    total_margin_used += new_trade_margin

    max_total_risk = balance * (MAX_TOTAL_RISK_PERCENT / 100.0)
    max_margin_allowed = balance * (MAX_MARGIN_USAGE_PERCENT / 100.0)

    if total_committed_risk > max_total_risk:
        return False, f"risk_cap_exceeded ({total_committed_risk:.2f} > {max_total_risk:.2f})"
    if total_margin_used > max_margin_allowed:
        return False, f"margin_cap_exceeded ({total_margin_used:.2f} > {max_margin_allowed:.2f})"
    if (free_margin - new_trade_margin) < MIN_FREE_MARGIN:
        return False, f"insufficient_free_margin (free {free_margin:.2f}, need buffer {MIN_FREE_MARGIN})"
    return True, "ok"

# ---------------- ORDER HELPERS ----------------
def place_market_order(symbol, direction, lot, price, sl, tp, comment=COMMENT):
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    filling = mt5.ORDER_FILLING_FOK
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot),
        "type": order_type,
        "price": price,
        "sl": round(sl, mt5.symbol_info(symbol).digits),
        "tp": round(tp, mt5.symbol_info(symbol).digits),
        "deviation": 20,
        "magic": MAGIC,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling
    }
    res = mt5.order_send(req)
    return res

def close_position(pos):
    symbol = pos.symbol
    vol = pos.volume
    ticket = pos.ticket
    if pos.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": ticket,
        "symbol": symbol,
        "volume": vol,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": MAGIC,
        "comment": "Nexus_Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK
    }
    return mt5.order_send(req)

# ---------------- MAIN LOOP (ENTRY + MANAGEMENT) ----------------
def main():
    if not initialize_mt5():
        print("MT5 init failed - exit")
        return
    ensure_log_file()
    strategy = "RiskStrategy"
    print("Running", strategy, "on", SYMBOL, "(DEMO recommended)")

    last_open_times = {}  # ticket -> open_time to compute duration

    while True:
        try:
            df = fetch_rates(SYMBOL, TIMEFRAME, BARS)
            if df is None or len(df) < 50:
                time.sleep(SLEEP_SECONDS); continue
            df = calc_ema(df)
            df = calc_supertrend(df)
            df = calc_atr(df, VOLATILITY_ATR_PERIOD)

            last = df.iloc[-1]
            atr = float(last['ATR'])
            ema_fast = float(last['ema_fast'])
            ema_slow = float(last['ema_slow'])
            st_dir = bool(last['ST_dir'])
            price = float(last['close'])

            # signal
            signal = None
            if ema_fast > ema_slow and st_dir:
                signal = "BUY"
            elif ema_fast < ema_slow and not st_dir:
                signal = "SELL"

            # Manage open positions first
            open_positions = mt5.positions_get(symbol=SYMBOL) or []
            # For each open position, enforce trailing & close rules
            for pos in open_positions:
                try:
                    # current market price
                    tick = mt5.symbol_info_tick(pos.symbol)
                    cur_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
                    # entry price and SL/TP from pos
                    entry_price = float(pos.price_open)
                    sl = float(pos.sl) if pos.sl else None
                    tp = float(pos.tp) if pos.tp else None
                    # compute profit in USD
                    profit = float(pos.profit)

                    # determine SL/TP distances in price units
                    # recalc atr for management (use last atr)
                    atr_now = atr
                    sl_distance = SL_ATR_MULTIPLIER * atr_now
                    tp_distance = TP_ATR_MULTIPLIER * atr_now

                    # Check for hitting logical TP/SL by price (bot-close)
                    if pos.type == mt5.POSITION_TYPE_BUY:
                        if cur_price >= (entry_price + tp_distance):
                            # close as TP hit
                            res = close_position(pos)
                            update_trade_close(pos.ticket, cur_price, float(profit), "TP_hit")
                            send_close_alert_profit(pos.symbol, pos.ticket, profit, (datetime.utcnow() - pd.to_datetime(pos.time_msc, unit='ms').to_pydatetime()).total_seconds())
                            continue
                        if cur_price <= (entry_price - sl_distance):
                            res = close_position(pos)
                            update_trade_close(pos.ticket, cur_price, float(profit), "SL_hit")
                            send_close_alert_loss(pos.symbol, pos.ticket, profit, (datetime.utcnow() - pd.to_datetime(pos.time_msc, unit='ms').to_pydatetime()).total_seconds())
                            continue
                        # trailing logic: if profit > 1*ATR, move SL to entry (breakeven) + small buffer
                        if profit >= (estimate_value_per_point(pos.symbol) * TRAIL_ATR_MULTIPLIER_MOVE * atr_now * pos.volume):
                            new_sl = max(sl if sl else -1e9, entry_price + 0.0001)  # small buffer
                            # update SL if improved
                            if new_sl > (sl if sl else -1e9):
                                try:
                                    mt5.order_send({
                                        'action': mt5.TRADE_ACTION_SLTP,
                                        'position': pos.ticket,
                                        'sl': round(new_sl, mt5.symbol_info(pos.symbol).digits),
                                        'tp': pos.tp,
                                        'symbol': pos.symbol
                                    })
                                except Exception:
                                    pass
                    else:
                        # SELL position
                        if cur_price <= (entry_price - tp_distance):
                            res = close_position(pos)
                            update_trade_close(pos.ticket, cur_price, float(profit), "TP_hit")
                            send_close_alert_profit(pos.symbol, pos.ticket, profit, (datetime.utcnow() - pd.to_datetime(pos.time_msc, unit='ms').to_pydatetime()).total_seconds())
                            continue
                        if cur_price >= (entry_price + sl_distance):
                            res = close_position(pos)
                            update_trade_close(pos.ticket, cur_price, float(profit), "SL_hit")
                            send_close_alert_loss(pos.symbol, pos.ticket, profit, (datetime.utcnow() - pd.to_datetime(pos.time_msc, unit='ms').to_pydatetime()).total_seconds())
                            continue
                        if profit >= (estimate_value_per_point(pos.symbol) * TRAIL_ATR_MULTIPLIER_MOVE * atr_now * pos.volume):
                            new_sl = min(sl if sl else 1e9, entry_price - 0.0001)
                            if new_sl < (sl if sl else 1e9):
                                try:
                                    mt5.order_send({
                                        'action': mt5.TRADE_ACTION_SLTP,
                                        'position': pos.ticket,
                                        'sl': round(new_sl, mt5.symbol_info(pos.symbol).digits),
                                        'tp': pos.tp,
                                        'symbol': pos.symbol
                                    })
                                except Exception:
                                    pass
                except Exception as e:
                    print("[manage pos] exception", e)

            # Clean up closed positions (update any orders closed by broker)
            # We will scan CSV for OPEN rows and if their ticket not present in mt5.positions_get, try to fetch last deal profit/time via history_deals_get -> else leave it; to keep things simple we rely on our proactive close logic, so skip here.

            # Now process new signal: compute SL/TP & lot & risk/margin checks
            if signal:
                # compute sl/tp using ATR
                atr_for_entry = atr
                sl_price = price - SL_ATR_MULTIPLIER * atr_for_entry if signal=="BUY" else price + SL_ATR_MULTIPLIER * atr_for_entry
                tp_price = price + TP_ATR_MULTIPLIER * atr_for_entry if signal=="BUY" else price - TP_ATR_MULTIPLIER * atr_for_entry

                # compute lot sized to risk RISK_PERCENT
                lot = compute_lot_for_risk(SYMBOL, price, sl_price, RISK_PERCENT)
                if lot <= 0:
                    print("[sizing] computed lot <=0, skipping")
                    time.sleep(SLEEP_SECONDS); continue

                # compute trade margin & trade risk amount
                new_trade_margin = estimate_margin_per_lot(SYMBOL, price, lot)
                point = mt5.symbol_info(SYMBOL).point
                sl_points = abs(price - sl_price) / point
                vpp = estimate_value_per_point(SYMBOL)
                new_trade_risk_amount = lot * sl_points * vpp

                allowed, reason = allowed_to_open_new(SYMBOL, new_trade_margin, new_trade_risk_amount)
                if not allowed:
                    print("[entry] skip: ", reason)
                    send_telegram_html(f"⚠️ NexusBot skipped entry {SYMBOL} ({signal}) - {reason}")
                    time.sleep(SLEEP_SECONDS); continue

                # get execution price from tick
                tick = mt5.symbol_info_tick(SYMBOL)
                exec_price = tick.ask if signal=="BUY" else tick.bid

                # Place order
                res = place_market_order(SYMBOL, signal, lot, exec_price, sl_price, tp_price, comment=COMMENT)
                if res is None:
                    print("[order] order_send returned None")
                    time.sleep(SLEEP_SECONDS); continue
                if getattr(res, "retcode", None) == 10009:
                    order_id = getattr(res, "order", None) or getattr(res, "deal", None) or 0
                    metrics = {
                        "atr_value": atr_for_entry,
                        "ema_fast": ema_fast,
                        "ema_slow": ema_slow,
                        "supertrend_dir": "UP" if st_dir else "DOWN"
                    }
                    send_entry_alert(strategy, SYMBOL, signal, exec_price, sl_price, tp_price, metrics, order_id)
                    log_trade_open(strategy, SYMBOL, signal, exec_price, sl_price, tp_price, metrics, order_id)
                    print(f"[order] placed {signal} ticket={order_id} lot={lot} price={exec_price}")
                else:
                    print("[order] failed retcode:", getattr(res, "retcode", None))

            time.sleep(SLEEP_SECONDS)

        except KeyboardInterrupt:
            print("Interrupted by user")
            break
        except Exception as e:
            print("[main loop] exception:", e)
            time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()
