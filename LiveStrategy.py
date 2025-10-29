# MirrorTrade_Strategy1_fixed.py
# Full-featured Nexus/Mirror trading bot
# - Dual-account mirroring (main + mirror)
# - ATR SL/TP, trailing, breakeven, partial close
# - Smart monitor_and_roll: close when profit >= absolute OR progressed >= ratio to TP AND EMA slowing
# - Safe stops to avoid 10016
# - Robust CSV logging + Telegram alerts
# - Timezone aware (UTC)
#
# NOTE: Test on DEMO. Keep both MT5 terminals open & logged in.

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
import time
import os
import math
from datetime import datetime, timedelta, timezone

# ---------------- USER CONFIG ----------------
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
BARS = 500

# Risk / sizing / margin caps (Safer for Gold)
RISK_PERCENT = 0.35                   # was 1.0 → keeps losses small
MAX_TOTAL_RISK_PERCENT = 25           # was 100 → no overexposure
MAX_MARGIN_USAGE_PERCENT = 10.0       # was 30 → prevents margin burnouts
MIN_FREE_MARGIN = 150.0               # was 50 → protects account

VOLATILITY_ATR_PERIOD = 14

# ATR-based Stop Loss & Take Profit (Correct for XAU M5)
SL_ATR_MULTIPLIER = 1.8               # was 0.75 → survive gold spikes
TP_ATR_MULTIPLIER = 1.2               # was 3.0 → increases TP hit-rate
MIN_SL_ATR_MULTIPLIER = 1.6           # safety floor to avoid tiny stops

# Trailing / breakeven / partial close logic
TRAIL_ATR_MULTIPLIER_MOVE = 0.8       # was 1.0 → breakeven earlier
TRAIL_STEP_ATR = 1.0
BREAKEVEN_ACTIVATE_ATR = 0.8          # was 1.0 → secure profits sooner

PARTIAL_CLOSE_ENABLED = True
PARTIAL_CLOSE_TRIGGER_ATR = 1.0
PARTIAL_CLOSE_PCT = 0.33              # was 0.5 → avoid over-cutting winning legs

# Monitoring & dynamic close logic
PROFIT_CLOSE_USD = 0.0                # was 1.0 → no early tiny exits
MONITOR_INTERVAL = 15                 # was 30 → faster trailing logic
DYNAMIC_TP_PROGRESS_RATIO = 0.45      # was 0.50 → exit earlier when trend weakens

EMA_SLOWING_LOOKBACK = 1

# Strategy indicators (unchanged, entry logic stays same)
EMA_FAST = 9
EMA_SLOW = 20
SUPERTREND_MULTIPLIER = 3.0

# Logging / Telegram
LOG_DIR = r"C:\Users\krish\Documents\XAUUSD\CSV_analysis"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "trades_log.csv")

# Telegram credentials (keep private)
TELEGRAM_TOKEN = "8299338855:AAGPO7keJwwIkglNghehqkSgTvIyhpa3fQg"
CHAT_ID = "-4912158984"
TELEGRAM_ON_LOGS = True   # send telegram for each log write

# MetaTrader terminal paths
MT5_MAIN_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
MT5_MIRROR_PATH = r"D:\Metatrader - Main\terminal64.exe"

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

# ---------------------------------------------------------
# Telegram alert wrappers  (place right after send_telegram_html)
# ---------------------------------------------------------
def send_entry_alert(strategy, symbol, direction, entry_price, sl, tp, metrics, order_ticket):
    """Send Telegram alert when a new position is opened."""
    try:
        msg = (
            f"📈 <b>{strategy}</b> | {symbol}\n"
            f"🟢 <b>{direction}</b> | Ticket: <code>{order_ticket}</code>\n"
            f"💰 Entry: {entry_price:.2f}\n"
            f"🛑 SL: {sl:.2f} | 🎯 TP: {tp:.2f}\n"
            f"📊 ATR: {metrics.get('atr_value',0):.2f} | "
            f"EMAf: {metrics.get('ema_fast',0):.2f} | EMAs: {metrics.get('ema_slow',0):.2f}\n"
            f"⚙️ ST Dir: {metrics.get('supertrend_dir','N/A')}"
        )
        send_telegram_html(msg)
    except Exception as e:
        print("[send_entry_alert] exception:", e)


def send_close_alert_profit(symbol, ticket, profit, duration_sec):
    """Send Telegram alert when a trade closes in profit."""
    try:
        mins = duration_sec / 60.0
        msg = (
            f"✅ <b>PROFIT CLOSE</b> | {symbol}\n"
            f"📄 Ticket: <code>{ticket}</code>\n"
            f"💵 Profit: ${profit:.2f}\n"
            f"⏱️ Duration: {mins:.1f} mins"
        )
        send_telegram_html(msg)
    except Exception as e:
        print("[send_close_alert_profit] exception:", e)


def send_close_alert_loss(symbol, ticket, profit, duration_sec):
    """Send Telegram alert when a trade closes in loss."""
    try:
        mins = duration_sec / 60.0
        msg = (
            f"❌ <b>LOSS CLOSE</b> | {symbol}\n"
            f"📄 Ticket: <code>{ticket}</code>\n"
            f"💸 P/L: ${profit:.2f}\n"
            f"⏱️ Duration: {mins:.1f} mins"
        )
        send_telegram_html(msg)
    except Exception as e:
        print("[send_close_alert_loss] exception:", e)

def send_partial_close_alert(symbol, ticket, volume, profit):
    try:
        msg = f"⚡ Partial close {symbol} ticket {ticket}\nVol {volume} Profit ${profit:.2f}"
        send_telegram_html(msg)
    except Exception:
        pass

def send_mirror_alert(symbol, direction, lot):
    try:
        msg = f"🪞 Mirrored {symbol} {direction} lot {lot}"
        send_telegram_html(msg)
    except Exception:
        pass

# Logging helpers
def ensure_log_file():
    cols = [
        "strategy_name","symbol","direction","entry_time","entry_price","sl","tp",
        "atr_value","ema_fast","ema_slow","supertrend_dir","order_ticket",
        "status","exit_time","exit_price","profit","exit_reason"
    ]
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=cols).to_csv(LOG_FILE, index=False)

def _send_log_telegram(action, ticket, details):
    if not TELEGRAM_ON_LOGS:
        return
    try:
        send_telegram_html(f"🗂️ Log {action}: Ticket {ticket}\n{details}")
    except Exception:
        pass

def log_trade_open(strategy, symbol, direction, entry_price, sl, tp, metrics, order_ticket):
    ensure_log_file()
    try:
        df = pd.read_csv(LOG_FILE)
    except Exception:
        df = pd.DataFrame()
    entry_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    try:
        order_ticket_int = int(order_ticket)
    except Exception:
        try:
            order_ticket_int = int(float(order_ticket))
        except Exception:
            order_ticket_int = 0
    # prevent duplicate
    if "order_ticket" in df.columns:
        try:
            if order_ticket_int in df["order_ticket"].fillna(-1).astype(float).astype(int).tolist():
                # already logged
                return
        except Exception:
            pass
    new_row = {
        "strategy_name": strategy,
        "symbol": symbol,
        "direction": direction,
        "entry_time": entry_time,
        "entry_price": float(entry_price),
        "sl": float(sl) if sl is not None else np.nan,
        "tp": float(tp) if tp is not None else np.nan,
        "atr_value": float(metrics.get("atr_value", 0)),
        "ema_fast": float(metrics.get("ema_fast", 0)),
        "ema_slow": float(metrics.get("ema_slow", 0)),
        "supertrend_dir": metrics.get("supertrend_dir", "N/A"),
        "order_ticket": order_ticket_int,
        "status": "OPEN",
        "exit_time": "",
        "exit_price": np.nan,
        "profit": np.nan,
        "exit_reason": ""
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)
    details = f"{symbol} {direction} Entry:{entry_price:.2f} SL:{sl:.2f} TP:{tp:.2f}" if (sl is not None and tp is not None) else f"{symbol} {direction} Entry:{entry_price:.2f}"
    print(f"[log] Opened -> Ticket {order_ticket_int} | {details}")
    _send_log_telegram("OPEN", order_ticket_int, details)

def update_trade_close(order_ticket, exit_price, profit, reason):
    ensure_log_file()
    try:
        df = pd.read_csv(LOG_FILE)
    except Exception:
        df = pd.DataFrame()
    try:
        order_ticket_int = int(order_ticket)
    except Exception:
        try:
            order_ticket_int = int(float(order_ticket))
        except Exception:
            order_ticket_int = order_ticket
    if "order_ticket" in df.columns:
        matches = df.index[df["order_ticket"] == int(order_ticket_int)].tolist()
    else:
        matches = []
    if not matches:
        # fallback: append closed row
        row = {
            "strategy_name":"RiskStrategy","symbol":SYMBOL,"direction":"",
            "entry_time":"","entry_price":np.nan,"sl":np.nan,"tp":np.nan,
            "atr_value":np.nan,"ema_fast":np.nan,"ema_slow":np.nan,"supertrend_dir":"",
            "order_ticket":int(order_ticket_int) if isinstance(order_ticket_int, (int, np.integer)) else order_ticket_int,
            "status":"CLOSED",
            "exit_time":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "exit_price":float(exit_price),"profit":float(profit),"exit_reason":reason
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        i = matches[0]
        df.at[i, "exit_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        df.at[i, "exit_price"] = float(exit_price)
        df.at[i, "profit"] = float(profit)
        df.at[i, "exit_reason"] = reason
        df.at[i, "status"] = "CLOSED"
    df.to_csv(LOG_FILE, index=False)
    print(f"[log] Closed -> Ticket {order_ticket_int} | Exit:{exit_price:.4f} Profit:{profit:.2f} Reason:{reason}")
    _send_log_telegram("CLOSE", order_ticket_int, f"Exit:{exit_price:.4f} Profit:{profit:.2f} Reason:{reason}")

# ---------------- MT5 / INDICATORS / SIZING HELPERS ----------------
def initialize_mt5_with_path(path):
    try:
        if not mt5.initialize(path):
            print(f"[mt5 init] failed for {path} ->", mt5.last_error())
            return False
        # Attempt to select symbol (best-effort)
        try:
            mt5.symbol_select(SYMBOL, True)
        except Exception:
            pass
        return True
    except Exception as e:
        print("[mt5 init] exception:", e)
        return False

def fetch_rates(symbol, timeframe, bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
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
    supertrend_dir = pd.Series(index=df.index, dtype='bool')
    supertrend_dir.iloc[0] = True
    for i in range(1, len(df)):
        if basic_ub.iloc[i] < final_ub.iloc[i-1] or df['close'].iloc[i-1] > final_ub.iloc[i-1]:
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            final_ub.iloc[i] = final_ub.iloc[i-1]
        if basic_lb.iloc[i] > final_lb.iloc[i-1] or df['close'].iloc[i-1] < final_lb.iloc[i-1]:
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            final_lb.iloc[i] = final_lb.iloc[i-1]

        if supertrend_dir.iloc[i-1] and df['close'].iloc[i] <= final_ub.iloc[i]:
            supertrend_dir.iloc[i] = False
        elif (not supertrend_dir.iloc[i-1]) and df['close'].iloc[i] >= final_lb.iloc[i]:
            supertrend_dir.iloc[i] = True
        else:
            supertrend_dir.iloc[i] = supertrend_dir.iloc[i-1]
    df['ST_dir'] = supertrend_dir
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
    try:
        if getattr(info, "trade_tick_value", None) and getattr(info, "trade_tick_size", None):
            return info.trade_tick_value / info.trade_tick_size
        if getattr(info, "tick_value", None):
            return info.tick_value
    except Exception:
        pass
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
        return getattr(info, "volume_min", 0.01)

    vpp = estimate_value_per_point(symbol)  # $ per point for 1 lot
    raw_lot = risk_amount / (sl_points * vpp)
    vol_step = getattr(info, "volume_step", 0.01) or 0.01
    vol_min = getattr(info, "volume_min", 0.01) or 0.01
    vol_max = getattr(info, "volume_max", 100.0) or 100.0

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

    print(f"[risk-check] balance={balance:.2f} free_margin={free_margin:.2f} "
          f"existing_risk={total_committed_risk - new_trade_risk_amount:.2f} "
          f"new_trade_risk={new_trade_risk_amount:.2f} total_committed_risk={total_committed_risk:.2f} "
          f"max_total_risk={max_total_risk:.2f} total_margin_used={total_margin_used:.2f} max_margin_allowed={max_margin_allowed:.2f}")

    if total_committed_risk > max_total_risk:
        return False, f"risk_cap_exceeded ({total_committed_risk:.2f} > {max_total_risk:.2f})"
    if total_margin_used > max_margin_allowed:
        return False, f"margin_cap_exceeded ({total_margin_used:.2f} > {max_margin_allowed:.2f})"
    if (free_margin - new_trade_margin) < MIN_FREE_MARGIN:
        return False, f"insufficient_free_margin (free {free_margin:.2f}, need buffer {MIN_FREE_MARGIN})"
    return True, "ok"

# ---------------- ORDER HELPERS ----------------
def safe_stops(symbol, entry_price, sl, tp):
    """
    Ensure SL/TP respects broker stop-levels.
    Adjusts SL/TP to at least the broker's minimal stop distance.
    """
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            return sl, tp
        min_level = getattr(info, "trade_stops_level", None)
        point = getattr(info, "point", None)
        if min_level is None or point is None:
            return sl, tp
        min_distance = float(min_level) * float(point)
        # SL adjustment
        if sl is not None:
            if entry_price > sl and (entry_price - sl) < min_distance:
                sl = entry_price - min_distance
            elif entry_price < sl and (sl - entry_price) < min_distance:
                sl = entry_price + min_distance
        # TP adjustment
        if tp is not None:
            if tp > entry_price and (tp - entry_price) < min_distance:
                tp = entry_price + min_distance
            elif tp < entry_price and (entry_price - tp) < min_distance:
                tp = entry_price - min_distance
        return sl, tp
    except Exception as e:
        print("[safe_stops] exception", e)
        return sl, tp

def place_market_order(symbol, direction, lot, price, sl, tp, comment=COMMENT):
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    filling = mt5.ORDER_FILLING_FOK
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot),
        "type": order_type,
        "price": price,
        "sl": round(sl, mt5.symbol_info(symbol).digits) if sl is not None else None,
        "tp": round(tp, mt5.symbol_info(symbol).digits) if tp is not None else None,
        "deviation": 20,
        "magic": MAGIC,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling
    }
    try:
        return mt5.order_send(req)
    except Exception as e:
        print("[order_send] exception", e)
        return None

def close_position(pos, close_volume=None):
    symbol = pos.symbol
    vol = pos.volume if close_volume is None else close_volume
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
        "symbol": pos.symbol,
        "volume": float(vol),
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": MAGIC,
        "comment": "Nexus_Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK
    }
    try:
        return mt5.order_send(req)
    except Exception as e:
        print("[close_position] exception", e)
        return None

# ---------------- MIRRORING ----------------
def mirror_order_to_second_account(direction, lot, sl, tp):
    """
    Switch connection to mirror MT5 terminal, place identical order, then switch back to main.
    """
    try:
        # Shutdown main connection
        try:
            mt5.shutdown()
        except Exception:
            pass
        time.sleep(0.6)

        # Init mirror
        if not initialize_mt5_with_path(MT5_MIRROR_PATH):
            print("[mirror] ❌ Failed to init mirror terminal:", mt5.last_error())
            # Try to reconnect to main before leaving
            time.sleep(0.6)
            initialize_mt5_with_path(MT5_MAIN_PATH)
            return

        print("[mirror] ✅ Connected to mirror terminal.")

        # Ensure symbol selected (best-effort)
        try:
            mt5.symbol_select(SYMBOL, True)
        except Exception:
            pass

        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            print("[mirror] ❌ No tick on mirror terminal for symbol", SYMBOL)
            mt5.shutdown()
            initialize_mt5_with_path(MT5_MAIN_PATH)
            return

        exec_price = tick.ask if direction == "BUY" else tick.bid
        sl2, tp2 = safe_stops(SYMBOL, exec_price, sl, tp)
        res = place_market_order(SYMBOL, direction, lot, exec_price, sl2, tp2, comment=COMMENT + "_MIRROR")
        if res is None:
            print("[mirror] ❌ Mirror order_send returned None")
        else:
            print("[mirror] ✅ Mirror order result:", res)
            send_mirror_alert(SYMBOL, direction, lot)
    except Exception as e:
        print("[mirror] exception:", e)
    finally:
        # Always reconnect to main terminal
        try:
            mt5.shutdown()
        except Exception:
            pass
        time.sleep(0.6)
        if not initialize_mt5_with_path(MT5_MAIN_PATH):
            print("[mirror] ⚠️ Could not reconnect to main terminal after mirroring:", mt5.last_error())
        else:
            print("[mirror] 🔁 Reconnected to main terminal.")

# ---------------- MONITOR & ROLL (IMPROVED) ----------------
def monitor_and_roll(df_latest, atr_latest):
    """
    Run every MONITOR_INTERVAL seconds.
    Close positions when:
      - profit >= PROFIT_CLOSE_USD (absolute), OR
      - position has progressed >= DYNAMIC_TP_PROGRESS_RATIO of distance to TP AND short-EMA momentum is flattening
    After close: update logs/telegram and optionally reopen if trend still valid.
    """
    open_positions = mt5.positions_get(symbol=SYMBOL) or []
    if not open_positions:
        return

    # Precompute EMA slope for momentum check
    ema_fast_now = None
    ema_fast_prev = None
    try:
        if df_latest is not None and len(df_latest) > EMA_SLOW + EMA_SLOWING_LOOKBACK:
            ema_fast_now = float(df_latest.iloc[-1]['ema_fast'])
            ema_fast_prev = float(df_latest.iloc[-1 - EMA_SLOWING_LOOKBACK]['ema_fast'])
    except Exception:
        ema_fast_now = ema_fast_prev = None

    for pos in open_positions:
        try:
            profit = float(pos.profit)
            entry_price = float(pos.price_open)
            volume = float(pos.volume)
            tp = float(pos.tp) if pos.tp else None
            sl = float(pos.sl) if pos.sl else None
            # current market price
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                continue
            cur_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask

            # Absolute profit trigger
            if profit >= PROFIT_CLOSE_USD:
                should_close = True
                reason = "ABSOLUTE_PROFIT"
            else:
                should_close = False
                reason = ""

            # Dynamic progress to TP trigger + EMA slowdown
            if not should_close and tp is not None:
                total_target_dist = abs(tp - entry_price)
                progressed = abs(cur_price - entry_price)
                progress_ratio = (progressed / total_target_dist) if total_target_dist > 0 else 0.0
                ema_slowing = False
                if ema_fast_now is not None and ema_fast_prev is not None:
                    # momentum flattening if ema_fast now <= ema_fast_prev (or small drop)
                    ema_slowing = ema_fast_now <= ema_fast_prev
                # Decide dynamic close
                if progress_ratio >= DYNAMIC_TP_PROGRESS_RATIO and ema_slowing:
                    should_close = True
                    reason = f"DYN_TP_PROGRESS({progress_ratio:.2f})_EMA_SLOWING"

            if should_close:
                print(f"[monitor] Closing pos ticket={pos.ticket} profit={profit:.2f} reason={reason}")
                close_res = close_position(pos)
                try:
                    exit_price = cur_price
                except Exception:
                    exit_price = cur_price
                # update log and alerts
                try:
                    update_trade_close(pos.ticket, exit_price, profit, f"MONITOR_CLOSE_{reason}")
                except Exception:
                    pass
                send_close_alert_profit(pos.symbol, pos.ticket, profit, (datetime.now(timezone.utc) - pd.to_datetime(pos.time_msc, unit='ms').to_pydatetime()).total_seconds())

                # Optional: re-open if trend still valid
                try:
                    direction = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
                    # use df_latest for fresh indicators
                    if df_latest is not None and len(df_latest) >= 3:
                        ema_f = float(df_latest.iloc[-1]['ema_fast'])
                        ema_s = float(df_latest.iloc[-1]['ema_slow'])
                        st_dir = bool(df_latest.iloc[-1]['ST_dir'])
                        trend_good = (direction == "BUY" and ema_f > ema_s and st_dir) or (direction == "SELL" and ema_f < ema_s and not st_dir)
                    else:
                        trend_good = False

                    if trend_good:
                        # compute new sl/tp using atr_latest
                        atr_val = atr_latest if atr_latest is not None else (float(df_latest.iloc[-1]['ATR']) if (df_latest is not None and 'ATR' in df_latest.columns) else None)
                        tick2 = mt5.symbol_info_tick(SYMBOL)
                        if tick2 is None:
                            continue
                        exec_price = tick2.ask if direction == "BUY" else tick2.bid
                        if atr_val and atr_val > 0:
                            new_sl = exec_price - SL_ATR_MULTIPLIER * atr_val if direction == "BUY" else exec_price + SL_ATR_MULTIPLIER * atr_val
                            new_tp = exec_price + TP_ATR_MULTIPLIER * atr_val if direction == "BUY" else exec_price - TP_ATR_MULTIPLIER * atr_val
                        else:
                            buff = exec_price * 0.005
                            new_sl = exec_price - buff if direction == "BUY" else exec_price + buff
                            new_tp = exec_price + buff if direction == "BUY" else exec_price - buff

                        new_sl, new_tp = safe_stops(SYMBOL, exec_price, new_sl, new_tp)
                        # compute lot same as previous position volume (respect broker min)
                        lot = float(pos.volume)
                        # check allowed
                        new_trade_margin = estimate_margin_per_lot(SYMBOL, exec_price, lot)
                        point = mt5.symbol_info(SYMBOL).point
                        sl_points = abs(exec_price - new_sl) / point if point else 0
                        vpp = estimate_value_per_point(SYMBOL)
                        new_trade_risk_amount = lot * sl_points * vpp
                        allowed, reason_allow = allowed_to_open_new(SYMBOL, new_trade_margin, new_trade_risk_amount)
                        if allowed:
                            res = place_market_order(SYMBOL, direction, lot, exec_price, new_sl, new_tp, comment=COMMENT + "_REOPEN")
                            if res is not None and getattr(res, "retcode", None) in (10009, 10004, 0, None):
                                order_id = getattr(res, "order", None) or getattr(res, "deal", None) or 0
                                metrics = {"atr_value": atr_val if atr_val else 0, "ema_fast": ema_f, "ema_slow": ema_s, "supertrend_dir": "UP" if st_dir else "DOWN"}
                                log_trade_open("RiskStrategy", SYMBOL, direction, exec_price, new_sl, new_tp, metrics, order_id)
                                send_entry_alert("RiskStrategy", SYMBOL, direction, exec_price, new_sl, new_tp, metrics, order_id)
                                # Mirror reopen
                                mirror_order_to_second_account(direction, lot, new_sl, new_tp)
                                print(f"[monitor] Reopened {direction} ticket={order_id} vol={lot} price={exec_price}")
                            else:
                                print("[monitor] Reopen failed or retcode not OK:", res)
                        else:
                            print(f"[monitor] Reopen skipped due to risk/margin: {reason_allow}")
                except Exception as e:
                    print("[monitor] reopen exception", e)

        except Exception as e:
            print("[monitor] exception on pos", e)

# ---------------- MAIN LOOP (ENTRY + MANAGEMENT) ----------------
def main():
    if not initialize_mt5_with_path(MT5_MAIN_PATH):
        print("MT5 init failed - exit")
        return

    ensure_log_file()
    strategy = "RiskStrategy"
    print("Running", strategy, "on", SYMBOL, "(DEMO recommended)")

    last_monitor = datetime.now(timezone.utc) - timedelta(seconds=MONITOR_INTERVAL)

    while True:
        try:
            now = datetime.now(timezone.utc)
            df = fetch_rates(SYMBOL, TIMEFRAME, BARS)
            if df is None or len(df) < 50:
                print(f"[{now.strftime('%H:%M:%S')}] warn - No or insufficient data ({0 if df is None else len(df)}).")
                # still run monitor on positions even if no df? we will skip reopening checks requiring df though
                atr = None
            else:
                df = calc_ema(df)
                df = calc_supertrend(df)
                df = calc_atr(df, VOLATILITY_ATR_PERIOD)
                atr = float(df.iloc[-1]['ATR'])
                ema_fast = float(df.iloc[-1]['ema_fast'])
                ema_slow = float(df.iloc[-1]['ema_slow'])
                st_dir = bool(df.iloc[-1]['ST_dir'])
                price = float(df.iloc[-1]['close'])

            # periodic monitor-and-roll
            if (datetime.now(timezone.utc) - last_monitor).total_seconds() >= MONITOR_INTERVAL:
                try:
                    monitor_and_roll(df, atr)
                except Exception as e:
                    print("[monitor_and_roll] exception", e)
                last_monitor = datetime.now(timezone.utc)

            # Manage open positions first
            open_positions = mt5.positions_get(symbol=SYMBOL) or []
            for pos in open_positions:
                try:
                    tick = mt5.symbol_info_tick(pos.symbol)
                    if tick is None:
                        continue
                    cur_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
                    entry_price = float(pos.price_open)
                    sl = float(pos.sl) if pos.sl else None
                    tp = float(pos.tp) if pos.tp else None
                    profit = float(pos.profit)
                    volume = float(pos.volume)

                    atr_now = atr if atr is not None else 0.0
                    sl_distance = SL_ATR_MULTIPLIER * atr_now
                    tp_distance = TP_ATR_MULTIPLIER * atr_now

                    vpp = estimate_value_per_point(pos.symbol)
                    trail_activate_usd = TRAIL_ATR_MULTIPLIER_MOVE * atr_now * vpp * volume
                    breakeven_activate_usd = BREAKEVEN_ACTIVATE_ATR * atr_now * vpp * volume
                    partial_trigger_usd = PARTIAL_CLOSE_TRIGGER_ATR * atr_now * vpp * volume

                    # BUY management
                    if pos.type == mt5.POSITION_TYPE_BUY:
                        # TP hit
                        if atr_now > 0 and cur_price >= (entry_price + tp_distance):
                            res = close_position(pos)
                            update_trade_close(pos.ticket, cur_price, profit, "TP_hit")
                            send_close_alert_profit(pos.symbol, pos.ticket, profit, (datetime.now(timezone.utc) - pd.to_datetime(pos.time_msc, unit='ms').to_pydatetime()).total_seconds())
                            continue
                        # SL hit
                        if atr_now > 0 and cur_price <= (entry_price - max(sl_distance, MIN_SL_ATR_MULTIPLIER*atr_now)):
                            res = close_position(pos)
                            update_trade_close(pos.ticket, cur_price, profit, "SL_hit")
                            send_close_alert_loss(pos.symbol, pos.ticket, profit, (datetime.now(timezone.utc) - pd.to_datetime(pos.time_msc, unit='ms').to_pydatetime()).total_seconds())
                            continue
                        # Partial close
                        if PARTIAL_CLOSE_ENABLED and profit >= partial_trigger_usd and volume > (mt5.symbol_info(pos.symbol).volume_min * 1.5):
                            close_vol = max(pos.volume * PARTIAL_CLOSE_PCT, mt5.symbol_info(pos.symbol).volume_min)
                            close_vol = round(close_vol, 2)
                            try:
                                close_res = close_position(pos, close_volume=close_vol)
                                send_partial_close_alert(pos.symbol, pos.ticket, close_vol, profit)
                            except Exception as e:
                                print("[partial close] exception", e)
                        # Breakeven
                        if profit >= breakeven_activate_usd:
                            new_sl = max(sl if sl else -1e9, entry_price + (0.0001 if mt5.symbol_info(pos.symbol).point < 1 else mt5.symbol_info(pos.symbol).point))
                            if new_sl > (sl if sl else -1e9):
                                try:
                                    mt5.order_send({'action': mt5.TRADE_ACTION_SLTP, 'position': pos.ticket, 'sl': round(new_sl, mt5.symbol_info(pos.symbol).digits), 'tp': pos.tp, 'symbol': pos.symbol})
                                    send_telegram_html(f"🔒 Moved SL to breakeven for BUY {pos.ticket}")
                                except Exception:
                                    pass
                        # Trailing
                        if profit >= trail_activate_usd:
                            desired_sl = cur_price - (TRAIL_STEP_ATR * atr_now)
                            if sl is None or desired_sl > sl + mt5.symbol_info(pos.symbol).point/2:
                                try:
                                    mt5.order_send({'action': mt5.TRADE_ACTION_SLTP, 'position': pos.ticket, 'sl': round(desired_sl, mt5.symbol_info(pos.symbol).digits), 'tp': pos.tp, 'symbol': pos.symbol})
                                    print(f"[trail] BUY {pos.ticket} set SL to {desired_sl:.5f}")
                                except Exception:
                                    pass

                    else:
                        # SELL management
                        if atr_now > 0 and cur_price <= (entry_price - tp_distance):
                            res = close_position(pos)
                            update_trade_close(pos.ticket, cur_price, profit, "TP_hit")
                            send_close_alert_profit(pos.symbol, pos.ticket, profit, (datetime.now(timezone.utc) - pd.to_datetime(pos.time_msc, unit='ms').to_pydatetime()).total_seconds())
                            continue
                        if atr_now > 0 and cur_price >= (entry_price + max(sl_distance, MIN_SL_ATR_MULTIPLIER*atr_now)):
                            res = close_position(pos)
                            update_trade_close(pos.ticket, cur_price, profit, "SL_hit")
                            send_close_alert_loss(pos.symbol, pos.ticket, profit, (datetime.now(timezone.utc) - pd.to_datetime(pos.time_msc, unit='ms').to_pydatetime()).total_seconds())
                            continue
                        if PARTIAL_CLOSE_ENABLED and profit >= partial_trigger_usd and volume > (mt5.symbol_info(pos.symbol).volume_min * 1.5):
                            close_vol = max(pos.volume * PARTIAL_CLOSE_PCT, mt5.symbol_info(pos.symbol).volume_min)
                            close_vol = round(close_vol, 2)
                            try:
                                close_res = close_position(pos, close_volume=close_vol)
                                send_partial_close_alert(pos.symbol, pos.ticket, close_vol, profit)
                            except Exception as e:
                                print("[partial close] exception", e)
                        if profit >= breakeven_activate_usd:
                            new_sl = min(sl if sl else 1e9, entry_price - (0.0001 if mt5.symbol_info(pos.symbol).point < 1 else mt5.symbol_info(pos.symbol).point))
                            if new_sl < (sl if sl else 1e9):
                                try:
                                    mt5.order_send({'action': mt5.TRADE_ACTION_SLTP, 'position': pos.ticket, 'sl': round(new_sl, mt5.symbol_info(pos.symbol).digits), 'tp': pos.tp, 'symbol': pos.symbol})
                                    send_telegram_html(f"🔒 Moved SL to breakeven for SELL {pos.ticket}")
                                except Exception:
                                    pass
                        if profit >= trail_activate_usd:
                            desired_sl = cur_price + (TRAIL_STEP_ATR * atr_now)
                            if sl is None or desired_sl < sl - mt5.symbol_info(pos.symbol).point/2:
                                try:
                                    mt5.order_send({'action': mt5.TRADE_ACTION_SLTP, 'position': pos.ticket, 'sl': round(desired_sl, mt5.symbol_info(pos.symbol).digits), 'tp': pos.tp, 'symbol': pos.symbol})
                                    print(f"[trail] SELL {pos.ticket} set SL to {desired_sl:.5f}")
                                except Exception:
                                    pass
                except Exception as e:
                    print("[manage pos] exception", e)

            # Now process new signal: compute SL/TP & lot & risk/margin checks
            if df is not None and len(df) >= 50:
                last = df.iloc[-1]
                ema_fast = float(last['ema_fast'])
                ema_slow = float(last['ema_slow'])
                st_dir = bool(last['ST_dir'])
                price = float(last['close'])

                signal = None
                if ema_fast > ema_slow and st_dir:
                    signal = "BUY"
                elif ema_fast < ema_slow and not st_dir:
                    signal = "SELL"

                print(f"[signal] EMAf={ema_fast:.4f} EMAs={ema_slow:.4f} ST={st_dir} -> {signal}")

                if signal:
                    atr_for_entry = atr
                    sl_price = price - SL_ATR_MULTIPLIER * atr_for_entry if signal=="BUY" else price + SL_ATR_MULTIPLIER * atr_for_entry
                    tp_price = price + TP_ATR_MULTIPLIER * atr_for_entry if signal=="BUY" else price - TP_ATR_MULTIPLIER * atr_for_entry

                    # apply safe stops before sizing
                    sl_price, tp_price = safe_stops(SYMBOL, price, sl_price, tp_price)

                    lot = compute_lot_for_risk(SYMBOL, price, sl_price, RISK_PERCENT)
                    if lot <= 0:
                        print("[sizing] computed lot <=0, skipping")
                        time.sleep(SLEEP_SECONDS); continue

                    new_trade_margin = estimate_margin_per_lot(SYMBOL, price, lot)
                    point = mt5.symbol_info(SYMBOL).point
                    sl_points = abs(price - sl_price) / point if point else 0
                    vpp = estimate_value_per_point(SYMBOL)
                    new_trade_risk_amount = lot * sl_points * vpp

                    allowed, reason = allowed_to_open_new(SYMBOL, new_trade_margin, new_trade_risk_amount)
                    if not allowed:
                        print("[entry] skip: ", reason)
                        send_telegram_html(f"⚠️ NexusBot skipped entry {SYMBOL} ({signal}) - {reason}")
                        time.sleep(SLEEP_SECONDS); continue

                    tick = mt5.symbol_info_tick(SYMBOL)
                    if tick is None:
                        print("[order] no tick available, skipping")
                        time.sleep(SLEEP_SECONDS); continue
                    exec_price = tick.ask if signal=="BUY" else tick.bid

                    res = place_market_order(SYMBOL, signal, lot, exec_price, sl_price, tp_price, comment=COMMENT)
                    if res is None:
                        print("[order] order_send returned None")
                        time.sleep(SLEEP_SECONDS); continue
                    # handle invalid stops retcode (10016) by adjusting or retrying
                    retcode = getattr(res, "retcode", None)
                    if retcode == 10016:
                        print("[order] 10016 invalid stops, trying safe_stops fallback")
                        # try adjust stops according to safe_stops
                        sl_price2, tp_price2 = safe_stops(SYMBOL, exec_price, sl_price, tp_price)
                        res2 = place_market_order(SYMBOL, signal, lot, exec_price, sl_price2, tp_price2, comment=COMMENT + "_FALLBACK")
                        if res2 is not None and getattr(res2, "retcode", None) in (10009, 10004, 0, None):
                            res = res2
                            print("[order] fallback with safe stops succeeded")
                        else:
                            print("[order] fallback failed:", res2)
                    if getattr(res, "retcode", None) in (10009, 10004, 0, None):
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

                        # Mirror the same order to second account
                        mirror_order_to_second_account(signal, lot, sl_price, tp_price)
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
