import os
os.environ ['GIT_PYTHON_GIT_EXECUTABLE'] = r'C:\Program Files\Git\cmd\git.exe'
import re
import io
import base64
import time
import shutil
from datetime import datetime, timedelta, timezone, UTC
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

# Optional Git + Telegram
try:
    import git  # GitPython
    HAVE_GIT = True
except Exception:
    HAVE_GIT = False

try:
    import requests
    HAVE_REQ = True
except Exception:
    HAVE_REQ = False

try:
    import MetaTrader5 as mt5
    HAVE_MT5 = True
except Exception:
    HAVE_MT5 = False

# ==========================
# Config (env-first)
# ==========================
LOG_FILE = os.getenv("LOG_FILE", r"C:\\Users\\krish\\Documents\\XAUUSD\\venv\\Scripts\\goldflash_nexus_log.txt")
REPORT_SYMBOLS = os.getenv("REPORT_SYMBOLS", "XAUUSD.s,XAGUSD.s,NAS100.s").split(",")
MT5_HISTORY_DAYS = int(os.getenv("MT5_HISTORY_DAYS", "14"))

# Save report exactly here:
REPORT_HTML = r"C:\Users\krish\Documents\XAUUSD\NexusBot_Monitoring\NexusBot_Monitoring2\Reports\report.html"

GITHUB_REPO_PATH = r"C:\Users\krish\Documents\XAUUSD\NexusBot_Monitoring\NexusBot_Monitoring2"

# This must match your GitHub Pages URL exactly for Telegram message:
GITHUB_PAGES_URL = "https://krishnagenaiexp11-nexusbot.github.io/NexusBot_Monitoring_2/Reports/report.html"

# Put your token and chat ID here or via environment variables for security
TELEGRAM_TOKEN = "8299338855:AAGPO7keJwwIkglNghehqkSgTvIyhpa3fQg"
TELEGRAM_CHAT_ID = "-4912158984"

INTERVAL_SECONDS = 900  # 15 min default

# Use fixed IST tz to align log + MT5
IST = timezone(timedelta(hours=5, minutes=30))

# Map of magic->strategy, e.g. "1001=goldflash,1002=silverwave"
MAGIC_STRATEGIES_STR = os.getenv("MAGIC_STRATEGIES", "")

def _parse_magic_map(s: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for part in [p.strip() for p in s.split(",") if p.strip()]:
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[int(k.strip())] = v.strip().lower()
            except Exception:
                pass
    return out

MAGIC_STRATEGIES = _parse_magic_map(MAGIC_STRATEGIES_STR)

# If using Git for Windows, hint GitPython
if os.name == "nt" and not os.environ.get("GIT_PYTHON_GIT_EXECUTABLE"):
    guess = r"C:\\Program Files\\Git\\cmd\\git.exe"
    if os.path.exists(guess):
        os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = guess

# ==========================
# Log parsing
# ==========================
ORDER_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s*\|\s*.*?Order placed\s*#(?P<order>\d+)\s*\|\s*(?P<symbol>[\w\.]+)\s*\|\s*(?P<strategy>[\w\-]+)\s*\|\s*(?P<side>BUY|SELL)", re.IGNORECASE)
PNL_RE   = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s*\|\s*.*?(Profit booked|Loss cut)\s*#(?P<order>\d+)\s*\((?P<pnl>[-\d\.]+)\)\s*(?:\|\s*(?P<symbol>[\w\.]+))?", re.IGNORECASE)
TRAIL_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s*:\s*Trailing update\s*#(?P<order>\d+)", re.IGNORECASE)

ERROR_KEYS = ("[main error]", "MetaTrader5", "volume")

def parse_log_file(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, int, List[str]]:
    """Return (orders_log, closures_log, trailing_updates_count, errors[])"""
    orders: List[dict] = []
    closures: List[dict] = []
    trail_updates = 0
    errors: List[str] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_stripped = line.strip()
            m = ORDER_RE.search(line_stripped)
            if m:
                gd = m.groupdict()
                orders.append({
                    "order_id": int(gd["order"]),
                    "symbol": gd["symbol"],
                    "strategy": gd["strategy"].lower(),
                    "side": gd["side"].upper(),
                    "ts": datetime.fromisoformat(gd["ts"]).replace(tzinfo=IST),
                })
                continue
            m = PNL_RE.search(line_stripped)
            if m:
                gd = m.groupdict()
                closures.append({
                    "order_id": int(gd["order"]),
                    "symbol": gd.get("symbol") or None,
                    "pnl": float(gd["pnl"]),
                    "ts": datetime.fromisoformat(gd["ts"]).replace(tzinfo=IST),
                })
                continue
            m = TRAIL_RE.search(line_stripped)
            if m:
                trail_updates += 1
            if any(k.lower() in line_stripped.lower() for k in ERROR_KEYS):
                errors.append(line_stripped)

    orders_log = pd.DataFrame(orders)
    closures_log = pd.DataFrame(closures)
    return orders_log, closures_log, trail_updates, errors

# ==========================
# MT5 fetch
# ==========================

def init_mt5() -> bool:
    if not HAVE_MT5:
        print("MetaTrader5 module not available.")
        return False
    if not mt5.initialize():
        print("MT5 initialize() failed.")
        return False
    for sym in REPORT_SYMBOLS:
        try:
            mt5.symbol_select(sym, True)
        except Exception:
            pass
    return True

def fetch_mt5_orders_deals(days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not init_mt5():
        return pd.DataFrame(), pd.DataFrame()
    utc_to = datetime.now(UTC)
    utc_from = utc_to - timedelta(days=days)

    orders_raw = mt5.history_orders_get(utc_from, utc_to) or []
    deals_raw  = mt5.history_deals_get(utc_from, utc_to) or []

    orders: List[dict] = []
    for o in orders_raw:
        if o.symbol not in REPORT_SYMBOLS:
            continue
        orders.append({
            "order_id": int(o.ticket),
            "symbol": o.symbol,
            "time_setup": datetime.fromtimestamp(o.time_setup, tz=timezone.utc).astimezone(IST),
            "magic": int(getattr(o, "magic", 0)) if getattr(o, "magic", None) is not None else None,
            "comment": getattr(o, "comment", "") or "",
        })

    deals: List[dict] = []
    for d in deals_raw:
        if d.symbol not in REPORT_SYMBOLS:
            continue
        if getattr(d, "entry", None) != mt5.DEAL_ENTRY_OUT:
            continue
        if getattr(d, "type", None) in {
            getattr(mt5, "DEAL_TYPE_COMMISSION", -1),
            getattr(mt5, "DEAL_TYPE_SWAP", -2),
            getattr(mt5, "DEAL_TYPE_BALANCE", -3),
            getattr(mt5, "DEAL_TYPE_TAX", -4),
            getattr(mt5, "DEAL_TYPE_CHARGE", -5),
            getattr(mt5, "DEAL_TYPE_CREDIT", -6),
        }:
            continue
        deals.append({
            "deal_id": int(d.ticket),
            "order_id": int(d.order),
            "symbol": d.symbol,
            "time": datetime.fromtimestamp(d.time, tz=timezone.utc).astimezone(IST),
            "profit": float(getattr(d, "profit", 0.0)),
            "comment": getattr(d, "comment", "") or "",
            "magic": int(getattr(d, "magic", 0)) if getattr(d, "magic", None) is not None else None,
            "price": float(getattr(d, "price", float("nan"))),
            "volume": float(getattr(d, "volume", float("nan"))),
        })

    try:
        mt5.shutdown()
    except Exception:
        pass

    return pd.DataFrame(orders), pd.DataFrame(deals)

# ==========================
# Strategy helpers
# ==========================

def _extract_strategy_from_comment(comment: str) -> Optional[str]:
    if not isinstance(comment, str) or not comment:
        return None
    m = re.search(r"strategy\s*[:=]\s*([A-Za-z0-9_\-]+)", comment, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r"\[([A-Za-z0-9_\-]{3,})\]", comment)
    if m:
        return m.group(1).lower()
    return None

def _strategy_from_magic(magic: Optional[int]) -> Optional[str]:
    if magic is None:
        return None
    return MAGIC_STRATEGIES.get(int(magic))

def _bot_manual_from_comment(comment: str) -> Tuple[bool, bool]:
    c = (comment or "").lower()
    is_bot = any(k in c for k in ("tp", "sl", "autoclose", "algo", "ea"))
    is_manual = ("close" in c or "manual" in c) and not is_bot
    return is_bot, is_manual

# ==========================
# Reconciliation & analytics
# ==========================

def reconcile_and_enrich(orders_log: pd.DataFrame,
                         closures_log: pd.DataFrame,
                         mt5_orders: pd.DataFrame,
                         mt5_deals: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    odf = mt5_orders.copy()
    ddf = mt5_deals.copy()

    if not orders_log.empty and "order_id" in orders_log.columns and "strategy" in orders_log.columns:
        ol = orders_log[["order_id", "strategy"]].drop_duplicates("order_id").rename(columns={"strategy": "strategy_from_log"})
        odf = odf.merge(ol, on="order_id", how="left")
    else:
        odf["strategy_from_log"] = None

    odf["strategy_from_comment"] = odf["comment"].map(_extract_strategy_from_comment)
    odf["strategy_from_magic"] = odf["magic"].map(_strategy_from_magic)
    odf["strategy"] = (
        odf["strategy_from_log"]
        .combine_first(odf["strategy_from_comment"])
        .combine_first(odf["strategy_from_magic"])
        .fillna("unknown")
    )
    odf.drop(columns=[c for c in ["strategy_from_log", "strategy_from_comment", "strategy_from_magic"] if c in odf.columns], inplace=True)

    ddf = ddf.merge(odf[["order_id", "strategy", "symbol"]].rename(columns={"symbol": "symbol_from_order"}), on="order_id", how="left")
    ddf["strategy_from_comment"] = ddf["comment"].map(_extract_strategy_from_comment)
    ddf["strategy_from_magic"] = ddf["magic"].map(_strategy_from_magic)
    ddf["strategy"] = (
        ddf["strategy"]
        .combine_first(ddf["strategy_from_comment"])
        .combine_first(ddf["strategy_from_magic"])
        .fillna("unknown")
    )
    ddf.drop(columns=[c for c in ["strategy_from_comment", "strategy_from_magic"] if c in ddf.columns], inplace=True)

    bm = ddf["comment"].map(_bot_manual_from_comment)
    ddf["is_bot"] = bm.map(lambda t: t[0])
    ddf["is_manual"] = bm.map(lambda t: t[1])

    pnl_log = closures_log.copy() if not closures_log.empty else pd.DataFrame(columns=["order_id","symbol","pnl","ts"])
    if not pnl_log.empty and pd.api.types.is_datetime64_any_dtype(pnl_log["ts"]):
        if pnl_log["ts"].dt.tz is None:
            pnl_log["ts"] = pnl_log["ts"].dt.tz_localize(IST)

    if ddf.empty:
        missing_ids = set(pnl_log["order_id"]) if not pnl_log.empty else set()
    else:
        missing_ids = set(pnl_log["order_id"]) - set(ddf["order_id"]) if not pnl_log.empty else set()

    fallback_rows: List[dict] = []
    if missing_ids:
        for oid in missing_ids:
            _rows = pnl_log[pnl_log["order_id"] == oid]
            if _rows.empty:
                continue
            sym = None
            if _rows["symbol"].notna().any():
                sym = _rows["symbol"].dropna().iloc[0]
            elif not odf.empty and (odf["order_id"] == oid).any():
                if "symbol_from_order" in odf.columns and odf.loc[odf["order_id"] == oid, "symbol_from_order"].notna().any():
                    sym = odf.loc[odf["order_id"] == oid, "symbol_from_order"].dropna().iloc[0]
                elif "symbol" in odf.columns and odf.loc[odf["order_id"] == oid, "symbol"].notna().any():
                    sym = odf.loc[odf["order_id"] == oid, "symbol"].dropna().iloc[0]

            ts = _rows.sort_values("ts")["ts"].iloc[-1]
            pnl = float(_rows["pnl"].sum())

            strat = None
            if not odf.empty and (odf["order_id"] == oid).any():
                if odf.loc[odf["order_id"] == oid, "strategy"].notna().any():
                    strat = odf.loc[odf["order_id"] == oid, "strategy"].dropna().iloc[0]
            elif not orders_log.empty and oid in orders_log["order_id"].values:
                if orders_log.loc[orders_log["order_id"] == oid, "strategy"].notna().any():
                    strat = orders_log.loc[orders_log["order_id"] == oid, "strategy"].dropna().iloc[0]

            fallback_rows.append({
                "deal_id": int(9_000_000_000 + oid),
                "order_id": int(oid),
                "symbol": sym or "",
                "time": ts,
                "profit": pnl,
                "comment": "log_fallback",
                "magic": None,
                "price": float("nan"),
                "volume": float("nan"),
                "strategy": (strat or "unknown"),
                "is_bot": False,
                "is_manual": False,
            })
    if fallback_rows:
        ddf = pd.concat([ddf, pd.DataFrame(fallback_rows)], ignore_index=True)

    if "symbol" not in ddf.columns:
        ddf["symbol"] = None
    ddf["symbol"] = ddf["symbol"].fillna(ddf.get("symbol_from_order"))

    if "profit" in ddf.columns:
        ddf["profit"] = ddf["profit"].astype(float)
    pnl_per_order = ddf.groupby(["order_id", "strategy"], dropna=False)["profit"].sum().reset_index()

    return odf, ddf, pnl_per_order

def _safe_profit_factor(series: pd.Series) -> float:
    pos_sum = series[series > 0].sum()
    neg_sum = series[series <= 0].sum()
    return float(pos_sum / abs(neg_sum)) if neg_sum < 0 else float("inf")

def summarize_strategy(orders_df: pd.DataFrame, pnl_per_order: pd.DataFrame) -> pd.DataFrame:
    orders_counts = orders_df.groupby("strategy").size().rename("Orders Placed").reset_index()

    comp = (pnl_per_order.groupby("strategy")["profit"].agg(
        **{
            "Completed Trades": "count",
            "Profitable Trades": lambda s: int((s > 0).sum()),
            "Losing Trades": lambda s: int((s <= 0).sum()),
            "Total PnL": "sum",
            "Avg PnL": "mean",
            "Avg Win": lambda s: s[s > 0].mean() if (s > 0).any() else 0.0,
            "Best Win": lambda s: s[s > 0].max() if (s > 0).any() else 0.0,
            "Avg Loss": lambda s: s[s <= 0].mean() if (s <= 0).any() else 0.0,
            "Worst Loss": lambda s: s[s <= 0].min() if (s <= 0).any() else 0.0,
            "Profit Factor": lambda s: _safe_profit_factor(s),
        }
    ).reset_index())

    out = orders_counts.merge(comp, on="strategy", how="outer").fillna(0)
    out["Hit Rate (%)"] = out.apply(lambda r: (r["Profitable Trades"]/r["Completed Trades"]*100.0) if r["Completed Trades"] else 0.0, axis=1)
    cols = [
        "strategy", "Orders Placed", "Completed Trades", "Profitable Trades", "Losing Trades", "Hit Rate (%)",
        "Total PnL", "Avg PnL", "Avg Win", "Best Win", "Avg Loss", "Worst Loss", "Profit Factor"
    ]
    out = out.reindex(columns=cols)
    out = out.sort_values(["Total PnL", "Hit Rate (%)"], ascending=[False, False])
    return out

def summarize_symbol(deals_df: pd.DataFrame) -> pd.DataFrame:
    if deals_df.empty:
        return pd.DataFrame(columns=["Symbol","Trades","Total PnL","Avg PnL","Min PnL","Max PnL"])
    sym = (deals_df.groupby("symbol")["profit"].agg(Trades="count", **{"Total PnL":"sum", "Avg PnL":"mean", "Min PnL":"min", "Max PnL":"max"})
            .sort_values("Total PnL", ascending=False).reset_index().rename(columns={"symbol":"Symbol"}))
    return sym

def combined_strat_symbol(deals_df: pd.DataFrame) -> pd.DataFrame:
    if deals_df.empty:
        return pd.DataFrame(columns=["Strategy","Symbol","Trades","Net PnL","Verdict"])
    g = deals_df.groupby(["strategy","symbol"])['profit'].agg(Trades='count', **{"Net PnL":'sum'}).reset_index()
    g["Verdict"] = np.where(g["Net PnL"]>0, "▲ Profitable", np.where(g["Net PnL"]<0, "▼ Loss", "▼ Marginal"))
    g = g.rename(columns={"strategy":"Strategy","symbol":"Symbol"}).sort_values(["Strategy","Symbol"]).reset_index(drop=True)
    return g

def manual_vs_bot(deals_df: pd.DataFrame) -> pd.DataFrame:
    if deals_df.empty or ("is_bot" not in deals_df.columns):
        return pd.DataFrame(columns=["Strategy","Symbol","Total Trades","Bot Closed","Manual Closed","Bot %","Manual %"])
    g = deals_df.groupby(["strategy","symbol"]).agg(
        **{"Total Trades": ("profit","count"), "Bot Closed": ("is_bot","sum"), "Manual Closed": ("is_manual","sum")}
    ).reset_index()
    g["Bot %"] = (g["Bot Closed"]/g["Total Trades"]*100).round(1)
    g["Manual %"] = (g["Manual Closed"]/g["Total Trades"]*100).round(1)
    return g.rename(columns={"strategy":"Strategy","symbol":"Symbol"}).sort_values(["Strategy","Symbol"])

def strategies_over_time(deals_df: pd.DataFrame) -> pd.DataFrame:
    if deals_df.empty:
        return pd.DataFrame(columns=["strategy","date","cum_pnl"])
    tmp = deals_df.copy()
    if tmp["time"].dt.tz is None:
        tmp["time"] = tmp["time"].dt.tz_localize(IST)
    tmp['date'] = tmp['time'].dt.tz_convert(IST).dt.floor('D')
    daily = tmp.groupby(['strategy','date'])['profit'].sum()
    cum = daily.groupby(level=0).cumsum().reset_index(name='cum_pnl')
    return cum

def drawdown_stats(deals_df: pd.DataFrame) -> Dict[str, float]:
    if deals_df.empty:
        return {"max_drawdown": 0.0, "largest_loss": 0.0, "largest_consec_losses": 0}
    eq = deals_df.sort_values('time').assign(eq=lambda d: d['profit'].cumsum())
    roll_max = eq['eq'].cummax()
    max_dd = float((roll_max - eq['eq']).max())
    largest_loss = float(deals_df['profit'].min())
    streak = 0
    max_streak = 0
    for v in deals_df.sort_values('time')['profit']:
        if v < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return {"max_drawdown": round(max_dd, 2), "largest_loss": round(largest_loss, 2), "largest_consec_losses": int(max_streak)}

def _fmt_num(x):
    if isinstance(x, (float, np.floating)):
        if np.isinf(x):
            return "∞"
        return f"{x:,.2f}"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return x

def _b64_small(fig):
    if not HAVE_PLT:
        return ""
    buf = io.BytesIO()
    fig.tight_layout(pad=1.2)
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def chart_cum_pnl(cum: pd.DataFrame) -> str:
    if not HAVE_PLT or cum.empty:
        return ""
    fig = plt.figure(figsize=(7,3.4))
    for strat, g in cum.groupby('strategy'):
        plt.plot(g['date'], g['cum_pnl'], label=str(strat).upper())
    plt.title("Cumulative PnL by Strategy", fontsize=11)
    plt.xlabel("Date", fontsize=9); plt.ylabel("PnL", fontsize=9)
    plt.legend(fontsize=8); plt.grid(False)
    return _b64_small(fig)

def chart_daily_pnl(deals_df: pd.DataFrame) -> str:
    if not HAVE_PLT or deals_df.empty:
        return ""
    tmp = deals_df.copy()
    if tmp["time"].dt.tz is None:
        tmp["time"] = tmp["time"].dt.tz_localize(IST)
    tmp['date'] = tmp['time'].dt.tz_convert(IST).dt.floor('D')
    daily = tmp.groupby('date')['profit'].sum().reset_index()
    fig = plt.figure(figsize=(7,3.4))
    plt.bar(daily['date'], daily['profit'])
    plt.title("Daily Net PnL", fontsize=11)
    plt.xlabel("Date", fontsize=9); plt.ylabel("PnL", fontsize=9)
    return _b64_small(fig)

def chart_bar(df: pd.DataFrame, x: str, y: str, title: str) -> str:
    if not HAVE_PLT or df.empty:
        return ""
    fig = plt.figure(figsize=(5.6,3.2))
    plt.bar(df[x], df[y])
    plt.title(title, fontsize=11)
    plt.xlabel("", fontsize=9); plt.ylabel("PnL", fontsize=9)
    return _b64_small(fig)

def html_table(df: pd.DataFrame, title: str, subtitle: str = "") -> str:
    if df is None or df.empty:
        table_html = "<p><i>No data.</i></p>"
    else:
        d = df.copy()
        d.columns = [str(c) for c in d.columns]
        num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
        for c in d.columns:
            if pd.api.types.is_numeric_dtype(d[c]):
                d[c] = d[c].map(_fmt_num)
        head = "".join(f"<th>{c}</th>" for c in d.columns)
        rows = []
        for _, row in d.iterrows():
            row_html = ""
            for c in d.columns:
                cell = row[c]
                cell_class = "num" if c in num_cols else ""
                row_html += f"<td class='{cell_class}'>{cell}</td>"
            rows.append(f"<tr>{row_html}</tr>")
        table_html = f"<div style='overflow-x:auto;'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"

    card = f"""
    <div class='card'>
      <div class='card-head'><div><div class='title'>{title}</div><div class='sub'>{subtitle}</div></div></div>
      <div class='card-body'>{table_html}</div>
    </div>
    """
    return card

def build_html(strategy_df: pd.DataFrame,
               symbol_df: pd.DataFrame,
               comb_df: pd.DataFrame,
               mvb_df: pd.DataFrame,
               deals_df: pd.DataFrame,
               cum_df: pd.DataFrame,
               dd: Dict[str, float]) -> str:
    now_str = datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S")
    total_pnl = float(strategy_df["Total PnL"].sum()) if not strategy_df.empty else 0.0
    overall_hit = (float(strategy_df["Profitable Trades"].sum()) / max(1.0, float(strategy_df["Completed Trades"].sum())) * 100.0) if not strategy_df.empty else 0.0
    total_trades = int(symbol_df["Trades"].sum()) if not symbol_df.empty else 0

    style = """
<link href="https://fonts.googleapis.com/css?family=Roboto+Mono&display=swap" rel="stylesheet">
<style>
:root { --bg:#ffffff; --text:#1f2937; --muted:#6b7280; --line:#e5e7eb; }
body { margin:0; background:var(--bg); color:var(--text); font:13px/1.45 -apple-system,Segoe UI,Roboto,Arial,sans-serif; }
.container { max-width:1000px; margin:28px auto 48px; padding:0 16px; }
h1 { font-size:20px; font-weight:600; margin-bottom:8px; }
.header { display:flex; align-items:baseline; justify-content:space-between; gap:8px; }
.badge { font-size:11px; color:var(--muted); }
.kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin:12px 0; }
.kpi { border:1px solid var(--line); border-radius:8px; padding:10px 12px; background:#fff; }
.kpi .label { font-size:11px; color:var(--muted); }
.kpi .value { font-size:16px; font-weight:600; margin-top:4px; }
.grid { display:grid; gap:12px; margin-bottom:18px; }
.two { grid-template-columns:1fr; }
@media (min-width:900px) { .two { grid-template-columns:1fr 1fr; } }
.card { border:1px solid var(--line); border-radius:8px; background:#fff; margin-bottom:16px; overflow-x:auto; }
.card-head { padding:10px 12px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; gap:8px; }
.card-head .title { font-size:13px;font-weight:600; }
.card-head .sub { font-size:11px; color:var(--muted);}
.card-body { padding:12px 12px; overflow-x:auto; }
.figure { max-width:100%; height:auto; border:1px solid var(--line); border-radius:6px; display:block; margin:0 auto 10px; }
.small { color:var(--muted); font-size:11px; }

/* Table improvements */
table { width:100%; border-collapse:collapse; }
th, td {
  min-width: 90px;
  max-width: 180px;
  padding: 8px;
  border-bottom: 1px solid var(--line);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 13px;
}
th { background:#f9fafb; position:sticky; top:0; white-space:nowrap; text-align:center; }
td { text-align:left; }
td.num {
  text-align:right;
  font-family: 'Roboto Mono', 'Consolas', monospace;
  font-variant-numeric: tabular-nums;
}

/* Mobile responsiveness */
@media (max-width:700px) {
  .container { padding:0 4px; }
  th, td { font-size:11px; padding:4px; }
  h1 { font-size:16px; }
}
</style>

    """

    cum_b64 = chart_cum_pnl(cum_df)
    daily_b64 = chart_daily_pnl(deals_df)
    by_strat = (deals_df.groupby('strategy')['profit'].sum().reset_index().sort_values('profit', ascending=False)
                if not deals_df.empty else pd.DataFrame(columns=['strategy','profit']))
    by_strat['strategy'] = by_strat['strategy'].astype(str).str.upper()
    strat_b64 = chart_bar(by_strat, 'strategy', 'profit', 'PnL by Strategy')
    by_sym = (deals_df.groupby('symbol')['profit'].sum().reset_index().sort_values('profit', ascending=False)
              if not deals_df.empty else pd.DataFrame(columns=['symbol','profit']))
    sym_b64 = chart_bar(by_sym, 'symbol', 'profit', 'PnL by Symbol')

    html = [f"<html><head><meta charset='utf-8'><title>Algo Trading Report</title>{style}</head><body>"]
    html.append("<div class='container'>")

    html.append("<div class='header'>")
    html.append(f"<h1>Algo Trading Report</h1><span class='badge'>Last updated — {now_str}</span>")
    html.append("</div>")

    html.append("<div class='kpis'>")
    html.append(f"<div class='kpi'><div class='label'>Total PnL</div><div class='value'>{_fmt_num(total_pnl)}</div></div>")
    html.append(f"<div class='kpi'><div class='label'>Hit Rate</div><div class='value'>{overall_hit:.1f}%</div></div>")
    html.append(f"<div class='kpi'><div class='label'>Total Trades</div><div class='value'>{_fmt_num(total_trades)}</div></div>")
    html.append(f"<div class='kpi'><div class='label'>Max Drawdown</div><div class='value'>{_fmt_num(dd['max_drawdown'])}</div></div>")
    html.append("</div>")

    html.append("<div class='grid two'>")
    html.append("<div class='card'><div class='card-head'><div><div class='title'>Cumulative PnL</div><div class='sub'>Per strategy, last N days</div></div></div><div class='card-body'>")
    html.append(f"<img class='figure' src='data:image/png;base64,{cum_b64}' />" if cum_b64 else "<p class='small'>No chart data.</p>")
    html.append("</div></div>")

    html.append("<div class='card'><div class='card-head'><div><div class='title'>Daily Net PnL</div><div class='sub'>All strategies</div></div></div><div class='card-body'>")
    html.append(f"<img class='figure' src='data:image/png;base64,{daily_b64}' />" if daily_b64 else "<p class='small'>No chart data.</p>")
    html.append("</div></div>")
    html.append("</div>")

    html.append("<div class='grid two'>")
    html.append("<div class='card'><div class='card-head'><div><div class='title'>PnL by Strategy</div><div class='sub'>Total over the period</div></div></div><div class='card-body'>")
    html.append(f"<img class='figure' src='data:image/png;base64,{strat_b64}' />" if strat_b64 else "<p class='small'>No chart data.</p>")
    html.append("</div></div>")

    html.append("<div class='card'><div class='card-head'><div><div class='title'>PnL by Symbol</div><div class='sub'>Total over the period</div></div></div><div class='card-body'>")
    html.append(f"<img class='figure' src='data:image/png;base64,{sym_b64}' />" if sym_b64 else "<p class='small'>No chart data.</p>")
    html.append("</div></div>")
    html.append("</div>")

    html.append(html_table(strategy_df.rename(columns={"strategy":"Strategy"}),
                           title="Table A — Strategy Performance",
                           subtitle="Log as primary; MT5 reconciles PnL & fills"))
    html.append(html_table(symbol_df, title="Table B — Symbol Performance", subtitle="MT5 deals, exits only"))
    html.append(html_table(comb_df, title="Table C — Strategy × Symbol", subtitle="Trades, net PnL, verdict"))
    html.append(html_table(mvb_df, title="Table D — Manual vs Bot-Closed", subtitle="Heuristic from comments"))

    html.append("<p class='small' style='margin-top:10px;'>Strategy names from log (preferred), then comment/magic if missing. Times are IST.</p>")
    html.append("</div></body></html>")
    return "".join(html)

def push_reports_to_github():
    try:
        repo = git.Repo(GITHUB_REPO_PATH)
        repo_root = os.path.abspath(repo.working_tree_dir)

        report_source = REPORT_HTML
        report_target = os.path.join(repo_root, "Reports", "report.html")

        # Copy report if not in correct place
        if os.path.abspath(report_source) != os.path.abspath(report_target):
            os.makedirs(os.path.dirname(report_target), exist_ok=True)
            shutil.copy2(report_source, report_target)
            print(f"Copied report to repo path: {report_target}")

        repo.git.add("Reports/report.html")

        if not repo.is_dirty(index=True, working_tree=True, untracked_files=True):
            print("No changes to commit; skipping push.")
            return

        repo.index.commit(f"Update report {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        origin = repo.remote(name="origin")
        origin.push(refspec="main:main")
        print("Pushed report to GitHub.")
    except Exception as e:
        print(f"Git push failed: {e}")

def send_telegram(strategy_df: pd.DataFrame):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID or strategy_df.empty or not HAVE_REQ:
        print("Telegram skipped: missing token/chat_id or no data.")
        return

    top = strategy_df.iloc[0]
    msg = (
        "📈 *Algo Trading Report Updated*\n\n"
        f"*Top Strategy:* `{str(top['strategy']).upper()}`\n"
        f"*Total PnL:* `{_fmt_num(float(top['Total PnL']))}`\n"
        f"*Hit Rate:* `{float(top['Hit Rate (%)']):.1f}%`\n\n"
        f"[View Report]({GITHUB_PAGES_URL})"
    )
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        print("Telegram sent.")
    except Exception as e:
        print(f"Telegram failed: {e}")

def generate_once() -> str:
    print(f"Parsing log: {LOG_FILE}")
    orders_log, closures_log, trail_cnt, err_lines = parse_log_file(LOG_FILE)
    print(f"Log: {len(orders_log)} orders, {len(closures_log)} closures; trailing updates: {trail_cnt}")

    print("Fetching MT5 history…")
    mt5_orders, mt5_deals = fetch_mt5_orders_deals(MT5_HISTORY_DAYS)
    print(f"MT5: {len(mt5_orders)} orders, {len(mt5_deals)} deals")

    orders_all, deals_all, pnl_per_order = reconcile_and_enrich(orders_log, closures_log, mt5_orders, mt5_deals)
    strat_df = summarize_strategy(orders_all, pnl_per_order)
    sym_df   = summarize_symbol(deals_all)
    comb_df  = combined_strat_symbol(deals_all)
    mvb_df   = manual_vs_bot(deals_all)
    cum_df   = strategies_over_time(deals_all)
    dd       = drawdown_stats(deals_all)

    html = build_html(strat_df, sym_df, comb_df, mvb_df, deals_all, cum_df, dd)

    # Save report HTML to the designated path
    os.makedirs(os.path.dirname(REPORT_HTML), exist_ok=True)
    with open(REPORT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Report written: {REPORT_HTML}")

    # Push report to GitHub and send Telegram notification
    push_reports_to_github()
    send_telegram(strat_df)

    return REPORT_HTML

def main_loop():
    while True:
        try:
            generate_once()
        except Exception as e:
            print(f"Error: {e}")
        print(f"Sleeping for {INTERVAL_SECONDS//60} minutes…")
        time.sleep(INTERVAL_SECONDS)

if __name__ == '__main__':
   # generate_once()
    # To run continually, uncomment the next line to loop every INTERVAL_SECONDS
    main_loop()
