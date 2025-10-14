# report_generator.py
# Rebuilds the requested report by analyzing:
# - MT5 history: ReportHistory-548269.xlsx
# - MT5 orders: ReportHistory_orders-548269.xlsx
# - Bot log: goldflash_nexus_log.txt
# Outputs CSV summaries and optional charts.

import re
import sys
import math
import json
import argparse
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np

# Optional: charts
try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

def load_positions_xlsx(path: Path) -> pd.DataFrame:
    x = pd.read_excel(path, sheet_name=0, header=None)
    # Find the row with 'Positions' and use the next row as header
    pos_rows = x.index[x.apply(lambda r: r.astype(str).str.contains('Positions', case=False, na=False).any(), axis=1)]
    if len(pos_rows):
        header_row = pos_rows[0] + 1
    else:
        # Fallback: first row where first cell == 'Time'
        hdr = x.index[x.iloc[:, 0].astype(str).str.strip().eq('Time')]
        if len(hdr) == 0:
            raise RuntimeError(f"Could not locate Positions header in {path}")
        header_row = hdr[0]
    header = [str(h).strip() for h in x.iloc[header_row].tolist()]
    df = x.iloc[header_row + 1:].copy()
    df.columns = header

    # Drop obvious non-data rows
    if 'Symbol' in df.columns:
        df = df[~(df['Symbol'].isna() & df.get('Profit', pd.Series([np.nan]*len(df))).isna())]

    # Ensure unique columns
    seen = {}
    new_cols = []
    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
    df.columns = new_cols

    # Normalize types
    for c in df.columns:
        if c.startswith('Time'):
            df[c] = pd.to_datetime(df[c], errors='coerce')
    for num in ['Volume', 'Price', 'Commission', 'Swap', 'Profit']:
        if num in df.columns:
            df[num] = pd.to_numeric(df[num], errors='coerce')
    if 'Symbol' in df.columns:
        df['Symbol'] = df['Symbol'].astype(str).str.strip()
    if 'Type' in df.columns:
        df['Type'] = df['Type'].astype(str).str.strip()

    # Derive CloseTime
    if 'Time_1' in df.columns:
        df['CloseTime'] = df['Time_1']
    else:
        # last time-like column
        time_like = [c for c in df.columns if c.startswith('Time') and c != 'Time']
        df['CloseTime'] = df[time_like[-1]] if time_like else pd.NaT
    return df.reset_index(drop=True)

def detect_comment_col(df: pd.DataFrame) -> str | None:
    cand = None
    for c in df.columns[::-1]:
        s = df[c].astype(str)
        if s.str.contains('tp|sl|Close Position|GoldFlash|Long pyramiding|Close', case=False, na=False).sum() > 5:
            cand = c
            break
    return cand

def parse_log(path: Path):
    lines = Path(path).read_text(encoding='utf-8', errors='ignore').splitlines()
    # Regex with optional trailing symbol for profit lines
    order_re = re.compile(r'^(?P<ts>\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\.\\d+) \\| .*Order placed #(?P<ticket>\\d+) \\| (?P<symbol>[A-Z0-9.]+) \\| (?P<strategy>[A-Za-z]+) \\| (?P<side>BUY|SELL)')
    profit_re = re.compile(r'^(?P<ts>\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\.\\d+) \\| .*Profit booked #(?P<ticket>\\d+) \\((?P<pnl>-?\\d+\\.?\\d*)\\)(?: \\| (?P<symbol>[A-Z0-9.]+))?')
    loss_re   = re.compile(r'^(?P<ts>\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\.\\d+) \\| .*Loss cut #(?P<ticket>\\d+) \\((?P<pnl>-?\\d+\\.?\\d*)\\) \\| (?P<symbol>[A-Z0-9.]+)')
    err_re    = re.compile(r'\\[main error\\] (?P<msg>.*)')

    orders_map = {}  # ticket -> dict(symbol, strategy, side, ts)
    closures = []    # {ts, ticket, pnl, type, symbol?}
    errors = []
    trail_updates = 0

    for ln in lines:
        m = order_re.search(ln)
        if m:
            gd = m.groupdict()
            orders_map[gd['ticket']] = {
                'symbol': gd['symbol'],
                'strategy': gd['strategy'],
                'side': gd['side'],
                'ts': pd.to_datetime(gd['ts'])
            }
            continue
        m = profit_re.search(ln)
        if m:
            gd = m.groupdict()
            closures.append({
                'ts': pd.to_datetime(gd['ts']),
                'ticket': gd['ticket'],
                'pnl': float(gd['pnl']),
                'close_type': 'profit',
                'symbol': gd.get('symbol') or None
            })
            continue
        m = loss_re.search(ln)
        if m:
            gd = m.groupdict()
            closures.append({
                'ts': pd.to_datetime(gd['ts']),
                'ticket': gd['ticket'],
                'pnl': float(gd['pnl']),
                'close_type': 'loss',
                'symbol': gd.get('symbol') or None
            })
            continue
        if 'Trailing SL' in ln:
            trail_updates += 1
        m = err_re.search(ln)
        if m:
            errors.append(ln)

    closures_df = pd.DataFrame(closures)
    if not closures_df.empty:
        odf = pd.DataFrame.from_dict(orders_map, orient='index').reset_index(names='ticket')
        # Try to fill missing symbols in closures from orders (by ticket)
        closures_df = closures_df.merge(odf[['ticket','symbol','strategy','ts']].rename(columns={'ts':'order_ts'}), on='ticket', how='left', suffixes=('','_ord'))
        # If closure symbol missing but order symbol present, fill
        closures_df['symbol'] = closures_df['symbol'].fillna(closures_df['symbol_ord'])
        # If no strategy (because ticket mismatch), infer by nearest prior order of same symbol
        if closures_df['strategy'].isna().any():
            odf = odf.sort_values('ts')
            def infer_strategy(row):
                if pd.isna(row['symbol']) or pd.isna(row['ts']):
                    return np.nan
                m = odf[(odf['symbol']==row['symbol']) & (odf['ts']<=row['ts'])].tail(1)
                return m['strategy'].iloc[0] if len(m)>0 else np.nan
            closures_df['strategy'] = closures_df.apply(infer_strategy, axis=1)
    else:
        closures_df = pd.DataFrame(columns=['ts','ticket','pnl','close_type','symbol','strategy','order_ts'])

    return orders_map, closures_df, errors, trail_updates

def strategy_performance(orders_map, closures_df):
    # Orders placed by strategy
    oc = pd.Series([v['strategy'] for v in orders_map.values()]).value_counts()
    orders_counts = oc.rename_axis('strategy').reset_index(name='Orders_Placed')

    if closures_df.empty:
        perf = pd.DataFrame(columns=['strategy','Completed_Trades','Profitable_Trades','Losing_Trades','Total_Profit','Total_Loss','Total_PnL'])
    else:
        pf = (closures_df
              .assign(is_win=lambda d: d['pnl']>0,
                      pos_p=lambda d: d['pnl'].where(d['pnl']>0,0.0),
                      neg_p=lambda d: d['pnl'].where(d['pnl']<0,0.0))
              .groupby('strategy', dropna=False)
              .agg(Completed_Trades=('pnl','count'),
                   Profitable_Trades=('is_win','sum'),
                   Total_Profit=('pos_p','sum'),
                   Total_Loss=('neg_p','sum'),
                   Total_PnL=('pnl','sum'))
              .reset_index())
        pf['Losing_Trades'] = pf['Completed_Trades'] - pf['Profitable_Trades']
        perf = pf

    out = orders_counts.merge(perf, on='strategy', how='left').fillna(0)
    out['Hit_Rate_pct'] = out.apply(
        lambda r: (r['Profitable_Trades']/(r['Profitable_Trades']+r['Losing_Trades'])*100) if (r['Profitable_Trades']+r['Losing_Trades'])>0 else 0,
        axis=1
    ).round(1)
    out['Verdict'] = out['Total_PnL'].apply(lambda x: '▲ Profitable' if x>0 else ('▼ Loss' if x<0 else '▼ Marginal'))
    cols = ['strategy','Orders_Placed','Completed_Trades','Profitable_Trades','Losing_Trades','Hit_Rate_pct','Total_Profit','Total_Loss','Total_PnL','Verdict']
    return out[cols].sort_values(['Total_PnL','Hit_Rate_pct'], ascending=[False, False])

def symbol_performance(closed_hist):
    d = (closed_hist[closed_hist['Symbol'].isin(['XAUUSD.s','XAGUSD.s','NAS100.s'])]
         .assign(is_win=lambda x: x['Profit']>0,
                 pos_p=lambda d: d['Profit'].where(d['Profit']>0,0.0),
                 neg_p=lambda d: d['Profit'].where(d['Profit']<0,0.0))
         .groupby('Symbol', as_index=False)
         .agg(Trades=('Profit','count'),
              Net_Profit=('pos_p','sum'),
              Net_Loss=('neg_p','sum'),
              Amount=('Profit','sum'),
              Hit_Rate_pct=('is_win','mean')))
    d['Hit_Rate_pct'] = (d['Hit_Rate_pct']*100).round(1)
    d[['Net_Profit','Net_Loss','Amount']] = d[['Net_Profit','Net_Loss','Amount']].round(2)
    return d.sort_values('Amount', ascending=False)

def combined_bot_closed(closures_df):
    if closures_df.empty:
        return pd.DataFrame(columns=['strategy','symbol','Trades','Net_PnL','Verdict'])
    g = (closures_df.groupby(['strategy','symbol'], dropna=False)
         .agg(Trades=('pnl','count'), Net_PnL=('pnl','sum')).reset_index())
    g['Verdict'] = g['Net_PnL'].apply(lambda x: '▲ Profitable' if x>0 else ('▼ Loss' if x<0 else '▼ Marginal'))
    return g.sort_values(['strategy','symbol'])

def manual_vs_bot(closed_orders, orders_map):
    if closed_orders.empty:
        return pd.DataFrame(columns=['Strategy','Symbol','Total_Trades','Bot_Closed','Manual_Closed','Bot_pct','Manual_pct'])
    comment_col = detect_comment_col(closed_orders)
    if not comment_col:
        # No comments to split
        mvb = (closed_orders.groupby(['Symbol'])
               .agg(Total_Trades=('Profit','count'))
               .reset_index())
        mvb['Bot_Closed'] = 0
        mvb['Manual_Closed'] = mvb['Total_Trades']
        mvb['Strategy'] = 'Unknown'
        mvb['Bot_pct'] = 0.0
        mvb['Manual_pct'] = 100.0
        return mvb[['Strategy','Symbol','Total_Trades','Bot_Closed','Manual_Closed','Bot_pct','Manual_pct']]
    clos = closed_orders.copy()
    clos['is_bot'] = clos[comment_col].astype(str).str.contains('\\[tp|\\[sl', case=False, na=False)
    clos['is_manual'] = clos[comment_col].astype(str).str.contains('Close Position', case=False, na=False)
    # Map Strategy by nearest prior order timestamp for same symbol
    odf = pd.DataFrame.from_dict(orders_map, orient='index').reset_index(names='ticket') if orders_map else pd.DataFrame(columns=['ticket','symbol','strategy','ts'])
    odf = odf.sort_values('ts') if not odf.empty else odf
    sym_col = 'Symbol'
    time_col = 'CloseTime'
    def map_strategy(row):
        if odf.empty or pd.isna(row.get(time_col)):
            return np.nan
        sym = str(row.get(sym_col, '')).strip()
        m = odf[(odf['symbol']==sym) & (odf['ts']<=row[time_col])].tail(1)
        return m['strategy'].iloc[0] if len(m)>0 else np.nan
    clos['Strategy'] = clos.apply(map_strategy, axis=1)
    mvb = (clos.groupby(['Strategy','Symbol'], dropna=False)
           .agg(Total_Trades=('Profit','count'),
                Bot_Closed=('is_bot','sum'),
                Manual_Closed=('is_manual','sum'))
           .reset_index())
    mvb['Strategy'] = mvb['Strategy'].fillna('Unknown')
    mvb['Bot_pct'] = (mvb['Bot_Closed']/mvb['Total_Trades']*100).round(1)
    mvb['Manual_pct'] = (mvb['Manual_Closed']/mvb['Total_Trades']*100).round(1)
    return mvb.sort_values(['Strategy','Symbol'])

def strategies_over_time(closures_df):
    if closures_df.empty or closures_df['strategy'].isna().all():
        return pd.DataFrame(columns=['strategy','date','cum_pnl'])
    st = closures_df.dropna(subset=['strategy']).copy()
    st['date'] = st['ts'].dt.floor('D')
    cum = (st.groupby(['strategy','date'])['pnl'].sum()
             .groupby(level=0).cumsum().reset_index(name='cum_pnl'))
    return cum

def drawdown_stats(closures_df):
    if closures_df.empty:
        return {'Max_Balance_Drawdown_Proxy': 0.0, 'Largest_Single_Loss': 0.0, 'Largest_Consecutive_Losses': 0}
    eq = closures_df.sort_values('ts').assign(eq=lambda d: d['pnl'].cumsum())
    roll_max = eq['eq'].cummax()
    max_dd = float((roll_max - eq['eq']).max()) if len(eq) else 0.0
    largest_loss = float(closures_df['pnl'].min()) if len(closures_df) else 0.0
    streak = 0
    max_streak = 0
    for v in closures_df.sort_values('ts')['pnl']:
        if v < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return {'Max_Balance_Drawdown_Proxy': round(max_dd, 2),
            'Largest_Single_Loss': round(largest_loss, 2),
            'Largest_Consecutive_Losses': int(max_streak)}

def save_chart_strategies_over_time(cum: pd.DataFrame, out_path: Path):
    if not HAVE_PLT or cum.empty:
        return
    plt.figure(figsize=(10,6))
    for strat, g in cum.groupby('strategy'):
        plt.plot(g['date'], g['cum_pnl'], label=strat)
    plt.title('Cumulative Net PnL by Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)

def save_chart_profit_equity(hist_df: pd.DataFrame, out_path: Path):
    # If Balance column exists, build equity curve; otherwise bar of realized PnL by day
    if not HAVE_PLT:
        return
    df = hist_df.copy()
    if 'Profit' not in df.columns:
        return
    df = df[df['Profit'].notna()].copy()
    if 'CloseTime' in df.columns and df['CloseTime'].notna().any():
        df['date'] = df['CloseTime'].dt.floor('D')
    else:
        df['date'] = df['Time'].dt.floor('D') if 'Time' in df.columns else pd.NaT
    daily = df.groupby('date', dropna=True)['Profit'].sum().reset_index()
    plt.figure(figsize=(10,6))
    plt.bar(daily['date'], daily['Profit'])
    plt.title('Realized PnL by Day')
    plt.xlabel('Date')
    plt.ylabel('Realized PnL')
    plt.tight_layout()
    plt.savefig(out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hist', default='ReportHistory-548269.xlsx')
    parser.add_argument('--orders', default='ReportHistory_orders-548269.xlsx')
    parser.add_argument('--log', default='goldflash_nexus_log.txt')
    args = parser.parse_args()

    hist = load_positions_xlsx(Path(args.hist))
    orders = load_positions_xlsx(Path(args.orders))
    closed_hist = hist[hist['Profit'].notna()].copy()
    closed_orders = orders[orders['Profit'].notna()].copy()

    orders_map, closures_df, errors, trail_updates = parse_log(Path(args.log))

    # 1) Strategy Performance (from log)
    strat_perf = strategy_performance(orders_map, closures_df)
    strat_perf.to_csv('strategy_performance.csv', index=False)

    # 2) Symbol Performance (from MT5)
    sym_perf = symbol_performance(closed_hist)
    sym_perf.to_csv('symbol_performance.csv', index=False)

    # 3) Combined Export & MT5 (Bot-Closed Trades Only) – from log closures aligned to strategies/symbols
    comb_bot = combined_bot_closed(closures_df)
    comb_bot.to_csv('combined_bot_closed.csv', index=False)

    # 4) Manual vs Bot-Closed (counts and %)
    mvb = manual_vs_bot(closed_orders, orders_map)
    mvb.to_csv('manual_vs_bot.csv', index=False)

    # 5) Strategies Over Time
    cum = strategies_over_time(closures_df)
    cum.to_csv('strategies_over_time.csv', index=False)
    save_chart_strategies_over_time(cum, Path('strategies_over_time.png'))

    # 6) Profit & Margin Impact (PnL by day chart; margin % requires margin columns if available)
    save_chart_profit_equity(closed_hist, Path('profit_by_day.png'))

    # 7) Recent Activity (Last 10 closure events)
    recent = closures_df.sort_values('ts').tail(10).copy()
    if not recent.empty:
        recent_out = recent[['ts','symbol','pnl']].copy()
        recent_out['Event'] = np.where(recent['pnl']>0, 'Profit booked', 'Loss cut')
        recent_out['Verdict'] = np.where(recent['pnl']>0, '▲ Profitable', '▼ Loss')
        recent_out = recent_out[['ts','Event','symbol','pnl','Verdict']]
    else:
        recent_out = pd.DataFrame(columns=['ts','Event','symbol','pnl','Verdict'])
    recent_out.to_csv('recent_activity.csv', index=False)

    # 9) Risk Indicators (volatility-based heuristic per strategy)
    if not closures_df.empty and closures_df['strategy'].notna().any():
        vol = closures_df.groupby('strategy')['pnl'].std().fillna(0)
        q = vol.rank(pct=True)
        def label(p):
            return 'Low' if p <= 0.2 else 'Medium' if p <= 0.4 else 'High' if p <= 0.7 else 'Very High'
        risk = q.map(label).reset_index()
        risk.columns = ['Strategy','Risk_Level']
    else:
        risk = pd.DataFrame(columns=['Strategy','Risk_Level'])
    risk.to_csv('risk_indicators.csv', index=False)

    # 10) Drawdown Analysis (proxy from log closures)
    dd = drawdown_stats(closures_df)
    pd.DataFrame([dd]).to_csv('drawdown_analysis.csv', index=False)

    # 11) Execution Notes
    exec_notes = {
        'Trailing_SL_TP_Updates': int(trail_updates),
        'MT5_API_Error_Lines': int(sum(1 for e in errors if 'MetaTrader5' in e)),
        'Any_Volume_Errors': any('volume' in e.lower() for e in errors)
    }
    Path('execution_notes.json').write_text(json.dumps(exec_notes, indent=2))

    # 12) Trading Recommendations (top-2 by hit rate and by PnL)
    sp = strat_perf.copy()
    best_by_hit = sp.sort_values('Hit_Rate_pct', ascending=False).head(2)[['strategy','Hit_Rate_pct']]
    best_by_pnl = sp.sort_values('Total_PnL', ascending=False).head(2)[['strategy','Total_PnL']]
    best_by_hit.to_csv('best_by_hit_rate.csv', index=False)
    best_by_pnl.to_csv('best_by_total_pnl.csv', index=False)

    # Print a short, human-readable summary
    print('\\n=== Strategy Performance ===')
    print(strat_perf.to_string(index=False))
    print('\\n=== Symbol Performance ===')
    print(sym_perf.to_string(index=False))
    print('\\n=== Combined Bot-Closed ===')
    print(comb_bot.to_string(index=False))
    print('\\n=== Manual vs Bot-Closed ===')
    print(mvb.to_string(index=False))
    print('\\n=== Recent Activity (last 10) ===')
    print(recent_out.to_string(index=False))
    print('\\n=== Drawdown (proxy from log) ===')
    print(dd)
    print('\\n=== Execution Notes ===')
    print(exec_notes)
    print('\\nArtifacts written: strategy_performance.csv, symbol_performance.csv, combined_bot_closed.csv, manual_vs_bot.csv, strategies_over_time.csv, recent_activity.csv, risk_indicators.csv, drawdown_analysis.csv, best_by_hit_rate.csv, best_by_total_pnl.csv, strategies_over_time.png (if matplotlib), profit_by_day.png (if matplotlib)')

if __name__ == '__main__':
    main()
