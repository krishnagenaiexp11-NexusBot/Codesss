#!/usr/bin/env python3
# ============================================
# RUN BACKTEST - Main Script
# ============================================

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from backtest_framework import RuleBacktester
from data_preparation import DataPreparator
import json


def print_results(metrics):
    """Pretty print backtest results"""
    
    print("\n" + "="*70)
    print("🎯 BACKTEST RESULTS")
    print("="*70)
    
    if 'error' in metrics:
        print(f"❌ {metrics['error']}")
        return
    
    print(f"\n📊 TRADE STATISTICS:")
    print(f"   Total Trades:           {metrics['total_trades']}")
    print(f"   Winning Trades:         {metrics['winning_trades']}")
    print(f"   Losing Trades:          {metrics['losing_trades']}")
    print(f"   Win Rate:               {metrics['win_rate']}%")
    
    print(f"\n💰 PROFIT/LOSS:")
    print(f"   Total P&L:              ${metrics['total_pnl']:.2f}")
    print(f"   Avg P&L per Trade:      ${metrics['avg_pnl_per_trade']:.2f}")
    print(f"   Gross Profit:           ${metrics['gross_profit']:.2f}")
    print(f"   Gross Loss:             ${abs(metrics['gross_loss']):.2f}")
    print(f"   Profit Factor:          {metrics['profit_factor']}")
    
    print(f"\n📈 TRADE DETAILS:")
    print(f"   Best Trade:             ${metrics['best_trade']:.2f}")
    print(f"   Worst Trade:            ${metrics['worst_trade']:.2f}")
    print(f"   Avg Win:                ${metrics['avg_win']:.2f}")
    print(f"   Avg Loss:               ${metrics['avg_loss']:.2f}")
    
    print(f"\n⚠️  RISK METRICS:")
    print(f"   Max Drawdown:           {metrics['max_drawdown']}%")
    print(f"   Max Consecutive Wins:   {metrics['max_consecutive_wins']}")
    print(f"   Max Consecutive Losses: {metrics['max_consecutive_losses']}")
    print(f"   Sharpe Ratio:           {metrics['sharpe_ratio']}")
    
    print(f"\n💵 ACCOUNT:")
    print(f"   Starting Balance:       $10,000.00")
    print(f"   Final Balance:          ${metrics['final_balance']:.2f}")
    print(f"   ROI:                    {metrics['roi']}%")
    
    print("\n" + "="*70)
    
    # Print first 10 trades
    if metrics['trades']:
        print(f"\n📋 SAMPLE TRADES (First 10):")
        print("-" * 120)
        print(f"{'#':<3} {'Direction':<6} {'Entry':<10} {'Exit':<10} {'P&L':<10} {'Pips':<8} {'Reason':<25}")
        print("-" * 120)
        
        for idx, trade in enumerate(metrics['trades'][:10], 1):
            reason = trade['exit_reason'][:22]
            pnl = trade['pnl']
            emoji = '✅' if pnl > 0 else '❌'
            print(f"{idx:<3} {trade['direction']:<6} {trade['entry_price']:<10.2f} "
                  f"{trade['exit_price']:<10.2f} ${pnl:<9.2f} {trade['pnl_pips']:<7.2f} {reason:<25} {emoji}")
        
        print("-" * 120)


def run_backtest(data_source='sample'):
    """
    Run backtest
    data_source: 'sample' (synthetic) or path to CSV file
    """
    
    print("\n" + "="*70)
    print("🚀 INTELLIGENT TRADING BOT - BACKTEST")
    print("="*70)
    
    # Load data
    print(f"\n📥 Loading data source: {data_source}...")
    
    if data_source == 'sample':
        df = DataPreparator.create_sample_data(num_candles=1000)
        print(f"   ✅ Generated {len(df)} synthetic candles")
    else:
        df = DataPreparator.load_csv_data(data_source)
        if df is None:
            print("❌ Failed to load data")
            return
        print(f"   ✅ Loaded {len(df)} candles from {data_source}")
    
    # Add indicators
    print(f"\n⚙️  Adding technical indicators...")
    df = DataPreparator.add_technical_indicators(df)
    print(f"   ✅ Indicators added")
    print(f"   Columns: {', '.join(df.columns.tolist())}")
    
    # Run backtest
    print(f"\n🔄 Running backtest...")
    backtester = RuleBacktester(initial_balance=10000, commission=0.0001)
    metrics = backtester.backtest(df, verbose=True)
    
    # Print results
    print_results(metrics)
    
    # Save results to JSON
    if metrics['trades']:
        output_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert non-serializable objects
        metrics_copy = metrics.copy()
        metrics_copy['trades'] = [
            {k: float(v) if isinstance(v, np.floating) else v for k, v in trade.items()}
            for trade in metrics_copy['trades']
        ]
        
        with open(output_file, 'w') as f:
            json.dump(metrics_copy, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    return metrics


if __name__ == '__main__':
    
    # Run with sample data
    print("\n" + "#"*70)
    print("# TEST 1: Sample Synthetic Data")
    print("#"*70)
    metrics1 = run_backtest('sample')
    
    # To run with your own CSV data, uncomment and modify:
    # print("\n" + "#"*70)
    # print("# TEST 2: Your CSV Data")
    # print("#"*70)
    # metrics2 = run_backtest('your_data.csv')
    
    print("\n✅ Backtest completed!")
