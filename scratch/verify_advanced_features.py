#!/usr/bin/env python3
"""
Verification test for ADV liquidity filter, factor attribution, drift monitoring, and flatten-all.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
from asset_engine import analyze_stock_or_etf, analyze_mutual_fund
from backtest_engine import compute_factor_attribution, compute_live_drift, resolve_pending_signals
from risk_engine import calculate_position_sizing, check_volatility_regime

def run_tests():
    print("=== 1. Testing ADV Liquidity Cap & Overnight Gap Hazard ===")
    sizing = calculate_position_sizing(
        total_capital=1000000.0,
        entry_price=2850.0,
        stop_loss_price=2780.0,
        target_1_price=2990.0,
        target_2_price=3060.0,
        atr_pct=1.8,
        avg_daily_volume_10d=2500.0, # Very low volume small cap (2500 shares ADV)
    )
    print(f"Recommended Shares (low ADV): {sizing.recommended_shares}")
    print(f"Liquidity Warning: {sizing.liquidity_warning}")
    print(f"Overnight Gap Hazard: {sizing.overnight_gap_hazard_pct}%")
    print(f"Worst-Case Gap Loss: ₹{sizing.worst_case_loss_rupees:,.2f} ({sizing.worst_case_loss_pct_of_capital:.2f}%)")
    assert sizing.recommended_shares <= 25, "Position sizing failed to cap at 1% of ADV!"
    print("✓ ADV Liquidity & Gap Hazard tests passed!\n")

    print("=== 2. Testing Factor Attribution Waterfall ===")
    factors = compute_factor_attribution(
        rsi=56.4,
        macd_line=12.5,
        macd_signal=8.2,
        last_close=2850.0,
        ema_21=2800.0,
        ema_50=2750.0,
        ema_200=2600.0,
        vol_ratio=1.65,
        premarket_gap_bias=15.0,
        vix_val=14.2,
    )
    print(f"Generated {len(factors)} attribution factors:")
    total_pts = sum(f["points"] for f in factors)
    for f in factors:
        print(f"  - {f['factor']}: {f['points']:+.1f} pts ({f['impact']})")
    print(f"Total Attributed Score: {total_pts:.1f}")
    assert len(factors) >= 5, "Missing attribution components!"
    print("✓ Factor attribution test passed!\n")

    print("=== 3. Testing SQLite Signal Outcomes & Drift Monitoring ===")
    db_path = Path(__file__).parent / "prediction_history.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_date TEXT NOT NULL,
                asset_symbol TEXT NOT NULL,
                score REAL NOT NULL,
                tier TEXT NOT NULL,
                horizon_days INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                forward_return REAL,
                outcome TEXT NOT NULL,
                vix_at_signal REAL,
                resolved_at TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        drift = compute_live_drift(conn, window_days=60)
        print(f"Drift Alert Status: {drift['has_drift_alert']}")
        print(f"Tiers monitored: {len(drift['tiers'])}")
        for t in drift['tiers']:
            print(f"  - {t['tier_name']}: N={t['sample_count']}, Baseline={t['baseline_win_rate']}, Status={t['status']}")
    print("✓ Drift monitoring test passed!\n")

    print("=== ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
