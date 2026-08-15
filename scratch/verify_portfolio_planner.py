#!/usr/bin/env python3
"""
Test script for Goal-based Lump Sum & SIP Portfolio Planner.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio_planner import generate_lump_sum_portfolio, generate_sip_plan

def test_planner():
    print("=== 1. Testing Lump-Sum Portfolio Allocator (₹5,00,000, 3Y, Moderate) ===")
    lump = generate_lump_sum_portfolio(500000.0, 3.0, "moderate")
    print(f"Invested: {lump['capital_amount']}")
    print(f"Horizon: {lump['horizon_years']}")
    print(f"Risk: {lump['risk_profile']}")
    print(f"Expected CAGR: {lump['expected_cagr']}")
    print(f"Projected Maturity Corpus: {lump['expected_maturity_corpus']} (+{lump['expected_gain_rupees']})")
    print(f"Portfolio Max Drawdown: {lump['portfolio_max_drawdown']}")
    print("Allocated Assets:")
    for a in lump['assets']:
        print(f"  - {a['asset_name']} ({a['category']}): {a['allocation_pct']} -> {a['allocation_rupees']}")
    assert len(lump['assets']) >= 3, "Missing asset allocation buckets!"
    print("✓ Lump-Sum Portfolio Planner test passed!\n")

    print("=== 2. Testing Smart SIP Basket Builder (₹10,000/mo, 5Y, High Risk) ===")
    sip = generate_sip_plan(10000.0, 5.0, "aggressive")
    print(f"Monthly SIP: {sip['monthly_sip_amount']}")
    print(f"Total Invested: {sip['total_invested']}")
    print(f"Projected Future Corpus: {sip['projected_corpus']} (+{sip['wealth_gain_rupees']})")
    print(f"Expected CAGR: {sip['expected_cagr']}")
    print("Curated Basket:")
    for f in sip['basket']:
        print(f"  - {f['scheme_name']}: {f['monthly_split_pct_fmt']} ({f['monthly_rupees_fmt']}/mo)")
    assert len(sip['basket']) >= 3, "Missing SIP basket funds!"
    print("✓ Smart SIP Basket Architect test passed!\n")

    print("=== ALL PORTFOLIO PLANNER TESTS PASSED! ===")

if __name__ == "__main__":
    test_planner()
