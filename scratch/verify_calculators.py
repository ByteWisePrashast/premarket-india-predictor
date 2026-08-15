import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from portfolio_planner import calculate_regular_sip, calculate_step_up_sip, calculate_lump_sum_calculator

def test_calculators():
    print("Testing Regular SIP...")
    sip_res = calculate_regular_sip(monthly_amount=10000.0, annual_return_pct=14.0, horizon_years=10.0)
    assert sip_res["ok"] is True
    assert sip_res["total_invested_num"] == 1200000.0
    assert sip_res["future_value_num"] > 2500000.0
    print(f"  SIP 10k/mo, 14%, 10Y: Invested={sip_res['total_invested']}, FV={sip_res['future_value']}")

    print("Testing Step-Up SIP...")
    step_up_res = calculate_step_up_sip(initial_monthly_amount=10000.0, annual_step_up_pct=10.0, annual_return_pct=14.0, horizon_years=10.0)
    assert step_up_res["ok"] is True
    assert step_up_res["future_value_num"] > sip_res["future_value_num"]
    print(f"  Step-Up SIP 10k/mo (+10%/yr), 14%, 10Y: Invested={step_up_res['total_invested']}, FV={step_up_res['future_value']}")
    print(f"  Advantage vs Flat SIP: {step_up_res['comparison_vs_flat_sip']['extra_corpus_created']} ({step_up_res['comparison_vs_flat_sip']['multiplier']})")

    print("Testing Lump-Sum Calculator...")
    lump_res = calculate_lump_sum_calculator(principal_amount=500000.0, annual_return_pct=14.0, horizon_years=10.0, inflation_rate_pct=6.0)
    assert lump_res["ok"] is True
    assert lump_res["future_value_nominal_num"] > 1800000.0
    assert lump_res["future_value_inflation_adjusted_num"] > 500000.0
    print(f"  Lump-Sum 5L, 14%, 10Y: Nominal FV={lump_res['future_value_nominal']}, Real FV={lump_res['future_value_inflation_adjusted']}")

    print("ALL CALCULATOR TESTS PASSED SUCCESSFULLY! ✅")

if __name__ == "__main__":
    test_calculators()
