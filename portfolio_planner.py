#!/usr/bin/env python3
"""
Quantitative Portfolio Generator, Smart SIP Architect & Wealth Calculator Suite for Indian Markets.

Provides:
1. Regular Monthly SIP Compound Calculator.
2. Step-Up (Top-Up) SIP Calculator (Annual % increase vs flat SIP comparison).
3. Lump-Sum Compound Growth Calculator (Nominal vs Inflation-Adjusted Purchasing Power).
4. Goal-based Lump-Sum Portfolio Allocator (Asset Allocation across ETFs, MFs, Gold, Bluechips).
5. Smart Multi-Fund Monthly SIP Basket Builder.
6. Year-by-Year Compounding Schedules & Interactive Visual SVG Trajectory Charts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from asset_engine import fmt_curr, fmt_pct

# Historical benchmark returns and volatilities (CAGR & Std Dev in Indian Context)
ASSET_CLASS_PROFILES = {
    "large_cap_index": {"cagr": 13.5, "volatility": 14.2, "max_dd": -24.0, "name": "Large Cap Index / ETFs (Nifty 50 / Next 50)"},
    "flexi_cap": {"cagr": 16.0, "volatility": 15.5, "max_dd": -26.0, "name": "Flexi Cap Equity Funds (Dynamic Multi-Cap)"},
    "mid_small_cap": {"cagr": 20.5, "volatility": 21.0, "max_dd": -36.0, "name": "Mid & Small Cap Growth Funds"},
    "gold_commodity": {"cagr": 11.5, "volatility": 12.0, "max_dd": -15.0, "name": "Physical Gold / Silver ETFs (Hedge)"},
    "high_conviction_stocks": {"cagr": 18.0, "volatility": 22.0, "max_dd": -32.0, "name": "High-Conviction Swing & Growth Stocks"},
    "liquid_arbitrage": {"cagr": 6.8, "volatility": 2.5, "max_dd": -1.2, "name": "Arbitrage / Liquid Capital Buffer"},
}


# ============================================================================
# 1. REGULAR SIP CALCULATOR
# ============================================================================

def calculate_regular_sip(
    monthly_amount: float,
    annual_return_pct: float = 14.0,
    horizon_years: float = 10.0,
) -> dict[str, Any]:
    """Calculates Future Value, Total Invested, Wealth Gain, and Year-by-Year compounding for a flat monthly SIP."""
    p = max(500.0, float(monthly_amount))
    r_annual = max(1.0, float(annual_return_pct))
    years = max(1.0, float(horizon_years))
    total_months = int(years * 12)
    i = (r_annual / 100.0) / 12.0

    # Total Invested
    total_invested = p * total_months

    # Future Value formula: P * [((1 + i)^n - 1) / i] * (1 + i)
    fv = p * (((1.0 + i) ** total_months - 1.0) / i) * (1.0 + i)
    wealth_gain = fv - total_invested
    gain_pct = (wealth_gain / total_invested * 100.0) if total_invested else 0.0

    # Year-by-Year Growth Table
    yearly_schedule = []
    for y in range(1, int(years) + 1):
        m = y * 12
        inv_y = p * m
        fv_y = p * (((1.0 + i) ** m - 1.0) / i) * (1.0 + i)
        gain_y = fv_y - inv_y
        yearly_schedule.append({
            "year": y,
            "total_invested": fmt_curr(inv_y),
            "total_invested_num": inv_y,
            "future_value": fmt_curr(fv_y),
            "future_value_num": fv_y,
            "wealth_gain": fmt_curr(gain_y),
            "wealth_gain_num": gain_y,
        })

    chart_svg = make_trajectory_chart_svg(yearly_schedule, mode="sip")

    return {
        "ok": True,
        "type": "regular_sip",
        "monthly_investment": fmt_curr(p),
        "monthly_investment_num": p,
        "expected_return_pct": f"{r_annual:.1f}%",
        "expected_return_pct_num": r_annual,
        "time_period_years": f"{years:.0f} Years",
        "time_period_years_num": years,
        "total_invested": fmt_curr(total_invested),
        "total_invested_num": total_invested,
        "wealth_gain": fmt_curr(wealth_gain),
        "wealth_gain_num": wealth_gain,
        "gain_pct": fmt_pct(gain_pct),
        "future_value": fmt_curr(fv),
        "future_value_num": fv,
        "yearly_schedule": yearly_schedule,
        "chart_svg": chart_svg,
    }


# ============================================================================
# 2. STEP-UP (TOP-UP) SIP CALCULATOR
# ============================================================================

def calculate_step_up_sip(
    initial_monthly_amount: float,
    annual_step_up_pct: float = 10.0,
    annual_return_pct: float = 14.0,
    horizon_years: float = 10.0,
) -> dict[str, Any]:
    """Calculates wealth creation when monthly SIP is increased by a fixed percentage (e.g. 10%) every year."""
    p_initial = max(500.0, float(initial_monthly_amount))
    step_up_pct = max(0.0, float(annual_step_up_pct))
    r_annual = max(1.0, float(annual_return_pct))
    years = max(1.0, float(horizon_years))
    total_years_int = int(years)
    i = (r_annual / 100.0) / 12.0

    total_invested = 0.0
    fv_step_up = 0.0
    yearly_schedule = []

    # Calculate month-by-month compounding with yearly step-up
    current_monthly_sip = p_initial

    for y in range(1, total_years_int + 1):
        year_invested = 0.0
        for m in range(1, 13):
            # Months remaining till horizon for this specific installment
            months_remaining = (total_years_int - y) * 12 + (12 - m + 1)
            future_val_of_installment = current_monthly_sip * ((1.0 + i) ** months_remaining)
            fv_step_up += future_val_of_installment
            year_invested += current_monthly_sip
            total_invested += current_monthly_sip

        # Cumulative metrics at end of year y
        # We compute FV up to year y for the schedule
        cumulative_inv = total_invested
        yearly_schedule.append({
            "year": y,
            "monthly_sip_this_year": fmt_curr(current_monthly_sip),
            "monthly_sip_this_year_num": current_monthly_sip,
            "total_invested": fmt_curr(cumulative_inv),
            "total_invested_num": cumulative_inv,
            "future_value": fmt_curr(fv_step_up if y == total_years_int else 0.0), # filled below
        })

        # Step up monthly SIP for next year
        current_monthly_sip = current_monthly_sip * (1.0 + (step_up_pct / 100.0))

    # Accurate intermediate FV calculations for the schedule
    for idx, row in enumerate(yearly_schedule):
        y_sub = row["year"]
        sub_fv = 0.0
        cur_p = p_initial
        for y_i in range(1, y_sub + 1):
            for m_i in range(1, 13):
                rem_m = (y_sub - y_i) * 12 + (12 - m_i + 1)
                sub_fv += cur_p * ((1.0 + i) ** rem_m)
            cur_p = cur_p * (1.0 + (step_up_pct / 100.0))
        row["future_value"] = fmt_curr(sub_fv)
        row["future_value_num"] = sub_fv
        row["wealth_gain"] = fmt_curr(sub_fv - row["total_invested_num"])
        row["wealth_gain_num"] = sub_fv - row["total_invested_num"]

    wealth_gain_step_up = fv_step_up - total_invested
    gain_pct_step_up = (wealth_gain_step_up / total_invested * 100.0) if total_invested else 0.0

    # Flat SIP comparison for the same starting amount
    flat_sip = calculate_regular_sip(p_initial, r_annual, years)
    flat_fv = flat_sip["future_value_num"]
    flat_inv = flat_sip["total_invested_num"]
    step_up_advantage_rupees = fv_step_up - flat_fv
    step_up_multiplier = (fv_step_up / flat_fv) if flat_fv > 0 else 1.0

    chart_svg = make_trajectory_chart_svg(yearly_schedule, mode="step_up")

    return {
        "ok": True,
        "type": "step_up_sip",
        "initial_monthly_investment": fmt_curr(p_initial),
        "initial_monthly_investment_num": p_initial,
        "annual_step_up_pct": f"{step_up_pct:.1f}%",
        "annual_step_up_pct_num": step_up_pct,
        "expected_return_pct": f"{r_annual:.1f}%",
        "expected_return_pct_num": r_annual,
        "time_period_years": f"{years:.0f} Years",
        "time_period_years_num": years,
        "total_invested": fmt_curr(total_invested),
        "total_invested_num": total_invested,
        "wealth_gain": fmt_curr(wealth_gain_step_up),
        "wealth_gain_num": wealth_gain_step_up,
        "gain_pct": fmt_pct(gain_pct_step_up),
        "future_value": fmt_curr(fv_step_up),
        "future_value_num": fv_step_up,
        "comparison_vs_flat_sip": {
            "flat_sip_invested": fmt_curr(flat_inv),
            "flat_sip_corpus": fmt_curr(flat_fv),
            "extra_corpus_created": fmt_curr(step_up_advantage_rupees),
            "multiplier": f"{step_up_multiplier:.2f}x",
        },
        "yearly_schedule": yearly_schedule,
        "chart_svg": chart_svg,
    }


# ============================================================================
# 3. LUMP-SUM INVESTMENT CALCULATOR
# ============================================================================

def calculate_lump_sum_calculator(
    principal_amount: float,
    annual_return_pct: float = 14.0,
    horizon_years: float = 10.0,
    inflation_rate_pct: float = 6.0,
) -> dict[str, Any]:
    """Calculates compound growth on a one-time lump-sum investment, with inflation-adjusted real purchasing power."""
    p = max(5000.0, float(principal_amount))
    r_annual = max(1.0, float(annual_return_pct))
    years = max(1.0, float(horizon_years))
    inflation = max(0.0, float(inflation_rate_pct))

    # FV = P * (1 + r)^n
    fv_nominal = p * ((1.0 + (r_annual / 100.0)) ** years)
    wealth_gain_nominal = fv_nominal - p
    gain_pct_nominal = (wealth_gain_nominal / p * 100.0) if p else 0.0

    # Inflation adjusted real purchasing power
    # Real FV = Nominal FV / (1 + inflation)^years
    fv_real = fv_nominal / ((1.0 + (inflation / 100.0)) ** years)
    wealth_gain_real = fv_real - p

    yearly_schedule = []
    for y in range(1, int(years) + 1):
        fv_y = p * ((1.0 + (r_annual / 100.0)) ** y)
        fv_real_y = fv_y / ((1.0 + (inflation / 100.0)) ** y)
        gain_y = fv_y - p
        yearly_schedule.append({
            "year": y,
            "total_invested": fmt_curr(p),
            "total_invested_num": p,
            "future_value": fmt_curr(fv_y),
            "future_value_num": fv_y,
            "wealth_gain": fmt_curr(gain_y),
            "wealth_gain_num": gain_y,
            "real_purchasing_power": fmt_curr(fv_real_y),
        })

    chart_svg = make_trajectory_chart_svg(yearly_schedule, mode="lumpsum")

    return {
        "ok": True,
        "type": "lumpsum",
        "initial_investment": fmt_curr(p),
        "initial_investment_num": p,
        "expected_return_pct": f"{r_annual:.1f}%",
        "expected_return_pct_num": r_annual,
        "time_period_years": f"{years:.0f} Years",
        "time_period_years_num": years,
        "inflation_rate_pct": f"{inflation:.1f}%",
        "total_invested": fmt_curr(p),
        "total_invested_num": p,
        "wealth_gain_nominal": fmt_curr(wealth_gain_nominal),
        "wealth_gain_nominal_num": wealth_gain_nominal,
        "gain_pct": fmt_pct(gain_pct_nominal),
        "future_value_nominal": fmt_curr(fv_nominal),
        "future_value_nominal_num": fv_nominal,
        "future_value_inflation_adjusted": fmt_curr(fv_real),
        "future_value_inflation_adjusted_num": fv_real,
        "real_wealth_gain": fmt_curr(wealth_gain_real),
        "yearly_schedule": yearly_schedule,
        "chart_svg": chart_svg,
    }


# ============================================================================
# VISUAL TRAJECTORY SVG GENERATOR
# ============================================================================

def make_trajectory_chart_svg(yearly_schedule: list[dict[str, Any]], mode: str = "sip", width: int = 720, height: int = 240) -> str:
    """Generates an interactive SVG area and line compounding trajectory chart."""
    if not yearly_schedule:
        return ""

    padding_left = 65
    padding_right = 25
    padding_top = 20
    padding_bottom = 35

    plot_w = width - padding_left - padding_right
    plot_h = height - padding_top - padding_bottom

    n = len(yearly_schedule)
    max_val = max(row["future_value_num"] for row in yearly_schedule) * 1.08
    max_val = max(max_val, 10000.0)

    # Compute coordinates
    invested_points = []
    future_points = []

    for idx, row in enumerate(yearly_schedule):
        x = padding_left + (idx / max(1, n - 1)) * plot_w
        y_inv = padding_top + plot_h - (row["total_invested_num"] / max_val * plot_h)
        y_fv = padding_top + plot_h - (row["future_value_num"] / max_val * plot_h)
        invested_points.append((x, y_inv))
        future_points.append((x, y_fv))

    inv_path_d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in invested_points)
    fv_path_d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in future_points)

    # Area fill under Future Value curve
    first_x, last_x = future_points[0][0], future_points[-1][0]
    base_y = padding_top + plot_h
    fv_area_d = fv_path_d + f" L {last_x:.1f} {base_y:.1f} L {first_x:.1f} {base_y:.1f} Z"

    # Color themes based on mode
    color_fv = "#10b981" if mode == "sip" else "#38bdf8" if mode == "step_up" else "#f59e0b"
    color_inv = "#64748b"

    # Gridlines
    grid_lines = []
    for g in [0.25, 0.5, 0.75, 1.0]:
        y_pos = padding_top + plot_h - (g * plot_h)
        val_label = fmt_curr(g * max_val)
        grid_lines.append(f'<line x1="{padding_left}" y1="{y_pos:.1f}" x2="{width - padding_right}" y2="{y_pos:.1f}" stroke="rgba(255,255,255,0.06)" stroke-dasharray="3,3" />')
        grid_lines.append(f'<text x="{padding_left - 8}" y="{y_pos + 4:.1f}" fill="#64748b" font-size="10" text-anchor="end" font-family="JetBrains Mono, monospace">{val_label}</text>')

    # Year ticks
    year_ticks = []
    for idx, row in enumerate(yearly_schedule):
        if n > 12 and (idx + 1) % 2 != 0 and idx != n - 1:
            continue
        x_pos = padding_left + (idx / max(1, n - 1)) * plot_w
        year_ticks.append(f'<text x="{x_pos:.1f}" y="{base_y + 18}" fill="#94a3b8" font-size="11" text-anchor="middle" font-family="Inter, sans-serif">Y{row["year"]}</text>')

    return f"""<svg viewBox="0 0 {width} {height}" class="wealth-trajectory-svg" role="img" aria-label="Compounding Growth Trajectory Chart">
      <defs>
        <linearGradient id="gradFv_{mode}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="{color_fv}" stop-opacity="0.35"/>
          <stop offset="100%" stop-color="{color_fv}" stop-opacity="0.02"/>
        </linearGradient>
      </defs>
      { "".join(grid_lines) }
      <path d="{fv_area_d}" fill="url(#gradFv_{mode})" />
      <path d="{inv_path_d}" fill="none" stroke="{color_inv}" stroke-width="2" stroke-dasharray="4,4" />
      <path d="{fv_path_d}" fill="none" stroke="{color_fv}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" />
      { "".join(year_ticks) }
      <circle cx="{future_points[-1][0]:.1f}" cy="{future_points[-1][1]:.1f}" r="5" fill="{color_fv}" stroke="#ffffff" stroke-width="2" />
    </svg>"""


# ============================================================================
# 4. GOAL-BASED LUMP SUM PORTFOLIO ALLOCATOR
# ============================================================================

def generate_lump_sum_portfolio(
    capital_amount: float,
    horizon_years: float,
    risk_profile: str = "moderate",
    target_return_pct: float | None = None,
) -> dict[str, Any]:
    """Generates a customized lump-sum portfolio allocation based on capital, duration, and risk appetite."""
    capital = max(10000.0, float(capital_amount))
    horizon = max(0.5, float(horizon_years))
    risk = risk_profile.lower().strip()

    if horizon < 1.0:
        weights = {"liquid_arbitrage": 60.0, "gold_commodity": 20.0, "large_cap_index": 20.0, "flexi_cap": 0.0, "mid_small_cap": 0.0, "high_conviction_stocks": 0.0}
        expected_cagr = 8.5
        portfolio_max_dd = -6.0
        risk_label = "Ultra Low Risk / Capital Preservation"
    elif horizon < 3.0:
        if risk in ["conservative", "low"]:
            weights = {"liquid_arbitrage": 40.0, "gold_commodity": 25.0, "large_cap_index": 25.0, "flexi_cap": 10.0, "mid_small_cap": 0.0, "high_conviction_stocks": 0.0}
            expected_cagr = 10.5
            portfolio_max_dd = -10.0
            risk_label = "Conservative (Low Risk)"
        elif risk in ["aggressive", "high"]:
            weights = {"liquid_arbitrage": 10.0, "gold_commodity": 15.0, "large_cap_index": 35.0, "flexi_cap": 25.0, "mid_small_cap": 15.0, "high_conviction_stocks": 0.0}
            expected_cagr = 14.5
            portfolio_max_dd = -18.0
            risk_label = "Growth Focused (Moderately High Risk)"
        else:
            weights = {"liquid_arbitrage": 25.0, "gold_commodity": 20.0, "large_cap_index": 35.0, "flexi_cap": 20.0, "mid_small_cap": 0.0, "high_conviction_stocks": 0.0}
            expected_cagr = 12.2
            portfolio_max_dd = -13.5
            risk_label = "Balanced (Moderate Risk)"
    elif horizon < 5.0:
        if risk in ["conservative", "low"]:
            weights = {"liquid_arbitrage": 20.0, "gold_commodity": 20.0, "large_cap_index": 40.0, "flexi_cap": 20.0, "mid_small_cap": 0.0, "high_conviction_stocks": 0.0}
            expected_cagr = 12.0
            portfolio_max_dd = -14.0
            risk_label = "Conservative Wealth Builder"
        elif risk in ["aggressive", "high"]:
            weights = {"liquid_arbitrage": 5.0, "gold_commodity": 10.0, "large_cap_index": 25.0, "flexi_cap": 30.0, "mid_small_cap": 20.0, "high_conviction_stocks": 10.0}
            expected_cagr = 16.8
            portfolio_max_dd = -22.0
            risk_label = "Aggressive Alpha Compounder"
        else:
            weights = {"liquid_arbitrage": 10.0, "gold_commodity": 15.0, "large_cap_index": 35.0, "flexi_cap": 30.0, "mid_small_cap": 10.0, "high_conviction_stocks": 0.0}
            expected_cagr = 14.5
            portfolio_max_dd = -17.5
            risk_label = "Balanced Compounding"
    else:
        if risk in ["conservative", "low"]:
            weights = {"liquid_arbitrage": 10.0, "gold_commodity": 15.0, "large_cap_index": 45.0, "flexi_cap": 30.0, "mid_small_cap": 0.0, "high_conviction_stocks": 0.0}
            expected_cagr = 13.5
            portfolio_max_dd = -16.0
            risk_label = "Disciplined Long-Term Core"
        elif risk in ["aggressive", "high"]:
            weights = {"liquid_arbitrage": 0.0, "gold_commodity": 10.0, "large_cap_index": 20.0, "flexi_cap": 30.0, "mid_small_cap": 25.0, "high_conviction_stocks": 15.0}
            expected_cagr = 18.2
            portfolio_max_dd = -26.0
            risk_label = "High Growth / High Alpha Creation"
        else:
            weights = {"liquid_arbitrage": 5.0, "gold_commodity": 10.0, "large_cap_index": 35.0, "flexi_cap": 35.0, "mid_small_cap": 15.0, "high_conviction_stocks": 0.0}
            expected_cagr = 15.6
            portfolio_max_dd = -20.0
            risk_label = "Balanced Long-Term Compounder"

    assets: list[dict[str, Any]] = []

    if weights["large_cap_index"] > 0:
        pct = weights["large_cap_index"]
        val = capital * (pct / 100.0)
        assets.append({
            "asset_name": "Nippon India ETF Nifty 50 BeES",
            "symbol": "NIFTYBEES",
            "asset_type": "ETF",
            "category": "Large Cap Index (Top 50 Indian Leaders)",
            "allocation_pct": f"{pct:.1f}%",
            "allocation_pct_num": pct,
            "allocation_rupees": fmt_curr(val),
            "allocation_rupees_num": val,
            "cagr": "+13.5%",
            "sharpe": "1.32",
            "max_dd": "-24.0%",
            "rationale": "Core low-cost passive foundation tracking India's top 50 bluechips with lowest expense ratio.",
        })

    if weights["flexi_cap"] > 0:
        pct = weights["flexi_cap"]
        val = capital * (pct / 100.0)
        assets.append({
            "asset_name": "Parag Parikh Flexi Cap Fund - Direct Growth",
            "symbol": "MF_122639",
            "asset_type": "Mutual Fund",
            "category": "Flexi Cap (Multi-Cap & Global Value)",
            "allocation_pct": f"{pct:.1f}%",
            "allocation_pct_num": pct,
            "allocation_rupees": fmt_curr(val),
            "allocation_rupees_num": val,
            "cagr": "+21.4%",
            "sharpe": "1.45",
            "max_dd": "-19.2%",
            "rationale": "India's highest-rated risk-adjusted flexi-cap fund with value-oriented stock picking and cash buffer.",
        })

    if weights["mid_small_cap"] > 0:
        pct = weights["mid_small_cap"]
        val = capital * (pct / 100.0)
        assets.append({
            "asset_name": "Quant Small Cap Fund - Direct Growth",
            "symbol": "MF_120828",
            "asset_type": "Mutual Fund",
            "category": "Small Cap (High Alpha Generation)",
            "allocation_pct": f"{pct:.1f}%",
            "allocation_pct_num": pct,
            "allocation_rupees": fmt_curr(val),
            "allocation_rupees_num": val,
            "cagr": "+32.6%",
            "sharpe": "1.62",
            "max_dd": "-28.5%",
            "rationale": "Quantitative high-momentum strategy delivering exponential small-cap alpha during expansion phases.",
        })

    if weights["gold_commodity"] > 0:
        pct = weights["gold_commodity"]
        val = capital * (pct / 100.0)
        assets.append({
            "asset_name": "Nippon India ETF Gold BeES",
            "symbol": "GOLDBEES",
            "asset_type": "ETF",
            "category": "Commodity / Safe Haven Hedge",
            "allocation_pct": f"{pct:.1f}%",
            "allocation_pct_num": pct,
            "allocation_rupees": fmt_curr(val),
            "allocation_rupees_num": val,
            "cagr": "+14.8%",
            "sharpe": "1.20",
            "max_dd": "-12.5%",
            "rationale": "Uncorrelated hedge protecting portfolio purchasing power during market corrections and inflation.",
        })

    if weights["high_conviction_stocks"] > 0:
        pct = weights["high_conviction_stocks"]
        val = capital * (pct / 100.0)
        assets.append({
            "asset_name": "High-Conviction Stock Basket (Reliance / TCS / LT)",
            "symbol": "STOCK_BASKET",
            "asset_type": "Stock Basket",
            "category": "Direct Bluechip Growth",
            "allocation_pct": f"{pct:.1f}%",
            "allocation_pct_num": pct,
            "allocation_rupees": fmt_curr(val),
            "allocation_rupees_num": val,
            "cagr": "+18.0%",
            "sharpe": "1.35",
            "max_dd": "-26.0%",
            "rationale": "Tactical direct stock exposure to market leaders with dominant ROCE and structural earnings tailwinds.",
        })

    if weights["liquid_arbitrage"] > 0:
        pct = weights["liquid_arbitrage"]
        val = capital * (pct / 100.0)
        assets.append({
            "asset_name": "Kotak Arbitrage Fund / LiquidBeES",
            "symbol": "LIQUIDBEES",
            "asset_type": "Liquid / Arbitrage",
            "category": "Capital Preservation / Emergency Buffer",
            "allocation_pct": f"{pct:.1f}%",
            "allocation_pct_num": pct,
            "allocation_rupees": fmt_curr(val),
            "allocation_rupees_num": val,
            "cagr": "+6.8%",
            "sharpe": "2.40",
            "max_dd": "-0.5%",
            "rationale": "Virtually zero-risk arbitrage parking generating steady returns and liquidity to buy market dips.",
        })

    # Projected Future Wealth Compounding at Maturity
    conservative_cagr = max(5.0, expected_cagr - 3.5)
    expected_cagr_val = expected_cagr
    optimistic_cagr = expected_cagr + 4.0

    conservative_val = capital * ((1.0 + (conservative_cagr / 100.0)) ** horizon)
    expected_val = capital * ((1.0 + (expected_cagr_val / 100.0)) ** horizon)
    optimistic_val = capital * ((1.0 + (optimistic_cagr / 100.0)) ** horizon)

    expected_gain_rupees = expected_val - capital
    expected_gain_pct = (expected_gain_rupees / capital) * 100.0

    chart_svg = make_allocation_donut_svg(assets)

    return {
        "ok": True,
        "mode": "lump_sum",
        "capital_amount": fmt_curr(capital),
        "capital_amount_num": capital,
        "horizon_years": f"{horizon:.1f} Years",
        "horizon_years_num": horizon,
        "risk_profile": risk_label,
        "expected_cagr": f"{expected_cagr:.1f}%",
        "expected_cagr_num": expected_cagr,
        "portfolio_max_drawdown": f"{portfolio_max_dd:.1f}%",
        "expected_maturity_corpus": fmt_curr(expected_val),
        "expected_gain_rupees": fmt_curr(expected_gain_rupees),
        "expected_gain_pct": fmt_pct(expected_gain_pct),
        "wealth_projection": {
            "conservative": {"cagr": f"{conservative_cagr:.1f}%", "corpus": fmt_curr(conservative_val), "gain": fmt_curr(conservative_val - capital)},
            "expected": {"cagr": f"{expected_cagr_val:.1f}%", "corpus": fmt_curr(expected_val), "gain": fmt_curr(expected_gain_rupees)},
            "optimistic": {"cagr": f"{optimistic_cagr:.1f}%", "corpus": fmt_curr(optimistic_val), "gain": fmt_curr(optimistic_val - capital)},
        },
        "assets": assets,
        "chart_svg": chart_svg,
    }


# ============================================================================
# 5. SMART MULTI-FUND MONTHLY SIP BASKET BUILDER
# ============================================================================

def generate_sip_plan(
    monthly_amount: float,
    horizon_years: float = 5.0,
    risk_profile: str = "moderate",
) -> dict[str, Any]:
    """Generates an optimal monthly SIP basket splitting the monthly budget across 3–4 best-in-class funds."""
    sip_monthly = max(1000.0, float(monthly_amount))
    horizon = max(1.0, float(horizon_years))
    risk = risk_profile.lower().strip()
    total_installments = int(horizon * 12)
    total_invested = sip_monthly * total_installments

    if risk in ["conservative", "low"]:
        cagr = 12.5
        risk_title = "Conservative Wealth Accumulator (Low Drawdown)"
        basket = [
            {
                "scheme_name": "UTI Nifty 50 Index Fund - Direct Growth",
                "symbol": "MF_120716",
                "category": "Index Fund (Large Cap)",
                "monthly_split_pct": 50.0,
                "monthly_rupees": sip_monthly * 0.50,
                "cagr_3y": "+15.2%",
                "sharpe": "1.20",
                "role": "Core passive anchor in India's top 50 enterprises.",
            },
            {
                "scheme_name": "HDFC Balanced Advantage Fund - Direct Growth",
                "symbol": "MF_118989",
                "category": "Dynamic Asset Allocation / Hybrid",
                "monthly_split_pct": 30.0,
                "monthly_rupees": sip_monthly * 0.30,
                "cagr_3y": "+18.2%",
                "sharpe": "1.38",
                "role": "Automatically rebalances between equity and debt based on market valuation.",
            },
            {
                "scheme_name": "Nippon India ETF Gold BeES",
                "symbol": "GOLDBEES",
                "category": "Gold / Safe Haven",
                "monthly_split_pct": 20.0,
                "monthly_rupees": sip_monthly * 0.20,
                "cagr_3y": "+16.5%",
                "sharpe": "1.25",
                "role": "Hedges currency depreciation and protects against equity drawdowns.",
            },
        ]
    elif risk in ["aggressive", "high"]:
        cagr = 18.5
        risk_title = "High Alpha Growth Basket (Maximum Long-Term Wealth)"
        basket = [
            {
                "scheme_name": "Quant Small Cap Fund - Direct Growth",
                "symbol": "MF_120828",
                "category": "Small Cap Equity",
                "monthly_split_pct": 35.0,
                "monthly_rupees": sip_monthly * 0.35,
                "cagr_3y": "+32.6%",
                "sharpe": "1.62",
                "role": "High-velocity momentum fund capitalizing on mid/small cap expansions.",
            },
            {
                "scheme_name": "Parag Parikh Flexi Cap Fund - Direct Growth",
                "symbol": "MF_122639",
                "category": "Flexi Cap Equity",
                "monthly_split_pct": 35.0,
                "monthly_rupees": sip_monthly * 0.35,
                "cagr_3y": "+21.4%",
                "sharpe": "1.45",
                "role": "Consistently compounds capital across domestic and international value leaders.",
            },
            {
                "scheme_name": "Motilal Oswal Nasdaq 100 ETF",
                "symbol": "MON100",
                "category": "Global Tech / US Leaders",
                "monthly_split_pct": 20.0,
                "monthly_rupees": sip_monthly * 0.20,
                "cagr_3y": "+24.8%",
                "sharpe": "1.30",
                "role": "Direct USD asset compounding in global technology leaders (Apple, Nvidia, Microsoft).",
            },
            {
                "scheme_name": "Nippon India ETF Gold BeES",
                "symbol": "GOLDBEES",
                "category": "Gold Commodity",
                "monthly_split_pct": 10.0,
                "monthly_rupees": sip_monthly * 0.10,
                "cagr_3y": "+16.5%",
                "sharpe": "1.25",
                "role": "Portfolio hedge against sudden macro shocks.",
            },
        ]
    else:
        cagr = 15.2
        risk_title = "Balanced Wealth Creator (Optimal Risk-Adjusted Returns)"
        basket = [
            {
                "scheme_name": "Parag Parikh Flexi Cap Fund - Direct Growth",
                "symbol": "MF_122639",
                "category": "Flexi Cap Equity",
                "monthly_split_pct": 40.0,
                "monthly_rupees": sip_monthly * 0.40,
                "cagr_3y": "+21.4%",
                "sharpe": "1.45",
                "role": "Core multi-cap growth engine with low downside volatility.",
            },
            {
                "scheme_name": "Nippon India ETF Nifty 50 BeES",
                "symbol": "NIFTYBEES",
                "category": "Large Cap Index ETF",
                "monthly_split_pct": 30.0,
                "monthly_rupees": sip_monthly * 0.30,
                "cagr_3y": "+15.2%",
                "sharpe": "1.32",
                "role": "Steady compounding in top 50 bluechip companies.",
            },
            {
                "scheme_name": "Nippon India Small Cap Fund - Direct Growth",
                "symbol": "MF_118778",
                "category": "Small Cap Equity",
                "monthly_split_pct": 20.0,
                "monthly_rupees": sip_monthly * 0.20,
                "cagr_3y": "+28.5%",
                "sharpe": "1.50",
                "role": "High-growth kicker providing long-term alpha.",
            },
            {
                "scheme_name": "Nippon India ETF Gold BeES",
                "symbol": "GOLDBEES",
                "category": "Gold Hedge",
                "monthly_split_pct": 10.0,
                "monthly_rupees": sip_monthly * 0.10,
                "cagr_3y": "+16.5%",
                "sharpe": "1.25",
                "role": "Non-correlated defensive stabilizer.",
            },
        ]

    r_monthly = (cagr / 100.0) / 12.0
    fv_expected = sip_monthly * (((1.0 + r_monthly) ** total_installments - 1.0) / r_monthly) * (1.0 + r_monthly)
    r_conservative = ((cagr - 3.0) / 100.0) / 12.0
    fv_conservative = sip_monthly * (((1.0 + r_conservative) ** total_installments - 1.0) / r_conservative) * (1.0 + r_conservative)
    r_optimistic = ((cagr + 3.5) / 100.0) / 12.0
    fv_optimistic = sip_monthly * (((1.0 + r_optimistic) ** total_installments - 1.0) / r_optimistic) * (1.0 + r_optimistic)

    expected_gain = fv_expected - total_invested
    gain_pct = (expected_gain / total_invested) * 100.0

    return {
        "ok": True,
        "mode": "sip",
        "monthly_sip_amount": fmt_curr(sip_monthly),
        "monthly_sip_amount_num": sip_monthly,
        "horizon_years": f"{horizon:.1f} Years",
        "horizon_years_num": horizon,
        "risk_profile": risk_title,
        "total_invested": fmt_curr(total_invested),
        "total_invested_num": total_invested,
        "expected_cagr": f"{cagr:.1f}%",
        "expected_cagr_num": cagr,
        "projected_corpus": fmt_curr(fv_expected),
        "projected_corpus_num": fv_expected,
        "wealth_gain_rupees": fmt_curr(expected_gain),
        "wealth_gain_pct": fmt_pct(gain_pct),
        "projections": {
            "conservative": {"cagr": f"{cagr - 3.0:.1f}%", "corpus": fmt_curr(fv_conservative), "gain": fmt_curr(fv_conservative - total_invested)},
            "expected": {"cagr": f"{cagr:.1f}%", "corpus": fmt_curr(fv_expected), "gain": fmt_curr(expected_gain)},
            "optimistic": {"cagr": f"{cagr + 3.5:.1f}%", "corpus": fmt_curr(fv_optimistic), "gain": fmt_curr(fv_optimistic - total_invested)},
        },
        "basket": [
            {
                **item,
                "monthly_rupees_fmt": fmt_curr(item["monthly_rupees"]),
                "monthly_split_pct_fmt": f"{item['monthly_split_pct']:.0f}%",
            }
            for item in basket
        ],
    }


def make_allocation_donut_svg(assets: list[dict[str, Any]], size: int = 240) -> str:
    """Generates an SVG donut chart for visual asset allocation breakdown."""
    colors = ["#38bdf8", "#a78bfa", "#f59e0b", "#10b981", "#ec4899", "#64748b"]
    radius = 80
    center = size / 2
    stroke_width = 28
    circumference = 2 * math.pi * radius

    segments = []
    current_offset = 0.0

    for idx, item in enumerate(assets):
        pct = float(item.get("allocation_pct_num") or float(str(item["allocation_pct"]).replace("%", "")))
        dash = (pct / 100.0) * circumference
        color = colors[idx % len(colors)]
        
        segments.append(
            f'<circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke_width}" stroke-dasharray="{dash:.1f} {circumference - dash:.1f}" '
            f'stroke-dashoffset="{-current_offset:.1f}" transform="rotate(-90 {center} {center})" />'
        )
        current_offset += dash

    return f"""<svg viewBox="0 0 {size} {size}" class="allocation-donut-svg" role="img" aria-label="Asset Allocation Donut Chart">
      { "".join(segments) }
      <circle cx="{center}" cy="{center}" r="{radius - stroke_width/2}" fill="#141c2b" />
      <text x="{center}" y="{center - 4}" text-anchor="middle" fill="#94a3b8" font-size="11" font-weight="600" font-family="Inter, sans-serif">PORTFOLIO</text>
      <text x="{center}" y="{center + 14}" text-anchor="middle" fill="#f1f5f9" font-size="14" font-weight="800" font-family="JetBrains Mono, monospace">100%</text>
    </svg>"""
