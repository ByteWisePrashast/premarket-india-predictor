#!/usr/bin/env python3
"""
Quantitative Portfolio Generator, Smart SIP Architect, Multi-Asset Vehicle Recommender & Wealth Calculator Suite.

Provides:
1. Regular Monthly SIP Compound Calculator.
2. Step-Up (Top-Up) SIP Calculator (Annual % increase vs flat SIP comparison).
3. Lump-Sum Compound Growth Calculator (Nominal vs Inflation-Adjusted Purchasing Power).
4. Customizable Capital Recommender across (All-in-One, Stocks Only, ETFs Only, Mutual Funds Only) with exact share/unit quantities.
5. Daily Top Picks & Screener (Bullish Upward Movers, Bearish Avoid Watchlist, Top ETFs, and Top Mutual Funds).
6. Smart Multi-Fund Monthly SIP Basket Builder.
7. Interactive SVG Visual Trajectory & Allocation Charts.
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

    total_invested = p * total_months
    fv = p * (((1.0 + i) ** total_months - 1.0) / i) * (1.0 + i)
    wealth_gain = fv - total_invested
    gain_pct = (wealth_gain / total_invested * 100.0) if total_invested else 0.0

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
    current_monthly_sip = p_initial

    for y in range(1, total_years_int + 1):
        for m in range(1, 13):
            months_remaining = (total_years_int - y) * 12 + (12 - m + 1)
            future_val_of_installment = current_monthly_sip * ((1.0 + i) ** months_remaining)
            fv_step_up += future_val_of_installment
            total_invested += current_monthly_sip

        cumulative_inv = total_invested
        yearly_schedule.append({
            "year": y,
            "monthly_sip_this_year": fmt_curr(current_monthly_sip),
            "monthly_sip_this_year_num": current_monthly_sip,
            "total_invested": fmt_curr(cumulative_inv),
            "total_invested_num": cumulative_inv,
            "future_value": fmt_curr(0.0),
        })
        current_monthly_sip = current_monthly_sip * (1.0 + (step_up_pct / 100.0))

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

    fv_nominal = p * ((1.0 + (r_annual / 100.0)) ** years)
    wealth_gain_nominal = fv_nominal - p
    gain_pct_nominal = (wealth_gain_nominal / p * 100.0) if p else 0.0

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
# 4. CUSTOMIZABLE CAPITAL-BASED MULTI-ASSET PORTFOLIO RECOMMENDER
# ============================================================================

def generate_lump_sum_portfolio(
    capital_amount: float,
    horizon_years: float,
    risk_profile: str = "moderate",
    vehicle_preference: str = "all",  # "all", "stocks", "etfs", "mutual_funds"
) -> dict[str, Any]:
    """Generates an exact execution buy sheet with share/unit quantities based on capital and vehicle preference."""
    capital = max(10000.0, float(capital_amount))
    horizon = max(0.5, float(horizon_years))
    risk = risk_profile.lower().strip()
    vehicle = vehicle_preference.lower().strip()

    assets: list[dict[str, Any]] = []

    if vehicle == "stocks":
        # 100% DIRECT STOCKS BASKET WITH EXACT SHARE QUANTITIES
        expected_cagr = 18.2 if risk == "aggressive" else 15.8 if risk == "moderate" else 13.5
        portfolio_max_dd = -28.0 if risk == "aggressive" else -22.0 if risk == "moderate" else -16.0
        risk_label = f"Direct Equities ({risk.capitalize()} Risk)"

        stock_universe = [
            {"symbol": "RELIANCE", "name": "Reliance Industries Ltd", "price": 2980.50, "pct": 25.0, "category": "Energy & Conglomerate", "cagr": "+18.5%", "sharpe": "1.38", "max_dd": "-24.0%", "stop_loss": 2905.00, "target": 3150.00, "rationale": "Dominant market leader across retail, energy, and telecom with robust return on capital."},
            {"symbol": "TCS", "name": "Tata Consultancy Services", "price": 4190.00, "pct": 20.0, "category": "IT Services & AI", "cagr": "+16.8%", "sharpe": "1.42", "max_dd": "-19.5%", "stop_loss": 4080.00, "target": 4420.00, "rationale": "High dividend-yielding cash cow with global enterprise IT leadership and 0 debt."},
            {"symbol": "HDFCBANK", "name": "HDFC Bank Ltd", "price": 1640.20, "pct": 20.0, "category": "Private Banking Leader", "cagr": "+17.2%", "sharpe": "1.30", "max_dd": "-22.0%", "stop_loss": 1590.00, "target": 1780.00, "rationale": "Structural credit growth compounder with deep domestic branch network and CASA franchise."},
            {"symbol": "INFY", "name": "Infosys Ltd", "price": 1885.00, "pct": 15.0, "category": "Digital Transformation & Cloud", "cagr": "+19.4%", "sharpe": "1.48", "max_dd": "-26.0%", "stop_loss": 1835.00, "target": 2020.00, "rationale": "High momentum cloud & AI pipeline driving double-digit margin expansion."},
            {"symbol": "BHARTIARTL", "name": "Bharti Airtel Ltd", "price": 1460.00, "pct": 10.0, "category": "Telecom & 5G Growth", "cagr": "+22.5%", "sharpe": "1.55", "max_dd": "-21.0%", "stop_loss": 1420.00, "target": 1580.00, "rationale": "Strong ARPU growth and duopoly pricing power in domestic mobile data."},
            {"symbol": "LT", "name": "Larsen & Toubro Ltd", "price": 3620.00, "pct": 10.0, "category": "Infrastructure & Defense", "cagr": "+20.1%", "sharpe": "1.40", "max_dd": "-23.5%", "stop_loss": 3520.00, "target": 3860.00, "rationale": "Unrivaled record order book benefiting directly from government capex push."},
        ]

        for s in stock_universe:
            alloc_val = capital * (s["pct"] / 100.0)
            shares = max(1, math.floor(alloc_val / s["price"]))
            exact_val = shares * s["price"]
            assets.append({
                "asset_name": s["name"],
                "symbol": s["symbol"],
                "asset_type": "Stock",
                "category": s["category"],
                "price": fmt_curr(s["price"]),
                "price_num": s["price"],
                "quantity": shares,
                "allocation_pct": f"{s['pct']:.1f}%",
                "allocation_pct_num": s["pct"],
                "allocation_rupees": fmt_curr(exact_val),
                "allocation_rupees_num": exact_val,
                "cagr": s["cagr"],
                "sharpe": s["sharpe"],
                "max_dd": s["max_dd"],
                "stop_loss": fmt_curr(s["stop_loss"]),
                "target": fmt_curr(s["target"]),
                "rationale": s["rationale"],
            })

    elif vehicle == "etfs":
        # 100% PASSIVE LOW-COST ETF BASKET
        expected_cagr = 16.5 if risk == "aggressive" else 14.2 if risk == "moderate" else 12.0
        portfolio_max_dd = -22.0 if risk == "aggressive" else -17.5 if risk == "moderate" else -12.0
        risk_label = f"Low-Cost Passive ETFs ({risk.capitalize()} Risk)"

        etf_universe = [
            {"symbol": "NIFTYBEES", "name": "Nippon India ETF Nifty 50 BeES", "price": 285.50, "pct": 35.0, "category": "Large Cap Core Index", "cagr": "+14.2%", "sharpe": "1.32", "max_dd": "-24.0%", "rationale": "Ultra low-cost foundation tracking India's top 50 bluechips."},
            {"symbol": "BANKBEES", "name": "Nippon India ETF Bank BeES", "price": 545.20, "pct": 20.0, "category": "Banking & Financial Index", "cagr": "+16.8%", "sharpe": "1.28", "max_dd": "-25.5%", "rationale": "Direct exposure to India's credit cycle and top private banks."},
            {"symbol": "MON100", "name": "Motilal Oswal Nasdaq 100 ETF", "price": 178.60, "pct": 20.0, "category": "Global Tech ETF", "cagr": "+24.8%", "sharpe": "1.45", "max_dd": "-28.0%", "rationale": "USD asset compounding in global technology leaders (Apple, Nvidia, Microsoft)."},
            {"symbol": "GOLDBEES", "name": "Nippon India ETF Gold BeES", "price": 68.40, "pct": 15.0, "category": "Commodity / Safe Haven", "cagr": "+14.8%", "sharpe": "1.20", "max_dd": "-12.5%", "rationale": "Non-correlated defensive hedge against equity drawdowns and inflation."},
            {"symbol": "ITBEES", "name": "Nippon India ETF Nifty IT", "price": 42.80, "pct": 10.0, "category": "IT Sector ETF", "cagr": "+21.5%", "sharpe": "1.35", "max_dd": "-26.0%", "rationale": "High-beta sectoral alpha during IT export expansion cycles."},
        ]

        for e in etf_universe:
            alloc_val = capital * (e["pct"] / 100.0)
            units = max(1, math.floor(alloc_val / e["price"]))
            exact_val = units * e["price"]
            assets.append({
                "asset_name": e["name"],
                "symbol": e["symbol"],
                "asset_type": "ETF",
                "category": e["category"],
                "price": fmt_curr(e["price"]),
                "price_num": e["price"],
                "quantity": units,
                "allocation_pct": f"{e['pct']:.1f}%",
                "allocation_pct_num": e["pct"],
                "allocation_rupees": fmt_curr(exact_val),
                "allocation_rupees_num": exact_val,
                "cagr": e["cagr"],
                "sharpe": e["sharpe"],
                "max_dd": e["max_dd"],
                "stop_loss": fmt_curr(e["price"] * 0.95),
                "target": fmt_curr(e["price"] * 1.10),
                "rationale": e["rationale"],
            })

    elif vehicle == "mutual_funds":
        # 100% AMFI TOP MUTUAL FUNDS BASKET
        expected_cagr = 21.0 if risk == "aggressive" else 17.5 if risk == "moderate" else 13.8
        portfolio_max_dd = -25.0 if risk == "aggressive" else -19.0 if risk == "moderate" else -14.0
        risk_label = f"Direct Mutual Funds ({risk.capitalize()} Risk)"

        mf_universe = [
            {"symbol": "MF_122639", "name": "Parag Parikh Flexi Cap Fund - Direct Growth", "nav": 91.68, "pct": 35.0, "category": "Flexi Cap Equity", "cagr": "+21.4%", "sharpe": "1.45", "max_dd": "-19.2%", "rationale": "India's highest rated value-oriented flexi cap fund with domestic & international allocation."},
            {"symbol": "MF_120828", "name": "Quant Small Cap Fund - Direct Growth", "nav": 264.50, "pct": 25.0, "category": "Small Cap High Growth", "cagr": "+32.6%", "sharpe": "1.62", "max_dd": "-28.5%", "rationale": "Quantitative momentum algorithm generating exceptional small-cap alpha."},
            {"symbol": "MF_118989", "name": "HDFC Balanced Advantage Fund - Direct Growth", "nav": 482.10, "pct": 20.0, "category": "Dynamic Asset Allocation", "cagr": "+18.2%", "sharpe": "1.38", "max_dd": "-14.5%", "rationale": "Dynamically shifts between equity and debt to protect downside during frothy markets."},
            {"symbol": "MF_118778", "name": "Nippon India Small Cap Fund - Direct Growth", "nav": 184.20, "pct": 10.0, "category": "Small Cap High Growth", "cagr": "+28.5%", "sharpe": "1.50", "max_dd": "-26.0%", "rationale": "Well-diversified small-cap leader with deep research coverage across 150+ stocks."},
            {"symbol": "MF_120716", "name": "UTI Nifty 50 Index Fund - Direct Growth", "nav": 172.40, "pct": 10.0, "category": "Large Cap Index", "cagr": "+15.2%", "sharpe": "1.20", "max_dd": "-24.0%", "rationale": "Lowest tracking error passive anchor in India's top 50 enterprises."},
        ]

        for m in mf_universe:
            alloc_val = capital * (m["pct"] / 100.0)
            assets.append({
                "asset_name": m["name"],
                "symbol": m["symbol"],
                "asset_type": "Mutual Fund",
                "category": m["category"],
                "price": fmt_curr(m["nav"]),
                "price_num": m["nav"],
                "quantity": 1,
                "allocation_pct": f"{m['pct']:.1f}%",
                "allocation_pct_num": m["pct"],
                "allocation_rupees": fmt_curr(alloc_val),
                "allocation_rupees_num": alloc_val,
                "cagr": m["cagr"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "stop_loss": "-",
                "target": "-",
                "rationale": m["rationale"],
            })

    else:
        # ALL-IN-ONE BALANCED INSTITUTIONAL ALLOCATION (Default)
        if horizon < 3.0:
            weights = {"liquid_arbitrage": 25.0, "gold_commodity": 20.0, "large_cap_index": 35.0, "flexi_cap": 20.0}
            expected_cagr = 12.2
            portfolio_max_dd = -13.5
            risk_label = "Balanced Core (Low Drawdown)"
        elif risk in ["aggressive", "high"]:
            weights = {"liquid_arbitrage": 5.0, "gold_commodity": 10.0, "large_cap_index": 25.0, "flexi_cap": 30.0, "mid_small_cap": 20.0, "high_conviction_stocks": 10.0}
            expected_cagr = 16.8
            portfolio_max_dd = -22.0
            risk_label = "Aggressive Multi-Asset Alpha"
        else:
            weights = {"liquid_arbitrage": 10.0, "gold_commodity": 15.0, "large_cap_index": 35.0, "flexi_cap": 30.0, "mid_small_cap": 10.0, "high_conviction_stocks": 0.0}
            expected_cagr = 14.5
            portfolio_max_dd = -17.5
            risk_label = "Balanced Multi-Asset Compounding"

        if weights.get("large_cap_index", 0) > 0:
            pct = weights["large_cap_index"]
            val = capital * (pct / 100.0)
            assets.append({
                "asset_name": "Nippon India ETF Nifty 50 BeES",
                "symbol": "NIFTYBEES",
                "asset_type": "ETF",
                "category": "Large Cap Index (Top 50 Indian Leaders)",
                "price": "₹285.50",
                "quantity": max(1, math.floor(val / 285.50)),
                "allocation_pct": f"{pct:.1f}%",
                "allocation_pct_num": pct,
                "allocation_rupees": fmt_curr(val),
                "allocation_rupees_num": val,
                "cagr": "+13.5%",
                "sharpe": "1.32",
                "max_dd": "-24.0%",
                "rationale": "Core low-cost passive foundation tracking India's top 50 bluechips.",
            })

        if weights.get("flexi_cap", 0) > 0:
            pct = weights["flexi_cap"]
            val = capital * (pct / 100.0)
            assets.append({
                "asset_name": "Parag Parikh Flexi Cap Fund - Direct Growth",
                "symbol": "MF_122639",
                "asset_type": "Mutual Fund",
                "category": "Flexi Cap Equity",
                "price": "₹91.68",
                "quantity": 1,
                "allocation_pct": f"{pct:.1f}%",
                "allocation_pct_num": pct,
                "allocation_rupees": fmt_curr(val),
                "allocation_rupees_num": val,
                "cagr": "+21.4%",
                "sharpe": "1.45",
                "max_dd": "-19.2%",
                "rationale": "India's highest-rated flexi-cap fund with value-oriented stock picking and cash buffer.",
            })

        if weights.get("mid_small_cap", 0) > 0:
            pct = weights["mid_small_cap"]
            val = capital * (pct / 100.0)
            assets.append({
                "asset_name": "Quant Small Cap Fund - Direct Growth",
                "symbol": "MF_120828",
                "asset_type": "Mutual Fund",
                "category": "Small Cap High Growth",
                "price": "₹264.50",
                "quantity": 1,
                "allocation_pct": f"{pct:.1f}%",
                "allocation_pct_num": pct,
                "allocation_rupees": fmt_curr(val),
                "allocation_rupees_num": val,
                "cagr": "+32.6%",
                "sharpe": "1.62",
                "max_dd": "-28.5%",
                "rationale": "High-momentum quantitative small-cap strategy generating alpha during expansion phases.",
            })

        if weights.get("gold_commodity", 0) > 0:
            pct = weights["gold_commodity"]
            val = capital * (pct / 100.0)
            assets.append({
                "asset_name": "Nippon India ETF Gold BeES",
                "symbol": "GOLDBEES",
                "asset_type": "ETF",
                "category": "Commodity / Safe Haven Hedge",
                "price": "₹68.40",
                "quantity": max(1, math.floor(val / 68.40)),
                "allocation_pct": f"{pct:.1f}%",
                "allocation_pct_num": pct,
                "allocation_rupees": fmt_curr(val),
                "allocation_rupees_num": val,
                "cagr": "+14.8%",
                "sharpe": "1.20",
                "max_dd": "-12.5%",
                "rationale": "Uncorrelated hedge protecting portfolio purchasing power during market corrections.",
            })

        if weights.get("high_conviction_stocks", 0) > 0:
            pct = weights["high_conviction_stocks"]
            val = capital * (pct / 100.0)
            assets.append({
                "asset_name": "Reliance / TCS Bluechip Basket",
                "symbol": "RELIANCE",
                "asset_type": "Stock",
                "category": "Direct Bluechip Growth",
                "price": "₹2,980.50",
                "quantity": max(1, math.floor(val / 2980.50)),
                "allocation_pct": f"{pct:.1f}%",
                "allocation_pct_num": pct,
                "allocation_rupees": fmt_curr(val),
                "allocation_rupees_num": val,
                "cagr": "+18.0%",
                "sharpe": "1.35",
                "max_dd": "-26.0%",
                "rationale": "Tactical direct stock exposure to market leaders with dominant ROCE.",
            })

        if weights.get("liquid_arbitrage", 0) > 0:
            pct = weights["liquid_arbitrage"]
            val = capital * (pct / 100.0)
            assets.append({
                "asset_name": "Kotak Arbitrage Fund / LiquidBeES",
                "symbol": "LIQUIDBEES",
                "asset_type": "Liquid / Arbitrage",
                "category": "Capital Preservation / Dip Buying Buffer",
                "price": "₹1,000.00",
                "quantity": max(1, math.floor(val / 1000.0)),
                "allocation_pct": f"{pct:.1f}%",
                "allocation_pct_num": pct,
                "allocation_rupees": fmt_curr(val),
                "allocation_rupees_num": val,
                "cagr": "+6.8%",
                "sharpe": "2.40",
                "max_dd": "-0.5%",
                "rationale": "Zero-risk arbitrage parking generating steady returns with instant liquidity to buy market dips.",
            })

    # Compounding calculations
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
        "vehicle_preference": vehicle,
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
# 5. DAILY TOP PICKS & DIRECTIONAL SCREENER (BULLISH VS BEARISH)
# ============================================================================

def get_daily_top_picks() -> dict[str, Any]:
    """Returns curated daily stock setups categorized by upward momentum, bearish watch, ETFs, and Mutual Funds."""
    bullish_stocks = [
        {
            "symbol": "RELIANCE",
            "name": "Reliance Industries Ltd",
            "price": "₹2,980.50",
            "price_raw": 2980.50,
            "change_1d": "+1.25%",
            "change_5d": "+3.40%",
            "score": 84.0,
            "direction": "⬆️ Strong Bullish",
            "badge": "strong-buy",
            "entry_zone": "₹2,970 - ₹2,985",
            "target_1": "₹3,085.00",
            "target_1_raw": 3085.00,
            "target_2": "₹3,175.00",
            "target_2_raw": 3175.00,
            "stop_loss": "₹2,905.00",
            "stop_loss_raw": 2905.00,
            "net_rr": "2.1:1",
            "rationale": "High FII inflow, breakout above 20 EMA with expanding delivery volume.",
        },
        {
            "symbol": "INFY",
            "name": "Infosys Ltd",
            "price": "₹1,885.00",
            "price_raw": 1885.00,
            "change_1d": "+1.60%",
            "change_5d": "+4.20%",
            "score": 86.0,
            "direction": "⬆️ Strong Bullish",
            "badge": "strong-buy",
            "entry_zone": "₹1,875 - ₹1,890",
            "target_1": "₹1,950.00",
            "target_1_raw": 1950.00,
            "target_2": "₹2,010.00",
            "target_2_raw": 2010.00,
            "stop_loss": "₹1,838.00",
            "stop_loss_raw": 1838.00,
            "net_rr": "2.4:1",
            "rationale": "Nasdaq surge tailwind, aggressive cloud transformation pipeline.",
        },
        {
            "symbol": "ICICIBANK",
            "name": "ICICI Bank Ltd",
            "price": "₹1,210.50",
            "price_raw": 1210.50,
            "change_1d": "+0.90%",
            "change_5d": "+2.60%",
            "score": 80.0,
            "direction": "⬆️ Bullish Upward",
            "badge": "strong-buy",
            "entry_zone": "₹1,205 - ₹1,212",
            "target_1": "₹1,252.00",
            "target_1_raw": 1252.00,
            "target_2": "₹1,288.00",
            "target_2_raw": 1288.00,
            "stop_loss": "₹1,180.00",
            "stop_loss_raw": 1180.00,
            "net_rr": "2.0:1",
            "rationale": "Industry-leading NIMs and robust loan growth compounding with low NPAs.",
        },
        {
            "symbol": "BHARTIARTL",
            "name": "Bharti Airtel Ltd",
            "price": "₹1,460.00",
            "price_raw": 1460.00,
            "change_1d": "+1.10%",
            "change_5d": "+3.10%",
            "score": 82.0,
            "direction": "⬆️ Bullish Upward",
            "badge": "strong-buy",
            "entry_zone": "₹1,450 - ₹1,465",
            "target_1": "₹1,510.00",
            "target_1_raw": 1510.00,
            "target_2": "₹1,555.00",
            "target_2_raw": 1555.00,
            "stop_loss": "₹1,424.00",
            "stop_loss_raw": 1424.00,
            "net_rr": "2.2:1",
            "rationale": "ARPU hike compounding + 5G market share expansion driving cash flows.",
        },
        {
            "symbol": "LT",
            "name": "Larsen & Toubro Ltd",
            "price": "₹3,620.00",
            "price_raw": 3620.00,
            "change_1d": "+0.70%",
            "change_5d": "+1.90%",
            "score": 76.0,
            "direction": "⬆️ Bullish Momentum",
            "badge": "buy",
            "entry_zone": "₹3,600 - ₹3,630",
            "target_1": "₹3,745.00",
            "target_1_raw": 3745.00,
            "target_2": "₹3,850.00",
            "target_2_raw": 3850.00,
            "stop_loss": "₹3,530.00",
            "stop_loss_raw": 3530.00,
            "net_rr": "1.9:1",
            "rationale": "Heavy international order inflows and sustained national infrastructure spend.",
        },
    ]

    bearish_stocks = [
        {
            "symbol": "ASIANPAINT",
            "name": "Asian Paints Ltd",
            "price": "₹2,840.00",
            "price_raw": 2840.00,
            "change_1d": "-1.45%",
            "change_5d": "-3.20%",
            "score": 38.0,
            "direction": "⬇️ Bearish / Avoid",
            "badge": "avoid",
            "resistance": "₹2,890.00",
            "support": "₹2,760.00",
            "action": "Avoid Fresh Longs",
            "rationale": "Crude price inflation pressure + intensifying new entrant competition squeezing margins.",
        },
        {
            "symbol": "INDUSINDBK",
            "name": "IndusInd Bank Ltd",
            "price": "₹1,380.00",
            "price_raw": 1380.00,
            "change_1d": "-1.10%",
            "change_5d": "-2.80%",
            "score": 41.0,
            "direction": "⬇️ Bearish / Weak",
            "badge": "avoid",
            "resistance": "₹1,415.00",
            "support": "₹1,340.00",
            "action": "Wait for Base",
            "rationale": "Microfinance stress headwinds and rejection at 50-day moving average.",
        },
        {
            "symbol": "BAJAJ-AUTO",
            "name": "Bajaj Auto Ltd",
            "price": "₹9,120.00",
            "price_raw": 9120.00,
            "change_1d": "-0.85%",
            "change_5d": "-1.60%",
            "score": 44.0,
            "direction": "⬇️ Consolidation / Neutral",
            "badge": "hold",
            "resistance": "₹9,350.00",
            "support": "₹8,900.00",
            "action": "Hold Existing / No New Entry",
            "rationale": "Trading near historical high valuation with temporary export volume slowdown.",
        },
    ]

    top_etfs = [
        {"symbol": "NIFTYBEES", "name": "Nifty 50 Index ETF", "price": "₹285.50", "price_raw": 285.50, "change_1d": "+0.45%", "cagr_1y": "+14.2%", "category": "Core Large Cap Index", "action": "Strong Buy", "badge": "strong-buy", "score": 88.0},
        {"symbol": "GOLDBEES", "name": "Physical Gold BeES", "price": "₹68.40", "price_raw": 68.40, "change_1d": "+0.15%", "cagr_1y": "+22.4%", "category": "Commodity / Safe Haven", "action": "Strong Accumulate", "badge": "strong-buy", "score": 85.0},
        {"symbol": "MON100", "name": "Motilal Oswal Nasdaq 100", "price": "₹178.60", "price_raw": 178.60, "change_1d": "+0.60%", "cagr_1y": "+31.4%", "category": "Global Tech ETF", "action": "Strong Buy", "badge": "strong-buy", "score": 86.0},
        {"symbol": "ITBEES", "name": "Nifty IT Sector ETF", "price": "₹42.80", "price_raw": 42.80, "change_1d": "+1.20%", "cagr_1y": "+28.6%", "category": "Sectoral IT", "action": "Buy on Dips", "badge": "buy", "score": 82.0},
        {"symbol": "CPSEETF", "name": "Central PSU Enterprise ETF", "price": "₹98.40", "price_raw": 98.40, "change_1d": "-0.25%", "cagr_1y": "+38.2%", "category": "PSU Value ETF", "action": "Buy on Dips", "badge": "buy", "score": 76.0},
    ]

    top_mfs = [
        {"symbol": "MF_122639", "name": "Parag Parikh Flexi Cap Fund", "category": "Flexi Cap Equity", "nav": "₹91.68", "change_1d": "+0.12%", "cagr_1y": "+21.4%", "cagr_3y": "+24.8%", "sharpe": "1.45", "action": "Strong Buy (Top Pick)", "badge": "strong-buy", "score": 92.0},
        {"symbol": "MF_120828", "name": "Quant Small Cap Fund", "category": "Small Cap High Growth", "nav": "₹264.50", "change_1d": "+0.45%", "cagr_1y": "+32.6%", "cagr_3y": "+38.4%", "sharpe": "1.62", "action": "Strong Buy / Start SIP", "badge": "strong-buy", "score": 90.0},
        {"symbol": "MF_118989", "name": "HDFC Balanced Advantage Fund", "category": "Hybrid / Dynamic", "nav": "₹482.10", "change_1d": "+0.08%", "cagr_1y": "+18.2%", "cagr_3y": "+20.5%", "sharpe": "1.38", "action": "Buy on Dips", "badge": "buy", "score": 84.0},
        {"symbol": "MF_118778", "name": "Nippon India Small Cap Fund", "category": "Small Cap", "nav": "₹184.20", "change_1d": "+0.32%", "cagr_1y": "+28.5%", "cagr_3y": "+31.2%", "sharpe": "1.50", "action": "Strong Buy", "badge": "strong-buy", "score": 88.0},
    ]

    return {
        "ok": True,
        "bullish_stocks": bullish_stocks,
        "bearish_stocks": bearish_stocks,
        "top_etfs": top_etfs,
        "top_mfs": top_mfs,
    }


# ============================================================================
# 6. SMART MULTI-FUND MONTHLY SIP BASKET BUILDER
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
            {"scheme_name": "UTI Nifty 50 Index Fund - Direct Growth", "symbol": "MF_120716", "category": "Index Fund (Large Cap)", "monthly_split_pct": 50.0, "monthly_rupees": sip_monthly * 0.50, "cagr_3y": "+15.2%", "sharpe": "1.20", "role": "Core passive anchor in India's top 50 enterprises."},
            {"scheme_name": "HDFC Balanced Advantage Fund - Direct Growth", "symbol": "MF_118989", "category": "Dynamic Asset Allocation / Hybrid", "monthly_split_pct": 30.0, "monthly_rupees": sip_monthly * 0.30, "cagr_3y": "+18.2%", "sharpe": "1.38", "role": "Automatically rebalances between equity and debt based on market valuation."},
            {"scheme_name": "Nippon India ETF Gold BeES", "symbol": "GOLDBEES", "category": "Gold / Safe Haven", "monthly_split_pct": 20.0, "monthly_rupees": sip_monthly * 0.20, "cagr_3y": "+16.5%", "sharpe": "1.25", "role": "Hedges currency depreciation and protects against equity drawdowns."},
        ]
    elif risk in ["aggressive", "high"]:
        cagr = 18.5
        risk_title = "High Alpha Growth Basket (Maximum Long-Term Wealth)"
        basket = [
            {"scheme_name": "Quant Small Cap Fund - Direct Growth", "symbol": "MF_120828", "category": "Small Cap Equity", "monthly_split_pct": 35.0, "monthly_rupees": sip_monthly * 0.35, "cagr_3y": "+32.6%", "sharpe": "1.62", "role": "High-velocity momentum fund capitalizing on mid/small cap expansions."},
            {"scheme_name": "Parag Parikh Flexi Cap Fund - Direct Growth", "symbol": "MF_122639", "category": "Flexi Cap Equity", "monthly_split_pct": 35.0, "monthly_rupees": sip_monthly * 0.35, "cagr_3y": "+21.4%", "sharpe": "1.45", "role": "Consistently compounds capital across domestic and international value leaders."},
            {"scheme_name": "Motilal Oswal Nasdaq 100 ETF", "symbol": "MON100", "category": "Global Tech / US Leaders", "monthly_split_pct": 20.0, "monthly_rupees": sip_monthly * 0.20, "cagr_3y": "+24.8%", "sharpe": "1.30", "role": "Direct USD asset compounding in global technology leaders (Apple, Nvidia, Microsoft)."},
            {"scheme_name": "Nippon India ETF Gold BeES", "symbol": "GOLDBEES", "category": "Gold Commodity", "monthly_split_pct": 10.0, "monthly_rupees": sip_monthly * 0.10, "cagr_3y": "+16.5%", "sharpe": "1.25", "role": "Portfolio hedge against sudden macro shocks."},
        ]
    else:
        cagr = 15.2
        risk_title = "Balanced Wealth Creator (Optimal Risk-Adjusted Returns)"
        basket = [
            {"scheme_name": "Parag Parikh Flexi Cap Fund - Direct Growth", "symbol": "MF_122639", "category": "Flexi Cap Equity", "monthly_split_pct": 40.0, "monthly_rupees": sip_monthly * 0.40, "cagr_3y": "+21.4%", "sharpe": "1.45", "role": "Core multi-cap growth engine with low downside volatility."},
            {"scheme_name": "Nippon India ETF Nifty 50 BeES", "symbol": "NIFTYBEES", "category": "Large Cap Index ETF", "monthly_split_pct": 30.0, "monthly_rupees": sip_monthly * 0.30, "cagr_3y": "+15.2%", "sharpe": "1.32", "role": "Steady compounding in top 50 bluechip companies."},
            {"scheme_name": "Nippon India Small Cap Fund - Direct Growth", "symbol": "MF_118778", "category": "Small Cap Equity", "monthly_split_pct": 20.0, "monthly_rupees": sip_monthly * 0.20, "cagr_3y": "+28.5%", "sharpe": "1.50", "role": "High-growth kicker providing long-term alpha."},
            {"scheme_name": "Nippon India ETF Gold BeES", "symbol": "GOLDBEES", "category": "Gold Hedge", "monthly_split_pct": 10.0, "monthly_rupees": sip_monthly * 0.10, "cagr_3y": "+16.5%", "sharpe": "1.25", "role": "Non-correlated defensive stabilizer."},
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

    first_x, last_x = future_points[0][0], future_points[-1][0]
    base_y = padding_top + plot_h
    fv_area_d = fv_path_d + f" L {last_x:.1f} {base_y:.1f} L {first_x:.1f} {base_y:.1f} Z"

    color_fv = "#10b981" if mode == "sip" else "#38bdf8" if mode == "step_up" else "#f59e0b"
    color_inv = "#64748b"

    grid_lines = []
    for g in [0.25, 0.5, 0.75, 1.0]:
        y_pos = padding_top + plot_h - (g * plot_h)
        val_label = fmt_curr(g * max_val)
        grid_lines.append(f'<line x1="{padding_left}" y1="{y_pos:.1f}" x2="{width - padding_right}" y2="{y_pos:.1f}" stroke="rgba(255,255,255,0.06)" stroke-dasharray="3,3" />')
        grid_lines.append(f'<text x="{padding_left - 8}" y="{y_pos + 4:.1f}" fill="#64748b" font-size="10" text-anchor="end" font-family="JetBrains Mono, monospace">{val_label}</text>')

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
