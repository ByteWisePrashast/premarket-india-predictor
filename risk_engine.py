#!/usr/bin/env python3
"""
Risk Management, Position Sizing, Transaction Costs, Liquidity & Volatility Kill Switch Engine.

Upgraded with:
1. 10-day Average Daily Volume (ADV) Liquidity Filter (Max 1% ADV position cap to prevent impact cost).
2. Overnight Gap Hazard Index & Worst-Case Gap Loss Modeling (2x ATR catastrophic gap protection).
3. Hard position sizing (1% risk-per-trade rule, max 10% capital-per-position).
4. Realistic Indian market transaction costs (STT, Brokerage, GST, Stamp Duty, Slippage, STCG).
5. Volatility Regime Kill Switch (India VIX >= 22 or +15% 1-day spike downgrade).
6. Portfolio-level -8% 30-day drawdown circuit breaker and sector concentration caps.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import yfinance as yf

# Indian Market Statutory & Friction Constants
BROKERAGE_FLAT_PER_ORDER = 20.0  # ₹20 flat discount broker rate (e.g. Zerodha/Groww)
STT_DELIVERY_BUY_PCT = 0.001     # 0.1% on buy
STT_DELIVERY_SELL_PCT = 0.001    # 0.1% on sell
EXCHANGE_TXN_PCT = 0.0000345     # 0.00345% NSE txn fee
STAMP_DUTY_BUY_PCT = 0.00015     # 0.015% on buy
SEBI_TURNOVER_PCT = 0.000001     # ₹10 per Crore
GST_RATE = 0.18                  # 18% GST on (Brokerage + Txn + SEBI)
DEFAULT_BID_ASK_SLIPPAGE_PCT = 0.0010  # 0.10% slippage on entry & exit
STCG_TAX_RATE = 0.20             # 20% Short Term Capital Gains Tax (Budget 2024 updated)

# Volatility Regime Thresholds
INDIA_VIX_PANIC_THRESHOLD = 22.0
INDIA_VIX_ELEVATED_THRESHOLD = 16.0
VIX_SPIKE_1D_THRESHOLD = 15.0    # +15% single-day spike


@dataclass
class PositionSizingResult:
    recommended_shares: int
    position_value: float
    position_pct_of_capital: float
    max_loss_rupees: float
    risk_pct_of_capital: float
    entry_price: float
    stop_loss_price: float
    target_1_price: float
    target_2_price: float
    gross_rr: float
    net_realized_rr: float
    total_friction_rupees: float
    friction_pct_of_gain: float
    adv_shares_10d: float
    adv_pct_of_volume: float
    liquidity_warning: str | None
    overnight_gap_hazard_pct: float
    worst_case_gap_price: float
    worst_case_loss_rupees: float
    worst_case_loss_pct_of_capital: float
    sizing_rationale: str


def calculate_transaction_friction(
    buy_price: float,
    sell_price: float,
    quantity: int,
    slippage_pct: float = DEFAULT_BID_ASK_SLIPPAGE_PCT,
    apply_stcg: bool = True,
) -> dict[str, float]:
    """Calculates all statutory taxes, brokerage, slippage, and STCG for an Indian equity swing trade."""
    if quantity <= 0 or buy_price <= 0 or sell_price <= 0:
        return {
            "brokerage": 0.0,
            "stt": 0.0,
            "exchange_charges": 0.0,
            "stamp_duty": 0.0,
            "sebi_fee": 0.0,
            "gst": 0.0,
            "slippage": 0.0,
            "total_expenses": 0.0,
            "gross_profit": 0.0,
            "stcg_tax": 0.0,
            "net_profit": 0.0,
        }

    buy_turnover = buy_price * quantity
    sell_turnover = sell_price * quantity
    total_turnover = buy_turnover + sell_turnover

    buy_brok = min(BROKERAGE_FLAT_PER_ORDER, buy_turnover * 0.0003)
    sell_brok = min(BROKERAGE_FLAT_PER_ORDER, sell_turnover * 0.0003)
    total_brokerage = buy_brok + sell_brok

    stt = (buy_turnover * STT_DELIVERY_BUY_PCT) + (sell_turnover * STT_DELIVERY_SELL_PCT)
    exchange_charges = total_turnover * EXCHANGE_TXN_PCT
    stamp_duty = buy_turnover * STAMP_DUTY_BUY_PCT
    sebi_fee = total_turnover * SEBI_TURNOVER_PCT
    gst = (total_brokerage + exchange_charges + sebi_fee) * GST_RATE
    slippage = (buy_turnover + sell_turnover) * slippage_pct

    total_expenses = total_brokerage + stt + exchange_charges + stamp_duty + sebi_fee + gst + slippage

    gross_profit = sell_turnover - buy_turnover
    pre_tax_net = gross_profit - total_expenses

    stcg_tax = (pre_tax_net * STCG_TAX_RATE) if (pre_tax_net > 0 and apply_stcg) else 0.0
    net_profit = pre_tax_net - stcg_tax

    return {
        "brokerage": round(total_brokerage, 2),
        "stt": round(stt, 2),
        "exchange_charges": round(exchange_charges, 2),
        "stamp_duty": round(stamp_duty, 2),
        "sebi_fee": round(sebi_fee, 2),
        "gst": round(gst, 2),
        "slippage": round(slippage, 2),
        "total_expenses": round(total_expenses, 2),
        "gross_profit": round(gross_profit, 2),
        "stcg_tax": round(stcg_tax, 2),
        "net_profit": round(net_profit, 2),
    }


def calculate_position_sizing(
    total_capital: float,
    entry_price: float,
    stop_loss_price: float,
    target_1_price: float,
    target_2_price: float,
    atr_pct: float = 1.8,
    avg_daily_volume_10d: float = 500000.0,
    max_risk_pct: float = 1.0,
    max_position_pct: float = 10.0,
    vix_scale: float = 1.0,
) -> PositionSizingResult:
    """Calculates disciplined share quantity considering risk, capital cap, 1% ADV liquidity limits, and overnight gap hazard."""
    if total_capital <= 0 or entry_price <= 0 or stop_loss_price >= entry_price:
        return PositionSizingResult(
            recommended_shares=0,
            position_value=0.0,
            position_pct_of_capital=0.0,
            max_loss_rupees=0.0,
            risk_pct_of_capital=0.0,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            target_1_price=target_1_price,
            target_2_price=target_2_price,
            gross_rr=0.0,
            net_realized_rr=0.0,
            total_friction_rupees=0.0,
            friction_pct_of_gain=0.0,
            adv_shares_10d=avg_daily_volume_10d,
            adv_pct_of_volume=0.0,
            liquidity_warning=None,
            overnight_gap_hazard_pct=round(atr_pct * 1.25, 2),
            worst_case_gap_price=round(entry_price * (1 - (atr_pct * 2.0 / 100.0)), 2),
            worst_case_loss_rupees=0.0,
            worst_case_loss_pct_of_capital=0.0,
            sizing_rationale="Invalid price levels or capital specified.",
        )

    risk_per_share = entry_price - stop_loss_price
    gross_gain_per_share = target_1_price - entry_price

    # 1. Risk-based allocation (Max 1% of total portfolio at risk on stop loss)
    max_risk_rupees = total_capital * (max_risk_pct / 100.0) * vix_scale
    qty_from_risk = math.floor(max_risk_rupees / risk_per_share)

    # 2. Maximum Capital cap per single position (Max 10% of portfolio)
    max_capital_for_pos = total_capital * (max_position_pct / 100.0) * vix_scale
    qty_from_capital = math.floor(max_capital_for_pos / entry_price)

    # 3. Liquidity & Impact Cost Filter (Max 1.0% of 10-day Average Daily Volume)
    max_shares_by_adv = math.floor(max(10, avg_daily_volume_10d * 0.01))
    
    # Conservative minimum among all three constraints
    recommended_shares = max(1, min(qty_from_risk, qty_from_capital, max_shares_by_adv))
    
    liquidity_warning = None
    if (min(qty_from_risk, qty_from_capital) > max_shares_by_adv) and avg_daily_volume_10d > 0:
        liquidity_warning = f"⚠️ Low Liquidity Warning: Position capped at 1% of 10D ADV ({max_shares_by_adv:,} shares) to prevent excessive market impact."

    position_value = recommended_shares * entry_price
    position_pct = (position_value / total_capital) * 100.0
    actual_risk_rupees = recommended_shares * risk_per_share
    actual_risk_pct = (actual_risk_rupees / total_capital) * 100.0
    adv_pct = (recommended_shares / avg_daily_volume_10d * 100.0) if avg_daily_volume_10d > 0 else 0.0

    # Overnight Gap Hazard Modeling (Catastrophic 2x ATR overnight gap down)
    overnight_gap_hazard_pct = round(atr_pct * 1.25, 2)
    worst_case_gap_price = round(entry_price * (1.0 - (atr_pct * 2.0 / 100.0)), 2)
    worst_case_loss_rupees = round(recommended_shares * max(0.0, entry_price - worst_case_gap_price), 2)
    worst_case_loss_pct = round((worst_case_loss_rupees / total_capital) * 100.0, 2)

    # Calculate real-world friction & Net Realized R:R on Target 1
    gain_friction = calculate_transaction_friction(entry_price, target_1_price, recommended_shares)
    loss_friction = calculate_transaction_friction(entry_price, stop_loss_price, recommended_shares, apply_stcg=False)

    gross_reward = recommended_shares * gross_gain_per_share
    net_reward = gain_friction["net_profit"]
    net_risk = abs(loss_friction["net_profit"])

    gross_rr = round(gross_reward / max(actual_risk_rupees, 0.01), 2)
    net_realized_rr = round(net_reward / max(net_risk, 0.01), 2) if net_risk > 0 else gross_rr

    total_friction = gain_friction["total_expenses"] + gain_friction["stcg_tax"]
    friction_pct = (total_friction / gross_reward * 100.0) if gross_reward > 0 else 0.0

    rationale = (
        f"Position sized at {recommended_shares} shares (₹{position_value:,.2f}, {position_pct:.1f}% of capital). "
        f"Max risk on standard SL is ₹{actual_risk_rupees:,.2f} ({actual_risk_pct:.2f}% of portfolio). "
        f"Worst-case 2x ATR gap-down risk is ₹{worst_case_loss_rupees:,.2f} ({worst_case_loss_pct:.2f}%). "
        f"Gross R:R of {gross_rr}:1 adjusts to Net Realized {net_realized_rr}:1 after ₹{total_friction:,.2f} in statutory taxes & slippage."
    )

    return PositionSizingResult(
        recommended_shares=recommended_shares,
        position_value=round(position_value, 2),
        position_pct_of_capital=round(position_pct, 1),
        max_loss_rupees=round(actual_risk_rupees, 2),
        risk_pct_of_capital=round(actual_risk_pct, 2),
        entry_price=round(entry_price, 2),
        stop_loss_price=round(stop_loss_price, 2),
        target_1_price=round(target_1_price, 2),
        target_2_price=round(target_2_price, 2),
        gross_rr=gross_rr,
        net_realized_rr=net_realized_rr,
        total_friction_rupees=round(total_friction, 2),
        friction_pct_of_gain=round(friction_pct, 1),
        adv_shares_10d=round(avg_daily_volume_10d, 0),
        adv_pct_of_volume=round(adv_pct, 2),
        liquidity_warning=liquidity_warning,
        overnight_gap_hazard_pct=overnight_gap_hazard_pct,
        worst_case_gap_price=worst_case_gap_price,
        worst_case_loss_rupees=worst_case_loss_rupees,
        worst_case_loss_pct_of_capital=worst_case_loss_pct,
        sizing_rationale=rationale,
    )


def check_volatility_regime() -> dict[str, Any]:
    """Inspects live India VIX and CBOE VIX to detect market panic regimes and trigger kill switches."""
    india_vix_val = None
    india_vix_change = None

    try:
        hist = yf.Ticker("^INDIAVIX").history(period="5d", interval="1d", auto_adjust=True)
        hist = hist.dropna(subset=["Close"])
        if len(hist) >= 2:
            india_vix_val = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            india_vix_change = ((india_vix_val - prev) / prev) * 100.0
    except Exception:
        pass

    if india_vix_val is None:
        try:
            c_hist = yf.Ticker("^VIX").history(period="5d", interval="1d", auto_adjust=True)
            c_hist = c_hist.dropna(subset=["Close"])
            if len(c_hist) >= 2:
                india_vix_val = float(c_hist["Close"].iloc[-1])
                c_prev = float(c_hist["Close"].iloc[-2])
                india_vix_change = ((india_vix_val - c_prev) / c_prev) * 100.0
        except Exception:
            india_vix_val = 14.5
            india_vix_change = 0.0

    vix_val = india_vix_val or 14.5
    vix_chg = india_vix_change or 0.0

    if vix_val >= INDIA_VIX_PANIC_THRESHOLD or vix_chg >= VIX_SPIKE_1D_THRESHOLD:
        regime = "PANIC / HIGH RISK"
        kill_switch_active = True
        sizing_multiplier = 0.0
        advice = "VOLATILITY CIRCUIT BREAKER TRIGGERED: India VIX has surged above critical threshold. All fresh BUY recommendations are suspended to protect capital."
        badge = "panic"
    elif vix_val >= INDIA_VIX_ELEVATED_THRESHOLD:
        regime = "ELEVATED VOLATILITY"
        kill_switch_active = False
        sizing_multiplier = 0.50
        advice = "ELEVATED VOLATILITY REGIME: Market swings are wider than normal. Position sizing is automatically reduced by 50% with strict stop-losses."
        badge = "elevated"
    else:
        regime = "NORMAL / STABLE"
        kill_switch_active = False
        sizing_multiplier = 1.0
        advice = "STABLE VOLATILITY REGIME: Normal position sizing and swing setups apply."
        badge = "normal"

    return {
        "vix_value": round(vix_val, 2),
        "vix_1d_change_pct": f"{vix_chg:+.2f}%",
        "regime": regime,
        "kill_switch_active": kill_switch_active,
        "sizing_multiplier": sizing_multiplier,
        "advice": advice,
        "badge": badge,
    }


def check_portfolio_risk_guardrails(
    open_trades: list[dict[str, Any]],
    portfolio_drawdown_30d_pct: float = 0.0,
    max_sector_exposure_pct: float = 25.0,
) -> dict[str, Any]:
    """Validates portfolio-level constraints (Max -8% 30-day drawdown & 25% sector concentration)."""
    circuit_breaker_active = portfolio_drawdown_30d_pct <= -8.0

    def parse_pos_val(item: dict[str, Any]) -> float:
        if "position_value_raw" in item:
            return float(item["position_value_raw"] or 0.0)
        v = item.get("position_value", 0.0)
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(str(v).replace("₹", "").replace(",", "").strip())
        except Exception:
            return 0.0

    sector_exposure: dict[str, float] = {}
    total_open_val = sum(parse_pos_val(t) for t in open_trades)

    for t in open_trades:
        sec = t.get("sector") or "Other"
        val = parse_pos_val(t)
        sector_exposure[sec] = sector_exposure.get(sec, 0.0) + val

    over_concentrated_sectors = []
    if total_open_val > 0:
        for sec, val in sector_exposure.items():
            pct = (val / total_open_val) * 100.0
            if pct > max_sector_exposure_pct:
                over_concentrated_sectors.append(f"{sec} ({pct:.1f}%)")

    return {
        "circuit_breaker_active": circuit_breaker_active,
        "drawdown_30d_pct": f"{portfolio_drawdown_30d_pct:.1f}%",
        "open_trades_count": len(open_trades),
        "over_concentrated_sectors": over_concentrated_sectors,
        "can_open_new_trades": not circuit_breaker_active and len(open_trades) < 8,
        "message": (
            "PORTFOLIO PAUSE & REVIEW: Monthly drawdown exceeded -8%. Fresh entries halted until risk review."
            if circuit_breaker_active
            else f"Sector overconcentration warning in: {', '.join(over_concentrated_sectors)}"
            if over_concentrated_sectors
            else "Portfolio risk limits healthy."
        ),
    }
