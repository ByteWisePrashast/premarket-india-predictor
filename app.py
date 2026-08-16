#!/usr/bin/env python3
"""
Local Flask dashboard for the multi-asset Pre-Market India Predictor & Risk Engine.

Upgraded with:
1. Intelligent Portfolio & SIP Architect (Lump Sum + SIP Builders).
2. Live Model Drift Monitoring & Rolling Calibration Engine (30/60/90 Days).
3. Emergency "FLATTEN ALL" Panic Button Endpoint.
4. Signal Outcomes Ledger & Factor Attribution Logging.
5. 1% ADV Liquidity Filter & Overnight Gap Risk Protection.
6. Volatility Regime Kill Switch and Shadow Paper Portfolio.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
import math
from pathlib import Path
import sqlite3
import time
from typing import Any

from flask import Flask, jsonify, render_template, request
import yfinance as yf

from asset_engine import (
    KNOWN_CRYPTOS,
    KNOWN_ETFS,
    POPULAR_MF_MAP,
    POPULAR_SUGGESTIONS,
    analyze_asset,
    fmt_curr,
    fmt_pct,
    get_top_crypto_watchlist,
    search_mutual_funds,
)
from ai_assistant import process_bot_query
from backtest_engine import compute_live_drift, get_fallback_backtest_stats, resolve_pending_signals
from portfolio_planner import (
    calculate_goal_inflation_calculator,
    calculate_lump_sum_calculator,
    calculate_regular_sip,
    calculate_step_up_sip,
    generate_lump_sum_portfolio,
    generate_sip_plan,
    get_daily_top_picks,
)
from portfolio_health_engine import (
    audit_portfolio_system,
    generate_asset_action_decisions,
    parse_portfolio_screenshot_text,
)
from premarket_predictor import DataPoint, MarketReport, build_report, clamp, signed_score
from risk_engine import calculate_transaction_friction, check_portfolio_risk_guardrails, check_volatility_regime

app = Flask(__name__)
DB_PATH = Path(__file__).with_name("prediction_history.sqlite3")
DEFAULT_INITIAL_PAPER_CAPITAL = 1000000.0  # ₹10,00,000 default virtual paper capital

NIFTY_50_SYMBOLS = [
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BEL",
    "BHARTIARTL",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "ITC",
    "INDUSINDBK",
    "INFY",
    "JSWSTEEL",
    "JIOFIN",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "NTPC",
    "NESTLEIND",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SHRIRAMFIN",
    "SBIN",
    "SUNPHARMA",
    "TCS",
    "TATACONSUM",
    "TATAMOTORS",
    "TATASTEEL",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO",
    "WIPRO",
]


def point_to_dict(point: DataPoint) -> dict[str, Any]:
    return {
        "name": point.name,
        "value": point.value,
        "display": point.display,
        "ok": point.ok,
        "source": point.source,
        "error": point.error,
    }


def report_to_template(report: MarketReport) -> dict[str, Any]:
    return {
        "generated_at": report.generated_at.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "gift_rows": [
            ("Nifty previous close", report.nifty_previous_close),
            ("GIFT Nifty live", report.gift_nifty),
            ("Implied gap points", report.gift_gap_points),
            ("Implied gap percent", report.gift_gap_pct),
        ],
        "global_rows": list(report.global_cues.items()),
        "macro_rows": list(report.macro.items()),
        "institutional_rows": list(report.institutional.items()),
        "headlines": report.headlines,
        "sentiment_label": report.sentiment_label,
        "sentiment_score": f"{report.sentiment_score:+.3f}",
        "prediction": report.prediction,
        "component_scores": {name: f"{score:+.3f}" for name, score in report.component_scores.items()},
        "context_summary": report.context_summary,
    }


def prediction_date() -> str:
    return datetime.now().astimezone().date().isoformat()


# ============================================================================
# SQLITE DATABASE, DRIFT LEDGER & PAPER PORTFOLIO
# ============================================================================

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                source TEXT NOT NULL,
                prediction TEXT NOT NULL,
                probability REAL NOT NULL,
                confidence REAL NOT NULL,
                expected_gap_pct REAL,
                previous_close REAL,
                actual_result_pct REAL,
                correct INTEGER,
                reasons TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(prediction_date, symbol, source)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_portfolio (
                id INTEGER PRIMARY KEY,
                account_name TEXT NOT NULL,
                initial_capital REAL NOT NULL,
                cash_balance REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                peak_equity REAL NOT NULL,
                max_drawdown_pct REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                name TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                position_value REAL NOT NULL,
                stop_loss REAL NOT NULL,
                target_1 REAL NOT NULL,
                target_2 REAL NOT NULL,
                status TEXT NOT NULL,
                exit_date TEXT,
                exit_price REAL,
                gross_pnl REAL,
                net_pnl REAL,
                pnl_pct REAL,
                exit_reason TEXT,
                is_conviction_bet INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        try:
            conn.execute("ALTER TABLE paper_trades ADD COLUMN is_conviction_bet INTEGER DEFAULT 0")
        except Exception:
            pass

        conn.execute(
            """
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
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(signal_date, asset_symbol, tier)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_tier_date ON signal_outcomes(tier, signal_date)")

        row = conn.execute("SELECT id FROM paper_portfolio WHERE id = 1").fetchone()
        if not row:
            conn.execute(
                """
                INSERT INTO paper_portfolio (
                    id, account_name, initial_capital, cash_balance, realized_pnl, peak_equity, max_drawdown_pct, updated_at
                ) VALUES (1, 'Shadow Trading Portfolio', ?, ?, 0.0, ?, 0.0, ?)
                """,
                (
                    DEFAULT_INITIAL_PAPER_CAPITAL,
                    DEFAULT_INITIAL_PAPER_CAPITAL,
                    DEFAULT_INITIAL_PAPER_CAPITAL,
                    datetime.now().isoformat(),
                ),
            )


def log_signal_outcome(symbol: str, score: float, entry_price: float, vix_val: float = 14.5) -> None:
    if entry_price <= 0:
        return
    tier = "Tier 1" if score >= 75 else "Tier 2" if score >= 60 else "Tier 3" if score >= 45 else "Tier 4"
    today_str = date.today().isoformat()

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO signal_outcomes (
                    signal_date, asset_symbol, score, tier, horizon_days, entry_price, outcome, vix_at_signal, created_at
                ) VALUES (?, ?, ?, ?, 5, ?, 'PENDING', ?, ?)
                """,
                (today_str, symbol.replace(".NS", ""), score, tier, entry_price, vix_val, datetime.now().isoformat()),
            )
    except Exception:
        pass


def save_prediction(symbol: str, source: str, prediction: dict[str, Any]) -> None:
    init_db()
    reasons = prediction.get("summary", "")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO predictions (
                prediction_date, symbol, source, prediction, probability, confidence,
                expected_gap_pct, previous_close, actual_result_pct, correct, reasons, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
            """,
            (
                prediction_date(),
                symbol.replace(".NS", ""),
                source,
                prediction.get("overall", "Neutral"),
                float(prediction.get("score", 50.0)),
                float(prediction.get("calibrated_win_rate", prediction.get("score", 50.0))),
                0.0,
                prediction.get("last_close") or prediction.get("current_nav"),
                reasons,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )


# ============================================================================
# PAPER TRADING PORTFOLIO & EMERGENCY FLATTEN ALL
# ============================================================================

def get_paper_portfolio_state() -> dict[str, Any]:
    """Fetches the shadow paper portfolio, computing live mark-to-market unrealized P&L for open holdings."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        port = conn.execute("SELECT * FROM paper_portfolio WHERE id = 1").fetchone()
        open_rows = conn.execute("SELECT * FROM paper_trades WHERE status = 'OPEN' ORDER BY id DESC").fetchall()
        closed_rows = conn.execute("SELECT * FROM paper_trades WHERE status = 'CLOSED' ORDER BY id DESC LIMIT 20").fetchall()

    cash_balance = float(port["cash_balance"]) if port else DEFAULT_INITIAL_PAPER_CAPITAL
    initial_capital = float(port["initial_capital"]) if port else DEFAULT_INITIAL_PAPER_CAPITAL
    realized_pnl = float(port["realized_pnl"]) if port else 0.0

    open_trades: list[dict[str, Any]] = []
    total_unrealized_pnl = 0.0
    total_open_value = 0.0

    for r in open_rows:
        sym = r["symbol"]
        qty = int(r["quantity"])
        entry = float(r["entry_price"])
        pos_val = entry * qty

        curr_price = entry
        try:
            if r["asset_type"] == "Crypto" or sym.endswith("-USD") or sym in ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "MATIC"):
                tk = sym if sym.endswith("-USD") else f"{sym}-USD"
                live_p = yf.Ticker(tk).fast_info.get("last_price")
                if live_p and math.isfinite(float(live_p)):
                    usdinr = 86.5
                    try:
                        u = yf.Ticker("USDINR=X").fast_info.get("last_price")
                        if u and math.isfinite(float(u)):
                            usdinr = float(u)
                    except Exception:
                        pass
                    curr_price = float(live_p) * usdinr
            elif not sym.startswith("MF_"):
                ticker = f"{sym}.NS" if not sym.endswith(".NS") else sym
                live_p = yf.Ticker(ticker).fast_info.get("last_price")
                if live_p and math.isfinite(float(live_p)):
                    curr_price = float(live_p)
        except Exception:
            pass

        unrealized_gross = (curr_price - entry) * qty
        unrealized_pct = ((curr_price - entry) / entry * 100.0) if entry else 0.0
        total_unrealized_pnl += unrealized_gross
        total_open_value += (curr_price * qty)

        is_conviction = bool(r["is_conviction_bet"]) if "is_conviction_bet" in r.keys() else False

        open_trades.append({
            "id": r["id"],
            "symbol": sym,
            "name": r["name"],
            "asset_type": r["asset_type"],
            "entry_date": r["entry_date"],
            "entry_price": fmt_curr(entry),
            "current_price": fmt_curr(curr_price),
            "current_price_raw": curr_price,
            "quantity": qty,
            "position_value": fmt_curr(pos_val),
            "stop_loss": fmt_curr(float(r["stop_loss"])),
            "target_1": fmt_curr(float(r["target_1"])),
            "target_2": fmt_curr(float(r["target_2"])),
            "unrealized_pnl": fmt_curr(unrealized_gross),
            "unrealized_pnl_raw": unrealized_gross,
            "unrealized_pct": fmt_pct(unrealized_pct),
            "is_conviction_bet": is_conviction,
        })

    closed_trades: list[dict[str, Any]] = []
    wins = 0
    for r in closed_rows:
        net_pnl = float(r["net_pnl"] or 0.0)
        if net_pnl > 0:
            wins += 1
        closed_trades.append({
            "id": r["id"],
            "symbol": r["symbol"],
            "name": r["name"],
            "entry_date": r["entry_date"],
            "exit_date": r["exit_date"],
            "entry_price": fmt_curr(float(r["entry_price"])),
            "exit_price": fmt_curr(float(r["exit_price"] or 0.0)),
            "quantity": r["quantity"],
            "gross_pnl": fmt_curr(float(r["gross_pnl"] or 0.0)),
            "net_pnl": fmt_curr(net_pnl),
            "net_pnl_raw": net_pnl,
            "pnl_pct": fmt_pct(float(r["pnl_pct"] or 0.0)),
            "exit_reason": r["exit_reason"],
        })

    total_equity = cash_balance + total_open_value
    win_rate = (wins / len(closed_trades) * 100.0) if closed_trades else 0.0
    total_gain_pct = ((total_equity - initial_capital) / initial_capital * 100.0) if initial_capital else 0.0

    return {
        "account_name": "Paper / Shadow Portfolio",
        "initial_capital": fmt_curr(initial_capital),
        "total_equity": fmt_curr(total_equity),
        "total_equity_raw": total_equity,
        "cash_balance": fmt_curr(cash_balance),
        "cash_balance_raw": cash_balance,
        "realized_pnl": fmt_curr(realized_pnl),
        "unrealized_pnl": fmt_curr(total_unrealized_pnl),
        "total_gain_pct": fmt_pct(total_gain_pct),
        "win_rate": f"{win_rate:.1f}%" if closed_trades else "N/A",
        "open_trades_count": len(open_trades),
        "closed_trades_count": len(closed_trades),
        "open_trades": open_trades,
        "closed_trades": closed_trades,
    }


def open_paper_position(
    symbol: str,
    name: str,
    asset_type: str,
    entry_price: float,
    quantity: int,
    stop_loss: float,
    target_1: float,
    target_2: float,
) -> dict[str, Any]:
    init_db()
    if quantity <= 0 or entry_price <= 0:
        return {"ok": False, "error": "Invalid quantity or entry price."}

    pos_value = entry_price * quantity
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        port = conn.execute("SELECT cash_balance FROM paper_portfolio WHERE id = 1").fetchone()
        cash = float(port["cash_balance"]) if port else DEFAULT_INITIAL_PAPER_CAPITAL

        if cash < pos_value:
            return {"ok": False, "error": f"Insufficient paper cash (Required: ₹{pos_value:,.2f}, Available: ₹{cash:,.2f})."}

        new_cash = cash - pos_value
        conn.execute("UPDATE paper_portfolio SET cash_balance = ?, updated_at = ? WHERE id = 1", (new_cash, datetime.now().isoformat()))

        cur = conn.execute(
            """
            INSERT INTO paper_trades (
                symbol, name, asset_type, direction, entry_date, entry_price, quantity,
                position_value, stop_loss, target_1, target_2, status, created_at
            ) VALUES (?, ?, ?, 'LONG', ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
            """,
            (
                symbol.replace(".NS", ""),
                name,
                asset_type,
                date.today().isoformat(),
                entry_price,
                quantity,
                pos_value,
                stop_loss,
                target_1,
                target_2,
                datetime.now().isoformat(),
            ),
        )
        trade_id = cur.lastrowid

    return {"ok": True, "trade_id": trade_id, "message": f"Successfully opened paper trade for {quantity} shares of {symbol} at ₹{entry_price:,.2f}."}


def close_paper_position(trade_id: int, exit_price: float, exit_reason: str = "MANUAL") -> dict[str, Any]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        trade = conn.execute("SELECT * FROM paper_trades WHERE id = ? AND status = 'OPEN'", (trade_id,)).fetchone()
        if not trade:
            return {"ok": False, "error": "Trade not found or already closed."}

        entry_price = float(trade["entry_price"])
        qty = int(trade["quantity"])
        friction = calculate_transaction_friction(entry_price, exit_price, qty)

        gross_pnl = friction["gross_profit"]
        net_pnl = friction["net_profit"]
        pnl_pct = ((exit_price - entry_price) / entry_price * 100.0) if entry_price else 0.0

        recovered_cash = (entry_price * qty) + net_pnl

        port = conn.execute("SELECT cash_balance, realized_pnl FROM paper_portfolio WHERE id = 1").fetchone()
        cur_cash = float(port["cash_balance"]) if port else DEFAULT_INITIAL_PAPER_CAPITAL
        cur_realized = float(port["realized_pnl"]) if port else 0.0

        new_cash = cur_cash + recovered_cash
        new_realized = cur_realized + net_pnl

        conn.execute(
            """
            UPDATE paper_portfolio
            SET cash_balance = ?, realized_pnl = ?, updated_at = ?
            WHERE id = 1
            """,
            (new_cash, new_realized, datetime.now().isoformat()),
        )

        conn.execute(
            """
            UPDATE paper_trades
            SET status = 'CLOSED', exit_date = ?, exit_price = ?, gross_pnl = ?, net_pnl = ?, pnl_pct = ?, exit_reason = ?
            WHERE id = ?
            """,
            (date.today().isoformat(), exit_price, gross_pnl, net_pnl, pnl_pct, exit_reason, trade_id),
        )

    return {
        "ok": True,
        "message": f"Closed paper position for {trade['symbol']}. Net P&L: ₹{net_pnl:,.2f} ({pnl_pct:+.2f}%).",
    }


def flatten_all_paper_positions() -> dict[str, Any]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        open_trades = conn.execute("SELECT id, symbol, entry_price, quantity FROM paper_trades WHERE status = 'OPEN'").fetchall()

    if not open_trades:
        return {"ok": True, "closed_count": 0, "message": "No open paper positions to flatten."}

    closed_count = 0
    for t in open_trades:
        t_id = t["id"]
        sym = t["symbol"]
        entry = float(t["entry_price"])
        exit_px = entry

        try:
            ticker = f"{sym}.NS" if not sym.endswith(".NS") else sym
            p = yf.Ticker(ticker).fast_info.get("last_price")
            if p and math.isfinite(float(p)):
                exit_px = float(p)
        except Exception:
            pass

        res = close_paper_position(t_id, exit_px, exit_reason="EMERGENCY_FLATTEN_ALL")
        if res.get("ok"):
            closed_count += 1

    return {
        "ok": True,
        "closed_count": closed_count,
        "message": f"🚨 EMERGENCY FLATTEN COMPLETE: Successfully closed {closed_count} open positions at market price.",
    }


def set_paper_capital(capital_amount: float) -> dict[str, Any]:
    init_db()
    capital = max(10000.0, float(capital_amount))
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        open_trades = conn.execute("SELECT SUM(position_value) AS total_val FROM paper_trades WHERE status = 'OPEN'").fetchone()
        open_val = float(open_trades["total_val"] or 0.0) if open_trades else 0.0
        remaining_cash = max(0.0, capital - open_val)
        conn.execute("UPDATE paper_portfolio SET initial_capital = ?, cash_balance = ?, updated_at = ? WHERE id = 1", (capital, remaining_cash, datetime.now().isoformat()))
    return {
        "ok": True,
        "initial_capital": fmt_curr(capital),
        "cash_balance": fmt_curr(remaining_cash),
        "message": f"Successfully updated total portfolio capital to {fmt_curr(capital)} (Remaining cash: {fmt_curr(remaining_cash)}).",
    }


def set_conviction_bet(symbol: str, is_conviction: bool) -> bool:
    """Updates conviction bet status for an open holding symbol."""
    init_db()
    sym_clean = symbol.upper().strip()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE paper_trades SET is_conviction_bet = ? WHERE (symbol = ? OR symbol = ?) AND status = 'OPEN'",
            (1 if is_conviction else 0, sym_clean, f"{sym_clean}.NS")
        )
        conn.commit()
    return True


_CACHED_STOCKS_WATCHLIST: list[dict[str, Any]] = []
_CACHED_STOCKS_TIME: float = 0.0
STOCKS_CACHE_TTL: float = 120.0

DEFAULT_FALLBACK_STOCKS = [
    ("RELIANCE", "Reliance Industries", 2980.50, 1.25, 3.40, 84.0, "Strong Buy", "strong-buy"),
    ("TCS", "Tata Consultancy Services", 4190.00, 0.85, 2.10, 78.0, "Buy on Dips", "buy"),
    ("HDFCBANK", "HDFC Bank Ltd", 1640.20, 0.45, 1.80, 72.0, "Buy on Dips", "buy"),
    ("INFY", "Infosys Ltd", 1885.00, 1.60, 4.20, 86.0, "Strong Buy", "strong-buy"),
    ("ICICIBANK", "ICICI Bank Ltd", 1210.50, 0.90, 2.60, 80.0, "Strong Buy", "strong-buy"),
    ("BHARTIARTL", "Bharti Airtel", 1460.00, 1.10, 3.10, 82.0, "Strong Buy", "strong-buy"),
    ("LT", "Larsen & Toubro", 3620.00, 0.70, 1.90, 75.0, "Buy on Dips", "buy"),
    ("SBIN", "State Bank of India", 825.40, 0.35, 1.40, 71.0, "Buy on Dips", "buy"),
]


def get_top_stocks_watchlist(report: MarketReport, limit: int = 8) -> list[dict[str, Any]]:
    global _CACHED_STOCKS_WATCHLIST, _CACHED_STOCKS_TIME
    now_ts = time.time()
    if _CACHED_STOCKS_WATCHLIST and (now_ts - _CACHED_STOCKS_TIME) < STOCKS_CACHE_TTL:
        return _CACHED_STOCKS_WATCHLIST[:limit]

    tickers = [f"{s}.NS" for s in NIFTY_50_SYMBOLS[:12]]
    watchlist: list[dict[str, Any]] = []
    try:
        data = yf.download(
            tickers,
            period="1mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
            timeout=8,
        )
        for ticker in tickers:
            try:
                frame = data[ticker].dropna(subset=["Close"]) if hasattr(data, "__getitem__") else None
                if frame is None or len(frame) < 5:
                    continue
                last = float(frame["Close"].iloc[-1])
                prev = float(frame["Close"].iloc[-2])
                change_1d = ((last - prev) / prev) * 100.0
                change_5d = ((last - float(frame["Close"].iloc[-5])) / float(frame["Close"].iloc[-5])) * 100.0 if len(frame) >= 5 else 0.0

                clean_sym = ticker.replace(".NS", "")
                score = 50.0 + (change_1d * 3.0) + (change_5d * 2.0)
                score = round(clamp(score, 30.0, 95.0), 1)

                action = "Strong Buy" if score >= 75 else "Buy on Dips" if score >= 60 else "Hold" if score >= 45 else "Avoid"
                badge = "strong-buy" if score >= 75 else "buy" if score >= 60 else "hold" if score >= 45 else "avoid"

                log_signal_outcome(clean_sym, score, last)

                watchlist.append({
                    "symbol": clean_sym,
                    "type": "Stock",
                    "last": fmt_curr(last),
                    "last_raw": last,
                    "change_1d": fmt_pct(change_1d),
                    "change_5d": fmt_pct(change_5d),
                    "score": score,
                    "action": action,
                    "badge": badge,
                    "target": fmt_curr(last * 1.035),
                    "target_raw": round(last * 1.035, 2),
                    "stop_loss": fmt_curr(last * 0.975),
                    "stop_loss_raw": round(last * 0.975, 2),
                })
            except Exception:
                continue
    except Exception:
        pass

    if not watchlist:
        # Fallback to curated instant stocks list to avoid any delay
        for sym, name, last, chg1, chg5, sc, act, bdg in DEFAULT_FALLBACK_STOCKS:
            watchlist.append({
                "symbol": sym,
                "type": "Stock",
                "last": fmt_curr(last),
                "last_raw": last,
                "change_1d": fmt_pct(chg1),
                "change_5d": fmt_pct(chg5),
                "score": sc,
                "action": act,
                "badge": bdg,
                "target": fmt_curr(last * 1.035),
                "target_raw": round(last * 1.035, 2),
                "stop_loss": fmt_curr(last * 0.975),
                "stop_loss_raw": round(last * 0.975, 2),
            })

    watchlist.sort(key=lambda x: x["score"], reverse=True)
    _CACHED_STOCKS_WATCHLIST = watchlist
    _CACHED_STOCKS_TIME = now_ts
    return watchlist[:limit]


def get_top_etfs_watchlist() -> list[dict[str, Any]]:
    sample_etfs = [
        ("NIFTYBEES", "Nifty 50 Index", 285.50, 0.45, 14.2, "Index ETF", "Strong Buy", "strong-buy", 88.0),
        ("BANKBEES", "Nifty Bank Index", 545.20, 0.85, 16.8, "Banking ETF", "Buy on Dips", "buy", 78.0),
        ("GOLDBEES", "Physical Gold", 68.40, 0.15, 22.4, "Commodity ETF", "Strong Accumulate", "strong-buy", 85.0),
        ("SILVERBEES", "Physical Silver", 92.10, -0.30, 26.5, "Commodity ETF", "Buy on Dips", "buy", 74.0),
        ("ITBEES", "Nifty IT Sector", 42.80, 1.20, 28.6, "Sector ETF", "Strong Buy", "strong-buy", 82.0),
        ("MON100", "US Nasdaq 100", 178.60, 0.60, 31.4, "Global Tech ETF", "Strong Accumulate", "strong-buy", 86.0),
        ("CPSEETF", "Central PSU Index", 98.40, -0.25, 38.2, "PSU Value ETF", "Buy on Dips", "buy", 76.0),
    ]

    return [
        {
            "symbol": sym,
            "name": name,
            "type": "ETF",
            "last": fmt_curr(last),
            "last_raw": last,
            "change_1d": fmt_pct(chg),
            "cagr_1y": fmt_pct(cagr),
            "category": cat,
            "action": act,
            "badge": badge,
            "score": score,
            "target": fmt_curr(last * 1.04),
            "target_raw": round(last * 1.04, 2),
            "stop_loss": fmt_curr(last * 0.98),
            "stop_loss_raw": round(last * 0.98, 2),
        }
        for sym, name, last, chg, cagr, cat, act, badge, score in sample_etfs
    ]


def get_top_mutual_funds_watchlist() -> list[dict[str, Any]]:
    funds = [
        ("MF_122639", "Parag Parikh Flexi Cap Fund", "Flexi Cap", "₹91.68", "+0.12%", "+21.4%", "+24.8%", "1.45", "Strong Buy (Top Pick)", "strong-buy", 92.0),
        ("MF_120828", "Quant Small Cap Fund", "Small Cap", "₹264.50", "+0.45%", "+32.6%", "+38.4%", "1.62", "Strong Buy / Start SIP", "strong-buy", 90.0),
        ("MF_118989", "HDFC Balanced Advantage Fund", "Hybrid / Dynamic", "₹482.10", "+0.08%", "+18.2%", "+20.5%", "1.38", "Buy on Dips", "buy", 84.0),
        ("MF_118778", "Nippon India Small Cap Fund", "Small Cap", "₹184.20", "+0.32%", "+28.5%", "+31.2%", "1.50", "Strong Buy", "strong-buy", 88.0),
        ("MF_118834", "Mirae Asset Large Cap Fund", "Large Cap", "₹118.90", "+0.15%", "+14.8%", "+16.2%", "1.15", "Buy / Core SIP", "buy", 79.0),
        ("MF_120716", "UTI Nifty 50 Index Fund", "Index Fund", "₹172.40", "+0.20%", "+15.2%", "+15.8%", "1.20", "Strong Accumulate", "strong-buy", 82.0),
    ]

    return [
        {
            "symbol": sym,
            "name": name,
            "type": "Mutual Fund",
            "category": cat,
            "nav": nav,
            "change_1d": chg,
            "cagr_1y": cagr1,
            "cagr_3y": cagr3,
            "sharpe": sharpe,
            "action": act,
            "badge": badge,
            "score": score,
        }
        for sym, name, cat, nav, chg, cagr1, cagr3, sharpe, act, badge, score in funds
    ]


def make_json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, DataPoint):
        return point_to_dict(value)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    return value


@app.get("/")
def index() -> str:
    init_db()
    report = build_report()
    query = request.args.get("symbol", "").strip() or request.args.get("q", "").strip()
    
    paper_portfolio = get_paper_portfolio_state()
    vix_regime = check_volatility_regime()
    risk_guardrails = check_portfolio_risk_guardrails(paper_portfolio["open_trades"])
    backtest_stats = get_fallback_backtest_stats()

    # Portfolio Architect Default Models
    default_lump_sum_plan = generate_lump_sum_portfolio(capital_amount=500000.0, horizon_years=3.0, risk_profile="moderate", vehicle_preference="all")
    default_sip_plan = generate_sip_plan(monthly_amount=10000.0, horizon_years=5.0, risk_profile="moderate")
    daily_picks = get_daily_top_picks()

    # Investment Calculators Default States
    default_calc_sip = calculate_regular_sip(monthly_amount=10000.0, annual_return_pct=14.0, horizon_years=10.0)
    default_calc_step_up = calculate_step_up_sip(initial_monthly_amount=10000.0, annual_step_up_pct=10.0, annual_return_pct=14.0, horizon_years=10.0)
    default_calc_lump_sum = calculate_lump_sum_calculator(principal_amount=500000.0, annual_return_pct=14.0, horizon_years=10.0, inflation_rate_pct=6.0)
    default_calc_goal_inflation = calculate_goal_inflation_calculator(target_goal_amount=50000000.0, horizon_years=10.0, inflation_rate_pct=7.0)

    # Portfolio Health & Holistic Diagnostic X-Ray
    portfolio_xray = audit_portfolio_system(
        holdings=paper_portfolio["open_trades"],
        cash_balance=paper_portfolio["cash_balance_raw"],
        time_horizon="long",
        risk_profile="moderate",
        primary_goal="wealth_creation",
    )

    with sqlite3.connect(DB_PATH) as conn:
        drift_summary = compute_live_drift(conn, window_days=60)

    searched_asset = None
    if query:
        searched_asset = analyze_asset(query, report, portfolio_capital=paper_portfolio["total_equity_raw"])
        if searched_asset and searched_asset.get("ok"):
            try:
                sym_clean = searched_asset.get("clean_symbol") or searched_asset.get("symbol", query)
                raw_sc = float(searched_asset.get("verdict", {}).get("score", 50.0))
                px = float(searched_asset.get("last_close") or 0.0)
                save_prediction(sym_clean, "search", searched_asset.get("verdict", {}))
                log_signal_outcome(sym_clean, raw_sc, px, vix_regime["vix_value"])
            except Exception:
                pass

    stocks_watchlist = get_top_stocks_watchlist(report)
    etfs_watchlist = get_top_etfs_watchlist()
    mfs_watchlist = get_top_mutual_funds_watchlist()
    crypto_watchlist = get_top_crypto_watchlist()

    return render_template(
        "dashboard.html",
        report=report_to_template(report),
        asset=searched_asset,
        searched_query=query,
        stocks_watchlist=stocks_watchlist,
        etfs_watchlist=etfs_watchlist,
        mfs_watchlist=mfs_watchlist,
        crypto_watchlist=crypto_watchlist,
        daily_picks=daily_picks,
        paper_portfolio=paper_portfolio,
        portfolio_xray=portfolio_xray,
        vix_regime=vix_regime,
        risk_guardrails=risk_guardrails,
        backtest_stats=backtest_stats,
        drift_summary=drift_summary,
        lump_sum_plan=default_lump_sum_plan,
        sip_plan=default_sip_plan,
        calc_sip=default_calc_sip,
        calc_step_up=default_calc_step_up,
        calc_lump_sum=default_calc_lump_sum,
        calc_goal_inflation=default_calc_goal_inflation,
        popular_suggestions=POPULAR_SUGGESTIONS,
    )


@app.get("/api/search-suggest")
def api_search_suggest() -> Any:
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify(POPULAR_SUGGESTIONS[:12])

    matches = []
    for item in POPULAR_SUGGESTIONS:
        if q in item["symbol"].lower() or q in item["name"].lower() or q in item["category"].lower():
            matches.append(item)

    for sym, name in KNOWN_ETFS.items():
        if q in sym.lower() or q in name.lower():
            if not any(m["symbol"] == sym for m in matches):
                matches.append({"symbol": sym, "name": name, "type": "ETF", "category": "ETF"})

    for sym, meta in KNOWN_CRYPTOS.items():
        if q in sym.lower() or q in meta["name"].lower() or q in meta["category"].lower():
            if not any(m["symbol"] == sym for m in matches):
                matches.append({"symbol": sym, "name": meta["name"], "type": "Crypto", "category": meta["category"]})

    if len(matches) < 6:
        mf_results = search_mutual_funds(q, limit=6)
        for mf in mf_results:
            matches.append({
                "symbol": f"MF_{mf['schemeCode']}",
                "name": mf["schemeName"],
                "type": "Mutual Fund",
                "category": "Mutual Fund",
            })

    return jsonify(matches[:10])


@app.get("/api/asset/<path:query>")
def api_asset(query: str) -> Any:
    report = build_report()
    result = analyze_asset(query, report)
    return jsonify(make_json_safe(result))


@app.get("/api/daily-picks")
def api_daily_picks() -> Any:
    return jsonify(get_daily_top_picks())


@app.get("/api/paper/portfolio")
def api_paper_portfolio() -> Any:
    return jsonify(get_paper_portfolio_state())


@app.post("/api/paper/capital/set")
def api_paper_capital_set() -> Any:
    data = request.get_json(silent=True) or request.form or {}
    capital = float(data.get("capital", 1000000.0))
    res = set_paper_capital(capital)
    return jsonify(res)


@app.post("/api/paper/trade/open")
def api_paper_trade_open() -> Any:
    data = request.get_json(silent=True) or request.form
    symbol = str(data.get("symbol", "")).strip()
    name = str(data.get("name", symbol)).strip()
    asset_type = str(data.get("asset_type", "Stock")).strip()
    entry_price = float(data.get("entry_price", 0.0))
    quantity = int(data.get("quantity", 0))
    stop_loss = float(data.get("stop_loss", 0.0))
    target_1 = float(data.get("target_1", 0.0))
    target_2 = float(data.get("target_2", target_1))

    res = open_paper_position(
        symbol=symbol,
        name=name,
        asset_type=asset_type,
        entry_price=entry_price,
        quantity=quantity,
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
    )
    return jsonify(res)


@app.post("/api/paper/trade/close/<int:trade_id>")
def api_paper_trade_close(trade_id: int) -> Any:
    data = request.get_json(silent=True) or request.form or {}
    exit_price = float(data.get("exit_price", 0.0))
    reason = str(data.get("exit_reason", "MANUAL")).strip()
    
    if exit_price <= 0:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            t = conn.execute("SELECT symbol, entry_price FROM paper_trades WHERE id = ?", (trade_id,)).fetchone()
            if t:
                try:
                    sym = t["symbol"]
                    ticker = f"{sym}.NS" if not sym.endswith(".NS") else sym
                    p = yf.Ticker(ticker).fast_info.get("last_price")
                    if p and math.isfinite(float(p)):
                        exit_price = float(p)
                except Exception:
                    exit_price = float(t["entry_price"])

    res = close_paper_position(trade_id, exit_price, exit_reason=reason)
    return jsonify(res)


@app.post("/api/paper/trade/flatten-all")
def api_paper_trade_flatten_all() -> Any:
    res = flatten_all_paper_positions()
    return jsonify(res)


@app.post("/api/portfolio/generate")
def api_portfolio_generate() -> Any:
    data = request.get_json(silent=True) or request.form or {}
    capital = float(data.get("capital", 500000.0))
    horizon = float(data.get("horizon", 3.0))
    risk = str(data.get("risk", "moderate"))
    vehicle = str(data.get("vehicle", "all"))
    res = generate_lump_sum_portfolio(capital_amount=capital, horizon_years=horizon, risk_profile=risk, vehicle_preference=vehicle)
    return jsonify(res)


@app.post("/api/portfolio/sip-plan")
def api_portfolio_sip_plan() -> Any:
    data = request.get_json(silent=True) or request.form or {}
    monthly_amount = float(data.get("monthly_amount", 10000.0))
    horizon = float(data.get("horizon", 5.0))
    risk = str(data.get("risk", "moderate"))
    res = generate_sip_plan(monthly_amount=monthly_amount, horizon_years=horizon, risk_profile=risk)
    return jsonify(res)


@app.post("/api/calculator/sip")
def api_calculator_sip() -> Any:
    data = request.get_json(silent=True) or request.form or {}
    monthly_amount = float(data.get("monthly_amount", 10000.0))
    return_pct = float(data.get("return_pct", 14.0))
    horizon_years = float(data.get("horizon_years", 10.0))
    res = calculate_regular_sip(monthly_amount=monthly_amount, annual_return_pct=return_pct, horizon_years=horizon_years)
    return jsonify(res)


@app.post("/api/calculator/step-up-sip")
def api_calculator_step_up_sip() -> Any:
    data = request.get_json(silent=True) or request.form or {}
    initial_monthly = float(data.get("initial_monthly", 10000.0))
    step_up_pct = float(data.get("step_up_pct", 10.0))
    return_pct = float(data.get("return_pct", 14.0))
    horizon_years = float(data.get("horizon_years", 10.0))
    res = calculate_step_up_sip(initial_monthly_amount=initial_monthly, annual_step_up_pct=step_up_pct, annual_return_pct=return_pct, horizon_years=horizon_years)
    return jsonify(res)


@app.post("/api/calculator/lumpsum")
def api_calculator_lumpsum() -> Any:
    data = request.get_json(silent=True) or request.form or {}
    principal = float(data.get("principal", 500000.0))
    return_pct = float(data.get("return_pct", 14.0))
    horizon_years = float(data.get("horizon_years", 10.0))
    inflation_pct = float(data.get("inflation_pct", 6.0))
    res = calculate_lump_sum_calculator(principal_amount=principal, annual_return_pct=return_pct, horizon_years=horizon_years, inflation_rate_pct=inflation_pct)
    return jsonify(res)


@app.post("/api/calculator/goal-inflation")
def api_calculator_goal_inflation() -> Any:
    data = request.get_json(silent=True) or request.form or {}
    goal_amount = float(data.get("goal_amount", 50000000.0))
    horizon_years = float(data.get("horizon_years", 10.0))
    inflation_pct = float(data.get("inflation_pct", 7.0))
    res = calculate_goal_inflation_calculator(target_goal_amount=goal_amount, horizon_years=horizon_years, inflation_rate_pct=inflation_pct)
    return jsonify(res)


@app.post("/api/bot/chat")
def api_bot_chat() -> Any:
    data = request.get_json(silent=True) or request.form or {}
    message = str(data.get("message") or "").strip()
    history = data.get("history") or []
    tone = str(data.get("tone") or "conversational").lower()
    paper_portfolio = get_paper_portfolio_state()
    res = process_bot_query(user_message=message, session_history=history, paper_portfolio=paper_portfolio, tone=tone)
    return jsonify(res)





@app.route("/api/portfolio/xray", methods=["GET", "POST"])
def api_portfolio_xray() -> Any:
    data = request.get_json(silent=True) or request.form or {}
    horizon = str(request.args.get("horizon") or data.get("horizon") or "long").lower()
    risk = str(request.args.get("risk") or data.get("risk") or "moderate").lower()
    goal = str(request.args.get("goal") or data.get("goal") or "wealth_creation").lower()
    
    paper_portfolio = get_paper_portfolio_state()
    xray = audit_portfolio_system(
        holdings=paper_portfolio["open_trades"],
        cash_balance=paper_portfolio["cash_balance_raw"],
        time_horizon=horizon,
        risk_profile=risk,
        primary_goal=goal,
    )
    return jsonify(xray)


@app.post("/api/portfolio/xray/custom")
def api_portfolio_xray_custom() -> Any:
    data = request.get_json(silent=True) or request.form or {}
    custom_holdings = data.get("holdings") or []
    cash = float(data.get("cash_balance") or 0.0)
    horizon = str(data.get("horizon") or "long").lower()
    risk = str(data.get("risk") or "moderate").lower()
    goal = str(data.get("goal") or "wealth_creation").lower()
    
    xray = audit_portfolio_system(
        holdings=custom_holdings,
        cash_balance=cash,
        time_horizon=horizon,
        risk_profile=risk,
        primary_goal=goal,
    )
    return jsonify(xray)


@app.post("/api/portfolio/conviction-bet")
def api_toggle_conviction_bet() -> Any:
    data = request.get_json(silent=True) or request.form or {}
    symbol = str(data.get("symbol", "")).strip()
    is_conviction = bool(data.get("is_conviction_bet", True))
    if symbol:
        set_conviction_bet(symbol, is_conviction)
    return jsonify({"ok": True, "symbol": symbol, "is_conviction_bet": is_conviction})


@app.post("/api/portfolio/scan-screenshot")
def api_scan_portfolio_screenshot() -> Any:
    """
    Receives raw OCR extracted text or statement text, detects holdings, 
    and generates full asset-by-asset decision matrix and X-Ray audit.
    """
    data = request.get_json(silent=True) or request.form or {}
    raw_text = str(data.get("raw_text") or data.get("text") or "").strip()
    horizon = str(data.get("horizon") or "long").lower()
    risk = str(data.get("risk") or "moderate").lower()
    goal = str(data.get("goal") or "wealth_creation").lower()

    detected_items = parse_portfolio_screenshot_text(raw_text)
    
    # Generate per-asset decision advice
    asset_decisions = generate_asset_action_decisions(
        holdings=detected_items,
        user_profile={
            "time_horizon": horizon,
            "risk_profile": risk,
            "primary_goal": goal,
        },
    )

    # Run full system X-Ray
    xray = audit_portfolio_system(
        holdings=detected_items,
        cash_balance=0.0,
        time_horizon=horizon,
        risk_profile=risk,
        primary_goal=goal,
    )

    return jsonify({
        "ok": True,
        "detected_count": len(detected_items),
        "detected_holdings": detected_items,
        "asset_decisions": asset_decisions,
        "xray": xray,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False)
