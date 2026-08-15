#!/usr/bin/env python3
"""
Pre-Market Indian Market Predictor

Runs best around 08:45 IST. The script gathers overnight and morning market
signals, scores them, and prints a terminal report for the expected Nifty 50
opening bias.

Data quality note:
Free public market endpoints can fail, move, or rate-limit. The script is
written to keep running with partial data and to make unavailable components
neutral instead of crashing the report.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

import requests
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from requests import Response
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


IST = ZoneInfo("Asia/Kolkata")
REQUEST_TIMEOUT_SECONDS = 12


@dataclass
class DataPoint:
    name: str
    value: float | None
    display: str
    ok: bool
    source: str
    error: str | None = None


@dataclass
class MarketReport:
    generated_at: datetime
    nifty_previous_close: DataPoint
    gift_nifty: DataPoint
    gift_gap_points: DataPoint
    gift_gap_pct: DataPoint
    global_cues: dict[str, DataPoint]
    macro: dict[str, DataPoint]
    institutional: dict[str, DataPoint]
    headlines: list[str]
    sentiment_label: str
    sentiment_score: float
    prediction: dict[str, float]
    context_summary: str
    component_scores: dict[str, float] = field(default_factory=dict)


def safe_call(name: str, fn: Callable[[], DataPoint]) -> DataPoint:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - morning reports should continue.
        return DataPoint(name, None, "N/A", False, "unavailable", str(exc))


def pct_fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:+.2f}%"


def num_fmt(value: float | None, decimals: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def money_fmt(value: float | None, suffix: str = "") -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:,.2f}{suffix}"


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def signed_score(value: float | None, scale: float) -> float:
    if value is None or not math.isfinite(value) or scale <= 0:
        return 0.0
    return clamp(value / scale, -1.0, 1.0)


def requests_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-IN,en;q=0.9",
        }
    )
    return session


def raise_for_bad_response(response: Response) -> None:
    if response.status_code >= 400:
        raise requests.HTTPError(f"HTTP {response.status_code}: {response.text[:160]}")


def yf_last_and_previous_close(ticker: str, period: str = "7d", interval: str = "1d") -> tuple[float, float]:
    hist = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    hist = hist.dropna(subset=["Close"])
    if len(hist) < 2:
        raise ValueError(f"not enough price history for {ticker}")
    return float(hist["Close"].iloc[-1]), float(hist["Close"].iloc[-2])


def yf_percent_change(ticker: str, name: str, source: str, period: str = "7d") -> DataPoint:
    last, prev = yf_last_and_previous_close(ticker, period=period)
    change = ((last - prev) / prev) * 100.0
    return DataPoint(name, change, pct_fmt(change), True, source)


def yf_last_price(ticker: str, name: str, source: str, period: str = "5d") -> DataPoint:
    hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    hist = hist.dropna(subset=["Close"])
    if hist.empty:
        raise ValueError(f"no price history for {ticker}")
    price = float(hist["Close"].iloc[-1])
    return DataPoint(name, price, num_fmt(price), True, source)


def fetch_nifty_previous_close() -> DataPoint:
    hist = yf.Ticker("^NSEI").history(period="10d", interval="1d", auto_adjust=False)
    hist = hist.dropna(subset=["Close"])
    if hist.empty:
        raise ValueError("no Nifty 50 history from yfinance")
    close = float(hist["Close"].iloc[-1])
    return DataPoint("Nifty 50 previous close", close, num_fmt(close), True, "yfinance:^NSEI")


def extract_first_number(payload: Any, candidate_keys: tuple[str, ...]) -> float | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = key.lower().replace("_", "").replace(" ", "")
            if normalized in candidate_keys:
                parsed = parse_number(value)
                if parsed is not None:
                    return parsed
            nested = extract_first_number(value, candidate_keys)
            if nested is not None:
                return nested
    if isinstance(payload, list):
        for item in payload:
            nested = extract_first_number(item, candidate_keys)
            if nested is not None:
                return nested
    return None


def parse_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if not isinstance(value, str):
        return None
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", value)
    if not match:
        return None
    return float(match.group(0).replace(",", ""))


def fetch_gift_nifty_from_custom_url(session: requests.Session) -> DataPoint:
    url = os.getenv("GIFT_NIFTY_URL", "").strip()
    if not url:
        raise ValueError("GIFT_NIFTY_URL is not configured")

    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    raise_for_bad_response(response)

    content_type = response.headers.get("Content-Type", "")
    if "json" in content_type:
        payload = response.json()
        price = extract_first_number(
            payload,
            (
                "lastprice",
                "lasttradedprice",
                "ltp",
                "price",
                "value",
                "last",
                "close",
            ),
        )
        if price is None:
            raise ValueError("could not find GIFT Nifty price in JSON payload")
        return DataPoint("GIFT Nifty live", price, num_fmt(price), True, url)

    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(" ", strip=True)
    price = parse_number(text)
    if price is None:
        raise ValueError("could not parse GIFT Nifty price from HTML")
    return DataPoint("GIFT Nifty live", price, num_fmt(price), True, url)


def fetch_gift_nifty_from_yfinance() -> DataPoint:
    symbols = [
        symbol.strip()
        for symbol in os.getenv("GIFT_NIFTY_SYMBOLS", "").split(",")
        if symbol.strip()
    ]
    if not symbols:
        raise ValueError("GIFT_NIFTY_SYMBOLS is not configured")

    errors: list[str] = []
    for symbol in symbols:
        try:
            price = yf.Ticker(symbol).fast_info.get("last_price")
            if price and math.isfinite(float(price)):
                price_float = float(price)
                return DataPoint("GIFT Nifty live", price_float, num_fmt(price_float), True, f"yfinance:{symbol}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{symbol}: {exc}")
    raise ValueError("; ".join(errors) or "no usable GIFT Nifty yfinance symbol")


def fetch_gift_nifty(session: requests.Session) -> DataPoint:
    errors: list[str] = []
    for fn in (lambda: fetch_gift_nifty_from_custom_url(session), fetch_gift_nifty_from_yfinance):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    raise ValueError(" | ".join(errors))


def calculate_gift_gap(gift: DataPoint, nifty_close: DataPoint) -> tuple[DataPoint, DataPoint]:
    if gift.value is None or nifty_close.value is None:
        error = "missing GIFT Nifty or Nifty previous close"
        return (
            DataPoint("GIFT gap points", None, "N/A", False, "derived", error),
            DataPoint("GIFT gap percent", None, "N/A", False, "derived", error),
        )
    gap_points = gift.value - nifty_close.value
    gap_pct = (gap_points / nifty_close.value) * 100.0
    return (
        DataPoint("GIFT gap points", gap_points, f"{gap_points:+.2f}", True, "derived"),
        DataPoint("GIFT gap percent", gap_pct, pct_fmt(gap_pct), True, "derived"),
    )


def fetch_global_cues() -> dict[str, DataPoint]:
    tickers = {
        "S&P 500": ("^GSPC", "yfinance:^GSPC"),
        "Nasdaq Composite": ("^IXIC", "yfinance:^IXIC"),
        "Nikkei 225": ("^N225", "yfinance:^N225"),
        "Hang Seng": ("^HSI", "yfinance:^HSI"),
    }
    return {
        name: safe_call(name, lambda t=ticker, n=name, s=source: yf_percent_change(t, n, s))
        for name, (ticker, source) in tickers.items()
    }


def fetch_macro_indicators() -> dict[str, DataPoint]:
    return {
        "USD/INR": safe_call("USD/INR", lambda: yf_last_price("INR=X", "USD/INR", "yfinance:INR=X")),
        "Brent Crude": safe_call("Brent Crude", lambda: yf_last_price("BZ=F", "Brent Crude", "yfinance:BZ=F")),
        "US 10Y Yield": safe_call(
            "US 10Y Yield",
            lambda: yf_last_price("^TNX", "US 10Y Yield", "yfinance:^TNX"),
        ),
        "India VIX": safe_call(
            "India VIX",
            lambda: yf_last_price("^INDIAVIX", "India VIX", "yfinance:^INDIAVIX"),
        ),
        "CBOE VIX": safe_call(
            "CBOE VIX",
            lambda: yf_last_price("^VIX", "CBOE VIX", "yfinance:^VIX"),
        ),
    }


def fetch_fii_dii_from_nse(session: requests.Session) -> dict[str, DataPoint]:
    session.get("https://www.nseindia.com", timeout=REQUEST_TIMEOUT_SECONDS)
    url = "https://www.nseindia.com/api/fiidiiTradeReact"
    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    raise_for_bad_response(response)
    payload = response.json()

    rows = payload if isinstance(payload, list) else payload.get("data", [])
    if not isinstance(rows, list):
        raise ValueError("unexpected NSE FII/DII payload")

    result: dict[str, DataPoint] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        category = str(row.get("category") or row.get("name") or "").upper()
        net = (
            row.get("netValue")
            or row.get("net")
            or row.get("netInvestment")
            or row.get("net_value")
        )
        parsed = parse_number(net)
        if parsed is None:
            buy = parse_number(row.get("buyValue") or row.get("buy"))
            sell = parse_number(row.get("sellValue") or row.get("sell"))
            parsed = None if buy is None or sell is None else buy - sell
        if parsed is None:
            continue
        if "FII" in category or "FPI" in category:
            result["FII net cash"] = DataPoint("FII net cash", parsed, money_fmt(parsed, " Cr"), True, url)
        elif "DII" in category:
            result["DII net cash"] = DataPoint("DII net cash", parsed, money_fmt(parsed, " Cr"), True, url)

    if "FII net cash" not in result or "DII net cash" not in result:
        raise ValueError("could not parse both FII and DII net cash values")
    return result


def fetch_fii_dii_manual_fallback() -> dict[str, DataPoint]:
    fii = parse_number(os.getenv("FII_NET_CASH_CR"))
    dii = parse_number(os.getenv("DII_NET_CASH_CR"))
    if fii is None or dii is None:
        raise ValueError("FII_NET_CASH_CR and DII_NET_CASH_CR are not configured")
    return {
        "FII net cash": DataPoint("FII net cash", fii, money_fmt(fii, " Cr"), True, ".env"),
        "DII net cash": DataPoint("DII net cash", dii, money_fmt(dii, " Cr"), True, ".env"),
    }


def fetch_institutional_data(session: requests.Session) -> dict[str, DataPoint]:
    try:
        return fetch_fii_dii_from_nse(session)
    except Exception as nse_error:  # noqa: BLE001
        try:
            return fetch_fii_dii_manual_fallback()
        except Exception as fallback_error:  # noqa: BLE001
            return {
                "FII net cash": DataPoint("FII net cash", None, "N/A", False, "unavailable", str(nse_error)),
                "DII net cash": DataPoint("DII net cash", None, "N/A", False, "unavailable", str(fallback_error)),
            }


def fetch_newsapi_headlines(session: requests.Session) -> list[str]:
    api_key = os.getenv("NEWSAPI_KEY", "").strip()
    if not api_key:
        raise ValueError("NEWSAPI_KEY is not configured")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": '(India OR Indian) AND (market OR Nifty OR Sensex OR stocks OR RBI OR rupee)',
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": api_key,
    }
    response = session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    raise_for_bad_response(response)
    articles = response.json().get("articles", [])
    headlines = [article.get("title", "").strip() for article in articles if article.get("title")]
    return dedupe_headlines(headlines)[:10]


def fetch_moneycontrol_headlines(session: requests.Session) -> list[str]:
    url = "https://www.moneycontrol.com/news/business/markets/"
    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    raise_for_bad_response(response)
    soup = BeautifulSoup(response.text, "html.parser")
    candidates = []
    for selector in ("h2 a", "h3 a", "li.clearfix a", "a"):
        for tag in soup.select(selector):
            text = tag.get_text(" ", strip=True)
            if len(text) >= 35 and any(token in text.lower() for token in ("nifty", "sensex", "market", "stock", "rupee")):
                candidates.append(text)
        if len(candidates) >= 10:
            break
    return dedupe_headlines(candidates)[:10]


def dedupe_headlines(headlines: list[str]) -> list[str]:
    seen: set[str] = set()
    clean: list[str] = []
    for headline in headlines:
        normalized = re.sub(r"\W+", " ", headline.lower()).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            clean.append(headline)
    return clean


def fetch_headlines(session: requests.Session) -> list[str]:
    errors: list[str] = []
    for fn in (lambda: fetch_newsapi_headlines(session), lambda: fetch_moneycontrol_headlines(session)):
        try:
            headlines = fn()
            if headlines:
                return headlines[:10]
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    print(f"Warning: news unavailable: {' | '.join(errors)}")
    return []


def analyze_headline_sentiment(headlines: list[str]) -> tuple[str, float]:
    if not headlines:
        return "Neutral", 0.0

    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(headline)["compound"] for headline in headlines]
    avg_score = statistics.fmean(scores)

    if avg_score >= 0.10:
        label = "Bullish"
    elif avg_score <= -0.10:
        label = "Bearish"
    else:
        label = "Neutral"
    return label, avg_score


def score_global_cues(global_cues: dict[str, DataPoint]) -> float:
    weights = {
        "S&P 500": 0.30,
        "Nasdaq Composite": 0.30,
        "Nikkei 225": 0.20,
        "Hang Seng": 0.20,
    }
    score = 0.0
    used_weight = 0.0
    for name, weight in weights.items():
        point = global_cues.get(name)
        if point and point.value is not None:
            score += signed_score(point.value, 1.25) * weight
            used_weight += weight
    return score / used_weight if used_weight else 0.0


def score_institutional(institutional: dict[str, DataPoint]) -> float:
    fii = institutional.get("FII net cash", DataPoint("", None, "", False, "")).value
    dii = institutional.get("DII net cash", DataPoint("", None, "", False, "")).value
    if fii is None and dii is None:
        return 0.0

    combined = (fii or 0.0) + 0.60 * (dii or 0.0)
    return signed_score(combined, 3500.0)


def score_macro(macro: dict[str, DataPoint]) -> float:
    # Higher crude, higher USD/INR, higher US yields, and higher VIX are headwinds for India.
    usd_inr = macro.get("USD/INR")
    brent = macro.get("Brent Crude")
    us10y = macro.get("US 10Y Yield")
    india_vix = macro.get("India VIX")
    us_vix = macro.get("CBOE VIX")

    score = 0.0
    weights_used = 0.0

    if usd_inr and usd_inr.value is not None:
        usd_change = safe_call(
            "USD/INR change",
            lambda: yf_percent_change("INR=X", "USD/INR change", "yfinance:INR=X"),
        ).value
        score += -signed_score(usd_change, 0.50) * 0.25
        weights_used += 0.25
    if brent and brent.value is not None:
        brent_change = safe_call(
            "Brent change",
            lambda: yf_percent_change("BZ=F", "Brent change", "yfinance:BZ=F"),
        ).value
        score += -signed_score(brent_change, 2.00) * 0.25
        weights_used += 0.25
    if us10y and us10y.value is not None:
        us10y_change = safe_call(
            "US 10Y change",
            lambda: yf_percent_change("^TNX", "US 10Y change", "yfinance:^TNX"),
        ).value
        score += -signed_score(us10y_change, 3.00) * 0.20
        weights_used += 0.20
    if india_vix and india_vix.value is not None:
        vix_change = safe_call(
            "India VIX change",
            lambda: yf_percent_change("^INDIAVIX", "India VIX change", "yfinance:^INDIAVIX"),
        ).value
        score += -signed_score(vix_change, 4.00) * 0.15
        weights_used += 0.15
    elif us_vix and us_vix.value is not None:
        us_vix_change = safe_call(
            "CBOE VIX change",
            lambda: yf_percent_change("^VIX", "CBOE VIX change", "yfinance:^VIX"),
        ).value
        score += -signed_score(us_vix_change, 5.00) * 0.15
        weights_used += 0.15

    return score / weights_used if weights_used else 0.0


def weighted_prediction(
    gift_gap_pct: DataPoint,
    global_cues: dict[str, DataPoint],
    institutional: dict[str, DataPoint],
    sentiment_score: float,
    macro: dict[str, DataPoint],
) -> tuple[dict[str, float], dict[str, float]]:
    component_scores = {
        "GIFT Nifty": signed_score(gift_gap_pct.value, 0.75),
        "Global cues": score_global_cues(global_cues),
        "FII/DII": score_institutional(institutional),
        "News sentiment": signed_score(sentiment_score, 0.50),
        "Macro": score_macro(macro),
    }
    weights = {
        "GIFT Nifty": 0.40,
        "Global cues": 0.20,
        "FII/DII": 0.20,
        "News sentiment": 0.10,
        "Macro": 0.10,
    }

    raw_score = sum(component_scores[name] * weights[name] for name in weights)
    confidence = abs(raw_score)
    flat_probability = clamp(45.0 - confidence * 30.0, 12.0, 55.0)
    directional_probability = 100.0 - flat_probability

    gap_up = directional_probability * (1.0 + raw_score) / 2.0
    gap_down = directional_probability - gap_up

    probabilities = {
        "GAP UP": round(clamp(gap_up, 0.0, 100.0), 1),
        "GAP DOWN": round(clamp(gap_down, 0.0, 100.0), 1),
        "FLAT": round(flat_probability, 1),
    }
    total = sum(probabilities.values())
    if total != 100.0:
        probabilities["FLAT"] = round(probabilities["FLAT"] + (100.0 - total), 1)
    return probabilities, component_scores


def make_context_summary(report: MarketReport) -> str:
    gift = report.gift_gap_pct.display
    global_score = report.component_scores.get("Global cues", 0.0)
    fii = report.institutional.get("FII net cash", DataPoint("", None, "N/A", False, "")).display
    dii = report.institutional.get("DII net cash", DataPoint("", None, "N/A", False, "")).display

    if report.prediction["GAP UP"] > report.prediction["GAP DOWN"] and report.prediction["GAP UP"] > report.prediction["FLAT"]:
        bias = "positive"
    elif report.prediction["GAP DOWN"] > report.prediction["GAP UP"] and report.prediction["GAP DOWN"] > report.prediction["FLAT"]:
        bias = "negative"
    else:
        bias = "balanced"

    global_phrase = "supportive" if global_score > 0.15 else "weak" if global_score < -0.15 else "mixed"
    return (
        f"The setup is {bias}: GIFT Nifty implies a {gift} opening gap, "
        f"global cues are {global_phrase}, FII/DII cash flow is {fii}/{dii}, "
        f"and news sentiment is {report.sentiment_label.lower()}."
    )


def build_report() -> MarketReport:
    load_dotenv()
    session = requests_session()

    nifty_close = safe_call("Nifty 50 previous close", fetch_nifty_previous_close)
    gift = safe_call("GIFT Nifty live", lambda: fetch_gift_nifty(session))
    gap_points, gap_pct = calculate_gift_gap(gift, nifty_close)

    global_cues = fetch_global_cues()
    macro = fetch_macro_indicators()
    institutional = fetch_institutional_data(session)
    headlines = fetch_headlines(session)
    sentiment_label, sentiment_score = analyze_headline_sentiment(headlines)
    prediction, component_scores = weighted_prediction(gap_pct, global_cues, institutional, sentiment_score, macro)

    report = MarketReport(
        generated_at=datetime.now(IST),
        nifty_previous_close=nifty_close,
        gift_nifty=gift,
        gift_gap_points=gap_points,
        gift_gap_pct=gap_pct,
        global_cues=global_cues,
        macro=macro,
        institutional=institutional,
        headlines=headlines,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        prediction=prediction,
        context_summary="",
        component_scores=component_scores,
    )
    report.context_summary = make_context_summary(report)
    return report


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def print_point(label: str, point: DataPoint) -> None:
    status = "" if point.ok else "  [unavailable]"
    print(f"{label:<24} {point.display:>14}{status}")


def print_dashboard(report: MarketReport) -> None:
    print("=" * 72)
    print("PRE-MARKET INDIAN MARKET PREDICTOR")
    print(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("=" * 72)

    print_section("GIFT Nifty")
    print_point("Nifty prev close", report.nifty_previous_close)
    print_point("GIFT Nifty live", report.gift_nifty)
    print_point("Implied gap points", report.gift_gap_points)
    print_point("Implied gap percent", report.gift_gap_pct)

    print_section("Global Markets")
    for name, point in report.global_cues.items():
        print_point(name, point)

    print_section("Macro Indicators")
    for name, point in report.macro.items():
        print_point(name, point)

    print_section("Institutional Cash Market")
    for name, point in report.institutional.items():
        print_point(name, point)

    print_section("News Sentiment")
    print(f"Aggregate sentiment     {report.sentiment_label:>14} ({report.sentiment_score:+.3f})")
    if report.headlines:
        for idx, headline in enumerate(report.headlines[:10], start=1):
            print(f"{idx:>2}. {headline}")
    else:
        print("No headlines available.")

    print_section("Weighted Signal Scores")
    for name, score in report.component_scores.items():
        print(f"{name:<24} {score:+.3f}")

    print_section("Prediction")
    print(f"Probability of GAP UP:   {report.prediction['GAP UP']:>5.1f}%")
    print(f"Probability of GAP DOWN: {report.prediction['GAP DOWN']:>5.1f}%")
    print(f"Probability of FLAT:     {report.prediction['FLAT']:>5.1f}%")
    print(f"AI summary: {report.context_summary}")
    print()
    print("Disclaimer: This is a quantitative pre-market model, not investment advice.")


def sleep_until_845_ist() -> None:
    now = datetime.now(IST)
    target = now.replace(hour=8, minute=45, second=0, microsecond=0)
    if now >= target:
        return
    seconds = (target - now).total_seconds()
    print(f"Waiting until {target.strftime('%Y-%m-%d %H:%M:%S %Z')} ({int(seconds)} seconds).")
    time.sleep(seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-market Nifty 50 opening predictor.")
    parser.add_argument(
        "--wait-until-845",
        action="store_true",
        help="Sleep until 08:45 IST before fetching data. Useful for a long-running job.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.wait_until_845:
        sleep_until_845_ist()
    report = build_report()
    print_dashboard(report)


if __name__ == "__main__":
    main()
