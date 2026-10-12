"""
Week 03 Lab 1 – KOSPI 단순수익률 vs 로그수익률
"""

import yfinance as yf
import numpy as np
import pandas as pd


# ── 1. 데이터 수집 ──────────────────────────────────────────────────────────

def fetch_kospi(start: str = "2022-01-01", end: str = "2023-12-31") -> pd.Series:
    """yfinance로 KOSPI(^KS11) 조정종가(Adj Close) 일간 시계열을 반환한다."""
    ticker = yf.Ticker("^KS11")
    df = ticker.history(start=start, end=end, auto_adjust=True)
    close = df["Close"].dropna()
    close.index = close.index.tz_localize(None)          # timezone 제거
    close.name = "Adj_Close"
    return close


# ── 2. 수익률 계산 ──────────────────────────────────────────────────────────

def calc_returns(prices: pd.Series) -> pd.DataFrame:
    """단순수익률(simple)과 로그수익률(log)을 계산해 DataFrame으로 반환한다."""
    simple = prices.pct_change().dropna()
    log    = np.log(prices / prices.shift(1)).dropna()

    returns = pd.DataFrame({
        "simple_return": simple,
        "log_return":    log,
    })
    returns["diff"] = (returns["simple_return"] - returns["log_return"]).abs()
    return returns


# ── 3. 메인 ────────────────────────────────────────────────────────────────

def main():
    prices  = fetch_kospi()
    returns = calc_returns(prices)

    n = len(returns)
    max_date = returns["diff"].idxmax()
    row      = returns.loc[max_date]

    print("=" * 60)
    print("KOSPI(^KS11) 일간 수익률 분석  (2022-01-01 ~ 2023-12-31)")
    print("=" * 60)

    print(f"\n[1] 총 관측치 수: {n} 거래일\n")

    print("[2] 단순수익률 vs 로그수익률 – 차이가 가장 큰 날")
    header = f"{'항목':<20} {'값':>15}"
    sep    = "-" * 38
    print(sep)
    print(header)
    print(sep)
    print(f"{'날짜':<20} {str(max_date.date()):>15}")
    print(f"{'단순수익률':<20} {row['simple_return']:>15.6f}")
    print(f"{'로그수익률':<20} {row['log_return']:>15.6f}")
    print(f"{'절대 차이':<20} {row['diff']:>15.6f}")
    print(sep)

    print("\n[3] 설명")
    print(
        "  단순수익률과 로그수익률의 차이는 수익률의 크기(|r|)가 클수록 커지므로,"
        f" {max_date.date()}는 분석 기간 중 가격 변동이 가장 극단적인 날이어서 차이가 가장 크다."
    )
    print()


if __name__ == "__main__":
    main()
