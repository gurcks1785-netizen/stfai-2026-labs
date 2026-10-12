"""
Week 03 Lab 5 – 분산투자 효과 실험
미국·한국 주식 각 20개, N=1~20 동일가중 포트폴리오 연변동성 시뮬레이션
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR      = Path(__file__).parent
TRADING_DAYS = 252
START, END   = "2022-01-01", "2024-12-31"


# ══════════════════════════════════════════════════════════════════════════════
# 1. 종목 정의  (업종 중복 없이 각 20개)
# ══════════════════════════════════════════════════════════════════════════════

# 선정 기준:
#  - S&P 500 / KOSPI 대표 대형주, 업종(GICS 세부 업종) 중복 배제
#  - 2022-01-01 이전 상장·거래 가능한 종목

US_STOCKS = {
    "AAPL":  "소비자 가전",
    "MSFT":  "엔터프라이즈 소프트웨어",
    "NVDA":  "반도체",
    "GOOGL": "인터넷 광고",
    "AMZN":  "전자상거래/클라우드",
    "JPM":   "상업은행",
    "JNJ":   "제약",
    "XOM":   "석유·가스 탐사",
    "WMT":   "종합소매",
    "NEE":   "전력유틸리티",
    "LMT":   "항공우주·방산",
    "CAT":   "건설기계",
    "LIN":   "산업가스",
    "PLD":   "산업용 리츠",
    "VZ":    "유선·무선통신",
    "KO":    "음료",
    "MDT":   "의료기기",
    "GS":    "투자은행",
    "FCX":   "구리 광업",
    "DIS":   "엔터테인먼트·미디어",
}

KR_STOCKS = {
    "005930.KS": "소비자 가전·반도체 (삼성전자)",
    "000660.KS": "메모리 반도체 (SK하이닉스)",
    "373220.KS": "EV 배터리 (LG에너지솔루션)",
    "207940.KS": "바이오의약품 위탁생산 (삼성바이오로직스)",
    "005380.KS": "완성차 (현대차)",
    "005490.KS": "철강 (POSCO홀딩스)",
    "105560.KS": "상업은행 (KB금융)",
    "068270.KS": "바이오시밀러 (셀트리온)",
    "035420.KS": "인터넷 플랫폼 (NAVER)",
    "012450.KS": "항공우주·방산 (한화에어로스페이스)",
    "010950.KS": "정유 (에쓰오일)",
    "033780.KS": "담배 (KT&G)",
    "015760.KS": "전력유틸리티 (한국전력)",
    "090430.KS": "화장품 (아모레퍼시픽)",
    "097950.KS": "식품가공 (CJ제일제당)",
    "010130.KS": "비철금속 (고려아연)",
    "035720.KS": "소셜미디어·핀테크 (카카오)",
    "028260.KS": "건설·종합상사 (삼성물산)",
    "009150.KS": "전자부품 (삼성전기)",
    "066570.KS": "가전 (LG전자)",
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. 데이터 수집
# ══════════════════════════════════════════════════════════════════════════════

def fetch_returns(ticker_dict: dict, label: str) -> pd.DataFrame:
    """
    조정종가를 내려받아 일간 단순수익률 DataFrame을 반환한다.
    데이터 비율이 80% 미만인 종목은 제외하고, 잔여 NaN은 0으로 처리한다.
    """
    print(f"  [{label}] 데이터 다운로드 중...")
    tickers = list(ticker_dict.keys())
    raw = yf.download(tickers, start=START, end=END,
                      auto_adjust=True, progress=False)["Close"]
    raw.index = raw.index.tz_localize(None)

    rets = raw.pct_change()
    # 데이터 충분한 종목만 유지 (전체 기간의 80% 이상)
    min_obs = int(0.80 * len(rets))
    valid   = rets.columns[rets.notna().sum() >= min_obs].tolist()
    rets    = rets[valid].fillna(0)
    rets    = rets.iloc[1:]        # 첫 행 NaN 제거

    n_valid = len(valid)
    print(f"         유효 종목 {n_valid}개 / 요청 {len(tickers)}개")
    if n_valid < len(tickers):
        excluded = [t for t in tickers if t not in valid]
        print(f"         제외: {excluded}")
    return rets


# ══════════════════════════════════════════════════════════════════════════════
# 3. 시뮬레이션
# ══════════════════════════════════════════════════════════════════════════════

def simulate_diversification(rets: pd.DataFrame,
                             n_max: int = 20,
                             n_sim: int = 200,
                             seed: int = 42) -> np.ndarray:
    """
    N = 1 ~ n_max 각각에 대해:
      - n_sim 번 무작위로 N개 종목을 뽑아 1/N 동일가중 포트폴리오 연변동성을 계산
      - 평균값을 반환  (shape: n_max)
    """
    rng    = np.random.default_rng(seed)
    stocks = rets.columns.tolist()
    n_stocks = len(stocks)
    avg_vols = np.zeros(n_max)

    for n in range(1, n_max + 1):
        n_pick = min(n, n_stocks)       # 보유 종목 수보다 많이 뽑지 않음
        vols = np.zeros(n_sim)
        for s in range(n_sim):
            chosen   = rng.choice(stocks, size=n_pick, replace=False)
            port_ret = rets[chosen].mean(axis=1)      # 동일가중 = 평균
            vols[s]  = port_ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
        avg_vols[n - 1] = vols.mean()

    return avg_vols


# ══════════════════════════════════════════════════════════════════════════════
# 4. 수렴 수준 추정
# ══════════════════════════════════════════════════════════════════════════════

def estimate_asymptote(avg_vols: np.ndarray, last_k: int = 5) -> float:
    """N이 가장 클 때 마지막 k개 평균을 수렴 추정치로 사용한다."""
    return avg_vols[-last_k:].mean()


# ══════════════════════════════════════════════════════════════════════════════
# 5. 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    SEP = "=" * 68

    # ── 종목 목록 출력 ──────────────────────────────────────────────────────
    print(SEP)
    print("종목 선정 (업종 중복 배제)")
    print(SEP)
    print("\n[미국 20개]")
    for t, s in US_STOCKS.items():
        print(f"  {t:<8}  {s}")
    print("\n[한국 20개]")
    for t, s in KR_STOCKS.items():
        print(f"  {t:<14}  {s}")

    # ── 데이터 수집 ─────────────────────────────────────────────────────────
    print(f"\n{SEP}\n데이터 수집\n{SEP}")
    us_rets = fetch_returns(US_STOCKS, "미국")
    kr_rets = fetch_returns(KR_STOCKS, "한국")

    n_max = 20

    # ── 시뮬레이션 200회 ────────────────────────────────────────────────────
    print(f"\n{SEP}\n시뮬레이션 (N_SIM = 200)\n{SEP}")
    us_200 = simulate_diversification(us_rets, n_max=n_max, n_sim=200, seed=42)
    kr_200 = simulate_diversification(kr_rets, n_max=n_max, n_sim=200, seed=42)

    # ── 시뮬레이션 20회 ─────────────────────────────────────────────────────
    print(f"\n{SEP}\n시뮬레이션 (N_SIM = 20)\n{SEP}")
    us_20  = simulate_diversification(us_rets, n_max=n_max, n_sim=20,  seed=42)
    kr_20  = simulate_diversification(kr_rets, n_max=n_max, n_sim=20,  seed=42)

    # ── 결과 표 ─────────────────────────────────────────────────────────────
    print(f"\n{SEP}\n연변동성 평균 (200회 시뮬레이션, 단위: %)\n{SEP}")
    print(f"  {'N':>3}  {'미국(200)':>10}  {'한국(200)':>10}  "
          f"{'미국(20)':>9}  {'한국(20)':>9}")
    print("  " + "-" * 46)
    ns = np.arange(1, n_max + 1)
    for i, n in enumerate(ns):
        print(f"  {n:>3}  {us_200[i]*100:>10.4f}  {kr_200[i]*100:>10.4f}  "
              f"{us_20[i]*100:>9.4f}  {kr_20[i]*100:>9.4f}")

    # ── 수렴 수준 ────────────────────────────────────────────────────────────
    us_asym = estimate_asymptote(us_200)
    kr_asym = estimate_asymptote(kr_200)

    print(f"\n{SEP}\n수렴 수준 추정 (N=16~20 평균)\n{SEP}")
    print(f"  미국: {us_asym*100:.4f}%")
    print(f"  한국: {kr_asym*100:.4f}%")
    print(f"""
  [의미]
  수렴 수준은 종목 수를 아무리 늘려도 제거되지 않는 '체계적 위험(systematic risk)'
  즉 시장 전체의 공통 변동성을 나타낸다.

  - 개별 주식의 위험은 '체계적 위험 + 비체계적 위험(idiosyncratic risk)'으로 분해된다.
  - N이 커질수록 비체계적 위험은 분산투자로 소거되고, 체계적 위험만 남는다.
  - 이론적으로 σ²_P → ρ̄·σ̄²  (ρ̄: 평균 쌍별 상관계수, σ̄²: 평균 개별 분산).
  - 미국({us_asym*100:.2f}%) vs 한국({kr_asym*100:.2f}%): 수렴 수준이 높은 시장이
    평균 종목 간 상관관계가 높거나 시장 공통 요인의 영향이 크다는 의미이다.
""")

    # ── 200 vs 20 비교 ───────────────────────────────────────────────────────
    print(f"\n{SEP}\n반복 횟수 200 vs 20 비교\n{SEP}")
    us_diff = np.abs(us_200 - us_20)
    kr_diff = np.abs(kr_200 - kr_20)
    print(f"  미국 최대 절대 차이: {us_diff.max()*100:.4f}%  (N={us_diff.argmax()+1})")
    print(f"  한국 최대 절대 차이: {kr_diff.max()*100:.4f}%  (N={kr_diff.argmax()+1})")
    print(f"""
  [해석]
  - N_SIM=20이면 샘플 수가 적어 표본 평균의 표준오차가 크고,
    곡선에 '들쭉날쭉한 노이즈'가 뚜렷하게 나타난다.
  - N_SIM=200이면 큰 수의 법칙(Law of Large Numbers)에 의해
    표본 평균이 참 평균에 수렴해 곡선이 부드러워진다.
  - N이 작을 때(N=1~3) 샘플 간 변동이 크기 때문에 차이가
    특히 두드러지며, N이 커질수록 반복 횟수 효과는 줄어든다.
""")

    # ── 그래프 ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, (label, v200, v20, asym, color) in zip(axes, [
        ("미국 (US)",  us_200, us_20, us_asym, "#1f77b4"),
        ("한국 (KR)",  kr_200, kr_20, kr_asym, "#d62728"),
    ]):
        ax.plot(ns, v200*100, color=color,       linewidth=2.2,
                label="200회 시뮬레이션")
        ax.plot(ns, v20*100,  color=color,       linewidth=1.2,
                linestyle="--", alpha=0.6, label="20회 시뮬레이션")
        ax.axhline(asym*100, color="gray", linewidth=1.2, linestyle=":",
                   label=f"수렴 추정 {asym*100:.2f}%")

        ax.set_xlabel("종목 수 N", fontsize=11)
        ax.set_ylabel("연변동성 (%)", fontsize=11)
        ax.set_title(f"분산투자 효과 – {label}", fontsize=12)
        ax.set_xticks(ns)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    plt.suptitle("Diversification Effect: US vs Korea  (2022–2024)",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    img_path = OUT_DIR / "lab5_diversification.png"
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    print(f"  그래프 저장 완료: {img_path}")


if __name__ == "__main__":
    main()
