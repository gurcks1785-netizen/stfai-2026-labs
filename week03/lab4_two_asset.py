"""
Week 03 Lab 4 – AAPL / KO 두 자산 포트폴리오 분석
연율화 통계 · 포트폴리오 표 · 최소분산 · 효율적 프론티어
상관계수 시나리오(-0.5, 0, 0.5) 곡선 비교
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

OUT_DIR      = Path(__file__).parent
TRADING_DAYS = 252


# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 수집
# ══════════════════════════════════════════════════════════════════════════════

def fetch_returns(tickers: list[str],
                  start: str = "2022-01-01",
                  end:   str = "2024-12-31") -> pd.DataFrame:
    """yfinance로 조정종가를 받아 일간 단순수익률 DataFrame을 반환한다."""
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    raw.index = raw.index.tz_localize(None)
    returns = raw.pct_change().dropna()
    returns.columns = tickers
    return returns


# ══════════════════════════════════════════════════════════════════════════════
# 2. 연율화 통계
# ══════════════════════════════════════════════════════════════════════════════

def annualized_stats(returns: pd.DataFrame) -> dict:
    """
    연율화 기대수익률, 표준편차, 상관계수를 계산한다.

    E(r)_ann  = E(r)_daily × 252
    σ_ann     = σ_daily × √252
    ρ         = Corr(r_A, r_B)   (일간 상관계수 = 연율화 상관계수로 동일)
    """
    er  = returns.mean() * TRADING_DAYS
    vol = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    corr_matrix = returns.corr()
    rho = corr_matrix.iloc[0, 1]
    cov_ann = rho * vol.iloc[0] * vol.iloc[1]
    return {"er": er, "vol": vol, "rho": rho, "cov_ann": cov_ann}


# ══════════════════════════════════════════════════════════════════════════════
# 3. 포트폴리오 통계
# ══════════════════════════════════════════════════════════════════════════════

def port_stats(w: float | np.ndarray,
               ea: float, eb: float,
               va: float, vb: float,
               cov: float):
    """
    E(P) = w·E(A) + (1-w)·E(B)
    σ²(P)= w²σ²(A) + (1-w)²σ²(B) + 2w(1-w)Cov(A,B)
    """
    ep = w * ea + (1 - w) * eb
    sp = np.sqrt(w**2 * va + (1-w)**2 * vb + 2*w*(1-w)*cov)
    return ep, sp


def min_var_weight(va: float, vb: float, cov: float) -> float:
    """
    최소분산 비중 공식:
    w* = (σ²(B) - Cov) / (σ²(A) + σ²(B) - 2·Cov)
    """
    return (vb - cov) / (va + vb - 2*cov)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 출력 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

SEP = "=" * 68

def print_section(title: str):
    print(f"\n{SEP}\n{title}\n{SEP}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 데이터 ──────────────────────────────────────────────────────────────
    tickers = ["AAPL", "KO"]
    rets    = fetch_returns(tickers)
    stats   = annualized_stats(rets)

    ea  = stats["er"]["AAPL"]
    eb  = stats["er"]["KO"]
    sa  = stats["vol"]["AAPL"]
    sb  = stats["vol"]["KO"]
    rho = stats["rho"]
    va, vb = sa**2, sb**2
    cov = stats["cov_ann"]           # = rho * sa * sb

    # ── [1] 연율화 통계 ─────────────────────────────────────────────────────
    print_section("STEP 1. 연율화 통계  (연율화 거래일 = 252)")
    print(f"  기간: 2022-01-01 ~ 2024-12-31  |  관측 거래일: {len(rets)}")
    print()
    print(f"  E(r)_ann  = E(r)_daily × 252")
    print(f"    AAPL : {rets['AAPL'].mean():.6f} × 252 = {ea:.4f}  ({ea*100:.2f}%)")
    print(f"    KO   : {rets['KO'].mean():.6f} × 252 = {eb:.4f}  ({eb*100:.2f}%)")
    print()
    print(f"  σ_ann     = σ_daily × √252")
    print(f"    AAPL : {rets['AAPL'].std(ddof=1):.6f} × {np.sqrt(252):.4f} = {sa:.4f}  ({sa*100:.2f}%)")
    print(f"    KO   : {rets['KO'].std(ddof=1):.6f} × {np.sqrt(252):.4f} = {sb:.4f}  ({sb*100:.2f}%)")
    print()
    print(f"  상관계수  ρ(AAPL, KO) = {rho:.6f}")
    print(f"  공분산    Cov_ann     = ρ·σ(A)·σ(B) = {rho:.4f}×{sa:.4f}×{sb:.4f} = {cov:.6f}")

    # ── [2] 포트폴리오 표 ────────────────────────────────────────────────────
    print_section("STEP 2. 포트폴리오 표  (w = AAPL 비중, 0.05 간격)")
    print(f"  E(P) = w·E(A) + (1-w)·E(B)")
    print(f"  σ(P) = √[ w²σ²(A) + (1-w)²σ²(B) + 2w(1-w)Cov(A,B) ]")
    print()
    print(f"  {'w(AAPL)':>9}  {'1-w(KO)':>8}  {'E(P)%':>9}  {'σ(P)%':>9}")
    print("  " + "-" * 42)

    weights   = np.round(np.arange(0.0, 1.01, 0.05), 2)
    ep_list   = []
    sp_list   = []
    for w in weights:
        ep, sp = port_stats(w, ea, eb, va, vb, cov)
        ep_list.append(ep)
        sp_list.append(sp)
        print(f"  {w:>9.2f}  {1-w:>8.2f}  {ep*100:>9.4f}  {sp*100:>9.4f}")

    # ── [3] 최소분산 ─────────────────────────────────────────────────────────
    print_section("STEP 3. 최소분산 포트폴리오")
    w_star    = min_var_weight(va, vb, cov)
    ep_star, sp_star = port_stats(w_star, ea, eb, va, vb, cov)

    print()
    print("  공식:")
    print("    w* = (σ²(B) − Cov) / (σ²(A) + σ²(B) − 2·Cov)")
    print(f"       = ({vb:.6f} − ({cov:.6f}))")
    print(f"         / ({va:.6f} + {vb:.6f} − 2×({cov:.6f}))")
    num = vb - cov
    den = va + vb - 2*cov
    print(f"       = {num:.6f} / {den:.6f}")
    print(f"       = {w_star:.6f}  → AAPL {w_star*100:.2f}%,  KO {(1-w_star)*100:.2f}%")
    print()
    print(f"  E(P*) = {ep_star*100:.4f}%")
    print(f"  σ(P*) = {sp_star*100:.4f}%")

    # 표 최소값과 대조
    idx_min  = int(np.argmin(sp_list))
    w_tbl    = weights[idx_min]
    sp_tbl   = sp_list[idx_min]
    print()
    print(f"  [확인] 표(0.05 간격) 최소: w = {w_tbl:.2f},  σ = {sp_tbl*100:.4f}%")
    print(f"         공식 연속 최소  : w = {w_star:.4f},  σ = {sp_star*100:.4f}%")
    gap = abs(w_star - w_tbl)
    print(f"         비중 차이: {gap:.4f}  (0.05 격자 오차 이내 {'✓' if gap <= 0.025 else '✗'})")

    # ── [4] 그래프 ───────────────────────────────────────────────────────────
    print_section("STEP 4. 그래프 생성")

    w_fine = np.linspace(0.0, 1.0, 500)

    fig, ax = plt.subplots(figsize=(9, 6))

    # 실제 상관계수 곡선
    ep_f = w_fine * ea + (1 - w_fine) * eb
    sp_f = np.sqrt(w_fine**2*va + (1-w_fine)**2*vb + 2*w_fine*(1-w_fine)*cov)
    ax.plot(sp_f*100, ep_f*100, color="#1f77b4", linewidth=2.5,
            label=f"실제 ρ = {rho:.3f}")

    # 0.05 간격 점
    ax.scatter(np.array(sp_list)*100, np.array(ep_list)*100,
               color="#1f77b4", s=30, zorder=5, alpha=0.8)

    # 최소분산점
    ax.scatter(sp_star*100, ep_star*100, color="red", s=120,
               marker="*", zorder=6, label=f"Min-Var  w*={w_star:.3f}")

    # 가상 상관계수 시나리오
    scenario_styles = {
        -0.5: ("#d62728", "--"),
         0.0: ("#2ca02c", "-."),
         0.5: ("#9467bd", ":"),
    }
    for rho_hyp, (color, ls) in scenario_styles.items():
        cov_hyp = rho_hyp * sa * sb
        sp_hyp  = np.sqrt(w_fine**2*va + (1-w_fine)**2*vb
                          + 2*w_fine*(1-w_fine)*cov_hyp)
        ax.plot(sp_hyp*100, ep_f*100, color=color, linewidth=1.8,
                linestyle=ls, label=f"가정 ρ = {rho_hyp:+.1f}")

    # 개별 자산
    ax.scatter(sa*100, ea*100, color="black", s=80, marker="D", zorder=7)
    ax.scatter(sb*100, eb*100, color="black", s=80, marker="D", zorder=7)
    ax.annotate("AAPL (w=1)", (sa*100, ea*100),
                textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.annotate("KO (w=0)",   (sb*100, eb*100),
                textcoords="offset points", xytext=(6, -12), fontsize=9)

    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
    ax.set_xlabel("Standard Deviation  σ(P)", fontsize=11)
    ax.set_ylabel("Expected Return  E(P)", fontsize=11)
    ax.set_title("Two-Asset Frontier: AAPL & KO (2022–2024)\n"
                 "with Hypothetical Correlation Scenarios", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    img_path = OUT_DIR / "lab4_frontier.png"
    fig.savefig(img_path, dpi=150)
    print(f"\n  그래프 저장 완료: {img_path}")

    # ── [5] 상관계수 영향 설명 ───────────────────────────────────────────────
    print_section("STEP 5. 상관계수가 프론티어 곡선에 미치는 영향")
    print("""
  1) ρ가 낮을수록(음수 방향일수록) 곡선이 왼쪽으로 휘어져
     동일한 기대수익률에서 더 낮은 표준편차를 달성할 수 있다.

  2) ρ = -1 이면 분산이 완전히 제거되는 포트폴리오가 존재해
     곡선이 σ = 0 꼭짓점을 갖는 두 직선으로 펼쳐지고,
     ρ = +1 이면 두 자산을 잇는 직선이 되어 분산 효과가 사라진다.

  3) 실제 ρ(AAPL, KO) ≈ {rho:.3f} 은 0 에 가까워 곡선이
     ρ = 0 가정선과 거의 겹치며, 적절한 혼합으로 변동성을
     두 자산의 가중평균보다 낮출 수 있다.
""".format(rho=rho))


if __name__ == "__main__":
    main()
