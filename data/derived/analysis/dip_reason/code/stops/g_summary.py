"""칼손절 + 40분봉 RSI 조건 연구 종합(g_summary, 2026-10-08).

사용자 가설(10/7): "40분봉 RSI 30 이하(이미 과매도)면 손절하면 안 좋다."
8계열(g_fixed g_waterfall g_grind g_structure g_comove g_htfscan g_rsiboth g_model) 연구·반박 검증 결과를 묶는다.
반박을 견딘 규칙은 g_fixed 최선안 하나뿐이라 OR 결합은 하지 않고, 그 규칙만 정의에서 다시 짜서 재확인한다
(새 매개변수 선택 없음). 같이 돌리는 것: 같은 손절 조건 없음, 반대 조건(conf<=40), 사용자 깨어 있는 시간(KST 8~24시) 한정.
계열 표의 숫자는 각 계열 결과 파일·반박 검증 파일(results/stops/g_*.txt, verify_g_*.txt)에서 옮겼다.
사용: py stops/g_summary.py   (결과: ../results/stops/g_summary.txt)
"""
import os, re, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import compare, stop_audit, load, base_trades, run, htf_rsi, UNIV  # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops"))
OUT = os.path.join(RES, "g_summary.txt")
U78 = UNIV["사용자78"]


# ---------------------------------------------------------------- 생존 규칙(정의에서 다시 씀)
def mk(p=0.05, g=40, gate="gt", awake=False):
    """고정 종가 -p 첫 신호 손절. gate: 'gt' conf>g일 때만 판다, 'le' conf<=g일 때만(반대 조건), None 조건 없음.
    첫 신호 = 그 거래에서 c[i-1] <= ent*(1-p)가 처음 성립한 봉 i. 그때 조건이 꺼져 있으면 그 거래는 끝까지 손절 안 함.
    conf[i] = stop_lab.htf_rsi(A, 40, 14)[1][i] (봉 i 시작까지 완성된 마지막 40분봉 RSI14). NaN이면 조건 꺼짐.
    awake: 봉 i 시작 시각이 KST 8~23시일 때만 판다(첫 신호가 밤이면 그 거래는 8시 정리까지 들고 간다)."""
    def make_rule(A):
        c = A["c"]
        cf = htf_rsi(A, 40, 14)[1]
        hk = ((A["tk"] // 10**9 // 3600) + 9) % 24

        def rule(i, ei, ent):
            thr = ent * (1 - p)
            if not c[i - 1] <= thr:
                return False
            if i - 1 > ei and c[ei:i - 1].min() <= thr:      # 앞 봉에서 이미 첫 신호가 났음
                return False
            if awake and hk[i] < 8:
                return False
            if gate is None:
                return True
            v = cf[i]
            if v != v:
                return False
            return bool(v > g) if gate == "gt" else bool(v <= g)
        return rule
    return make_rule


# ---------------------------------------------------------------- 계열별 최선안(연구·반박 검증 파일에서 옮김, 사용자78 차이)
# (계열, 최선안, 선택(e24,eReg,l24,lReg), 조건없음, 반대조건, 헛손절(e,l), 손절수(e24,l24), 반박, 반박 이유, 시험 수, 네 칸 플러스 수)
FAM = [
    ("고정·추적", "종가 -5% 첫 신호 & conf40>40", (0.619, 0.432, 0.172, 0.072), (-0.191, -0.348, 0.202, -0.106),
     (-0.726, -0.761, 0.030, -0.178), (0.113, 0.375), (53, 16), False, "반박 실패(근거 약함)", 99, 27),
    ("폭포(하락 속도)", "30분 -3% & 매수가 이하 & dev40>35", (0.775, 0.142, 0.170, 0.074), (-0.651, -0.239, -0.159, -0.110),
     (-0.820, -0.210, -0.260, -0.170), (0.198, 0.312), (107, 16), True, "이웃 g30~33 late24 음수", 97, 28),
    ("흘러내림(시간)", "프리·애프터 매수 10:00ET 반등 없음 & 30<dev40<=45", (0.450, 0.0, -0.210, 0.0), (0.160, 0.0, -0.240, 0.0),
     (-0.190, 0.0, -0.190, 0.0), (0.115, 0.324), (78, 34), True, "late24 -0.21(t -1.97)", 94, 1),
    ("구조 붕괴", "직전 정규장 저가 종가 이탈 & conf40>30", (0.734, 1.166, -0.584, -0.245), (0.660, 1.090, -0.580, -0.240),
     (-0.230, -0.480, -0.280, -0.200), (0.326, 0.331), (437, 122), True, "late 두 칸 음수, 조건 거의 안 걸림", 75, 0),
    ("묶음 동반 하락", "그룹 중앙 30분 <= -2% & dev40>30", (1.615, 1.756, -0.930, -0.072), (-0.095, 0.486, -1.344, -0.261),
     (-2.010, -1.381, -0.637, -0.199), (0.470, 0.567), (488, 121), True, "late24 -0.93, 이웃 11개 late24 음수", 83, 28),
    ("큰 봉 길이 지도", "본전+2%|깨어 & 80분 RSI14 dev>30", (0.790, 0.280, -0.440, -0.050), (0.630, 0.430, -0.470, -0.090),
     (-0.210, 0.060, -0.050, -0.040), (0.413, 0.605), (155, 43), True, "late 두 칸 음수", 97, 23),
    ("기준RSI+40분", "매수 15분 뒤 RSI 최고<30 & dev40>30", (0.976, 1.309, -0.708, -0.019), (-1.640, 0.310, -1.400, -0.120),
     (-2.580, -0.950, -0.710, -0.110), (0.492, 0.629), (478, 105), True, "late24 -0.71, early 하루 쏠림", 93, 13),
    ("결합 모델", "T60 로지스틱(40분 RSI 특징) p>0.269", (1.658, -0.004, -0.841, 0.0), (1.285, 0.001, -1.204, 0.0),
     (-0.130, -0.040, -0.010, 0.0), (0.609, 0.792), (202, 77), True, "late24 -0.84, 손절 전부 프리 매수", 98, 12),
]
CELLS = ("e24", "eReg", "l24", "lReg")


def fam_tables():
    rows, eff = [], []
    for nm, rule, b, u, r, fs, st, ref, why, n, pos in FAM:
        rows.append(f"| {nm} | {rule} | {b[0]:+.2f} | {b[1]:+.2f} | {b[2]:+.2f} | {b[3]:+.2f} | "
                    f"{u[0]:+.2f} / {u[2]:+.2f} | {r[0]:+.2f} / {r[2]:+.2f} | {fs[0]:.0%} / {fs[1]:.0%} | {st[0]} / {st[1]} | "
                    f"{'생존' if not ref else '반박'}: {why} |")
        eff.append(dict(계열=nm, **{f"조건-무조건_{c}": b[k] - u[k] for k, c in enumerate(CELLS)},
                        **{f"조건-반대_{c}": b[k] - r[k] for k, c in enumerate(CELLS)}))
    return rows, pd.DataFrame(eff)


# ---------------------------------------------------------------- 큰 봉 길이 견고성(g_htfscan.txt 지도 파싱)
UNG = {"종가-3%": (-2.34, -0.69, -0.61, -0.22), "종가-5%": (-0.19, -0.35, 0.20, -0.11), "본전+2%|깨어": (0.63, 0.43, -0.47, -0.09)}
BLOCKS = {"e24": "-- early 24시간 (선택용)", "eReg": "-- early 정규장만", "l24": "-- late 24시간 (확인용)", "lReg": "-- late 정규장만 (확인용)"}


def htf_map():
    txt = open(os.path.join(RES, "g_htfscan.txt"), encoding="utf-8").read().splitlines()
    maps = {}
    for key, head in BLOCKS.items():
        k = txt.index(head)
        cols = txt[k + 1].split()[1:]
        rows = {}
        for line in txt[k + 2:]:
            if not line.strip() or line.startswith("--"):
                break
            m = re.match(r"^\s*(.+?)\s+((?:[+-]\d+\.\d+\s*){7})$", line)
            rows[m.group(1).strip()] = [float(x) for x in m.group(2).split()]
        maps[key] = pd.DataFrame(rows, index=cols).T
    return maps


def htf_stats(maps):
    out = []
    for base, u in UNG.items():
        rs = [r for r in maps["e24"].index if r.startswith(base)]
        line = [f"  {base} ({len(rs)}설정 x 7길이 = {len(rs) * 7}칸)"]
        for k, c in enumerate(CELLS):
            d = maps[c].loc[rs] - u[k]
            line.append(f"{c}: 조건이 무조건보다 나은 칸 {int((d.values > 0).sum())}/{d.size}, 평균 개선 {d.values.mean():+.2f}, "
                        f"플러스 칸 {int((maps[c].loc[rs].values > 0).sum())}/{d.size}")
        allpos = sum(int(((maps['e24'].loc[r] > 0) & (maps['eReg'].loc[r] > 0) & (maps['l24'].loc[r] > 0) & (maps['lReg'].loc[r] > 0)).sum()) for r in rs)
        line.append(f"네 칸 플러스 {allpos}/{len(rs) * 7}")
        out.append(line)
    # 길이별: 종가-3%·-5% 조건 12설정의 평균 개선
    rs = [r for r in maps["e24"].index if not r.startswith("본전")]
    ug = np.array([UNG["종가-3%"] if r.startswith("종가-3%") else UNG["종가-5%"] for r in rs])
    per_len = {}
    for k, c in enumerate(CELLS):
        per_len[c] = (maps[c].loc[rs].values - ug[:, [k]]).mean(axis=0)
    pl = pd.DataFrame(per_len, index=maps["e24"].columns).T
    return out, pl


def day_conc(mkr, period):
    t = run(load(period), mkr, period, False, syms=U78)
    b = base_trades(period)
    b = b[b.sym.isin(U78)]
    days = sorted(set(b.day) | set(t.day))
    d = (t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)).sort_values(ascending=False)
    return d


if __name__ == "__main__":
    t0 = time.time()
    specs = [("생존 규칙: 종가-5% 첫신호 & conf40>40", mk()),
             ("같은 손절 조건 없음: 종가-5% 첫신호", mk(gate=None)),
             ("반대 조건: 종가-5% 첫신호 & conf40<=40", mk(gate="le")),
             ("생존 규칙 + KST 8~24시만", mk(awake=True))]
    dfs, auds = {}, {}
    for lab, m in specs:
        dfs[lab] = compare(m, lab, show=False)
        auds[lab] = {(p, reg): stop_audit(m, p, reg_only=reg) for p in ("early", "late") for reg in (False, True)}
        print(f"{lab} {time.time() - t0:.0f}s", flush=True)
    dc = {p: day_conc(specs[0][1], p) for p in ("early", "late")}
    frows, eff = fam_tables()
    maps = htf_map()
    hst, pl = htf_stats(maps)

    def cell(df, per, scope, un="사용자78"):
        r = df[(df.기간 == per) & (df.범위 == scope) & (df.종목 == un)].iloc[0]
        return r

    L = []
    w = lambda s="": L.append(s)
    w("칼손절 + 40분봉 RSI 조건 연구 종합 (g_summary, 2026-10-08)")
    w("=" * 72)
    w("가설(사용자 10/7): 40분봉 RSI 30 이하(이미 과매도)일 때 손절하면 안 좋다 -> 낮으면 보류, 아직 높은데 빠질 때만 자른다.")
    w("기준(손절 없음, 사용자78): 24시간 early +27.12 · late +17.04, 정규장만 early +1.47 · late +2.28. 숫자는 모두 기준 대비 수익률 합 차이.")
    w("선택 규칙: 매개변수는 early(사용자78 24시간 최대, 정규장만 >= 0)로만 고르고 late는 전방 확인으로만 봄.")
    w()
    w("결론: 가설의 방향은 맞다. 그러나 조건을 붙여도 칼손절은 '손해를 거의 없애는' 수준이고, 손절 없음보다 확실히 낫지는 않다.")
    w("8계열 최선안 중 반박을 견딘 것은 1개(종가 -5% 첫 신호 & 완성 40분봉 RSI>40)이고, 네 칸 모두 플러스지만 폭이 작다(late 24시간 +0.17 = 기준의 1%).")
    w()
    w("-" * 72)
    w("1. 계열별 최선안 (사용자78 차이, 헛손절 = stop_audit 24시간)")
    w("-" * 72)
    w("| 계열 | 최선안(early로 고름) | early 24h | early 정규 | late 24h | late 정규 | 조건 없음 e24/l24 | 반대 조건 e24/l24 | 헛손절 e/l | 손절 e/l | 검증 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in frows:
        w(r)
    w("* 결합 모델의 '조건 없음'은 같은 시점·분위에서 40분 RSI 특징을 뺀 모델, early 값은 학습 안쪽.")
    w(f"시험 규칙 수(연구 단계): {sum(f[9] for f in FAM)}개 (계열별 {', '.join(f'{f[0]} {f[9]}' for f in FAM)}).")
    w(f"사용자78 네 칸 모두 플러스: {sum(f[10] for f in FAM)}개 (계열별 {', '.join(f'{f[0]} {f[10]}' for f in FAM)}). "
      "지난 연구 611개 중 3개보다 훨씬 많지만, 상당수는 late 손절 1~30건에 +0.01~+0.3 수준이고 같은 손절의 문턱 변형이다.")
    w("이 종합에서 새로 돌린 규칙: 4개(생존 규칙 재확인 1, 조건 없음·반대 조건 대조 2, KST 8~24시 한정 1). 새 매개변수 선택 없음.")
    w("반박 검증 단계 이웃: 고정 10, 폭포 18, 흘러내림 13, 구조 9, 묶음 11, 큰 봉 10(+부록 11), 기준RSI 10, 모델 12. "
      "계열 최선안 8개 중 7개 반박(late 실패 6, 이웃 부호 뒤집힘 1).")
    w()
    w("-" * 72)
    w("2. 결합 시험 -> 반박을 견딘 규칙 1개뿐이라 OR 결합 안 함(g_combo.py 만들지 않음). 그 규칙만 정의에서 다시 짜서 재확인.")
    w("-" * 72)
    for lab, _ in specs:
        df = dfs[lab]
        w(f"[{lab}]")
        f = df[["기간", "범위", "종목", "건수", "합", "차이", "t", "나은날", "나쁜날", "손절", "최악", "차이_10월6일"]].copy()
        w(f.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
        for (p, reg), a in auds[lab].items():
            if a.get("손절", 0):
                w(f"  stop_audit {p} {'정규장만' if reg else '24시간'}: 손절 {a['손절']}, 헛손절비율 {a['헛손절비율']:.3f}, "
                  f"손절평균 {a['손절평균']:+.4f}, 기준평균 {a['기준평균']:+.4f}, 아낀손실합 {a['아낀손실합']:+.3f}, 크게잃을거래 막음 {a['크게잃을거래_막음']}")
            else:
                w(f"  stop_audit {p} {'정규장만' if reg else '24시간'}: 손절 0")
        w()
    s0 = dfs[specs[0][0]]
    w("생존 규칙 재현 확인(연구 보고 +0.6186/+0.4324/+0.1723/+0.0722 대비): " + ", ".join(
        f"{p} {sc} {cell(s0, p, sc).차이:+.4f}" for p in ("early", "late") for sc in ("24시간", "정규장만")))
    for p, d in dc.items():
        w(f"날짜 집중 {p} 24시간(사용자78): 합 {d.sum():+.3f}, 상위 {d.index[0]} {d.iloc[0]:+.3f}, {d.index[1]} {d.iloc[1]:+.3f}; "
          f"최대 1일 빼면 {d.sum() - d.iloc[0]:+.3f}, 상위 2일 빼면 {d.sum() - d.iloc[0] - d.iloc[1]:+.3f}")
    w()
    w("-" * 72)
    w("3. 40분봉 RSI 조건의 일반 효과")
    w("-" * 72)
    w("3a. 계열 최선안: 조건 붙은 값 - 같은 손절 조건 없음 / 조건 붙은 값 - 반대 조건 (사용자78)")
    w(eff.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    m = eff.drop(columns="계열")
    w("8계열 평균: " + ", ".join(f"{k} {v:+.2f}" for k, v in m.mean().items()))
    w("플러스 계열 수: " + ", ".join(f"{k} {int((m[k] > 1e-9).sum())}/8" for k in m.columns))
    w("-> 조건은 같은 손절의 손해를 줄인다(조건-무조건 early24 8/8, late24 6/8). 그러나 late 24시간에서 '높을 때만 자름'이 '낮을 때만 자름'보다")
    w("   나은 계열은 3/8뿐이다(평균 -0.16). 두 기간 모두 방향이 선 곳은 깊이 손절(고정·폭포)이다(3e).")
    w()
    w("3b. 손절 신호 첫 도달 순간 분해(stop_rsi40_diag.py, 137종목, '그때 팔기 - 끝까지 들고 가기' 평균, +면 손절 이득)")
    w("  종가 -3%: early dev40<=30 552건 -0.53% / >30 517건 +0.02% / >40 56건 +0.84%;  late <=30 145건 -0.54% / >30 90건 +0.31% / >35 32건 +0.91%")
    w("  종가 -2%: early <=30 -0.57% / >30 -0.12%;  late <=30 -0.47% / >30 -0.05% / >35 +0.32%")
    w("  종가 -5%: early <=30 -0.49% / >30 -0.09%;  late <=30 -0.11% / >30 +0.88%(20건)")
    w("  -> 과매도에서 자른 손절이 건당 약 -0.5% 손해(두 기간 같은 부호). 과매도가 아닐 때 자른 손절은 0 근처~소폭 이득.")
    w()
    w("3c. 고정 손절 7종 x dev>30 재판단(g_fixed): early 24시간 7/7 개선(평균 +1.47), late 24시간 5/7 개선, 2개(-5% -0.01, -7% -0.04)는 거의 같음(평균 +0.38).")
    w("    예: 종가 -3% 조건 없음 -2.34/-0.69/-0.61/-0.22 -> dev>30 -0.35/+0.10/+0.27/-0.07, 반대(conf<=40) -2.28/-1.48/-0.85/-0.29.")
    w("    종가 -2% 조건 없음 -3.96/-0.68/-1.03/-0.54 -> dev>40 재판단 -0.10/-0.01/+0.23/+0.08 (조일수록 손해인 손절도 조건으로 0 근처까지 회복).")
    w()
    w("3d. 큰 봉 길이 견고성(g_htfscan 지도: 큰 봉 20·30·40·60·80·120·160분 x RSI 7·14 x dev/conf x g 30·35)")
    for line in hst:
        w(line[0])
        for x in line[1:]:
            w("     " + x)
    w("  길이별 평균 개선(종가 -3%·-5% 12설정, 조건 - 조건 없음):")
    w(pl.to_string(float_format=lambda v: f"{v:+.2f}"))
    w("  -> 효과는 40분만의 것이 아니라 20~80분이 넓은 고원. early에서 40분이 1위인 설정이 많았으나 late로 이어지지 않음(early-late 순위 상관 +0.20).")
    w("  -> 종가 -5%의 late 플러스는 대부분 손절 자체 몫(조건 없는 -5% late24 +0.20). late에서 조건 몫이 큰 건 종가 -3%(조건 없음 -0.61)다.")
    w()
    w("3e. 조건이 안 통한 곳: 깊이와 무관한 신호(구조 붕괴·묶음·시간·모델)는 late 24시간에서 조건이 반대 조건보다 낫지 않았다")
    w("    (구조 -0.58 vs -0.28, 묶음 -0.93 vs -0.64, 흘러내림 -0.21 vs -0.19, 모델 -0.84 vs -0.01, 기준RSI -0.71 vs -0.71).")
    w("    조건이 두 기간 모두 갈라 준 곳은 '매수가 대비 -X%' 깊이 손절(고정·폭포)뿐이다.")
    w()
    w("-" * 72)
    w("4. 추천")
    w("-" * 72)
    w("미리 정한 기준(사용자78 네 칸 모두 기준보다 나음 + 반박 검증 통과)을 형식상 통과한 규칙은 1개:")
    w("  '종가 -5% 첫 신호 & 완성 40분봉 RSI14 > 40' (+0.62 / +0.43 / +0.17 / +0.07, t 1.49 / 2.03 / 1.59 / 1.14, 손절 53/20/16/5건)")
    w("판정: 약한 추천(전방 확인 대기). 지금 바로 지표에 넣을 근거는 약하다. 이유:")
    w("  - 크기: late 24시간 +0.17은 기준 +17.04의 1%. late 손절 16건 중 AKAM 9/25 한 건이 이득의 48%.")
    w("  - late 24시간에서는 조건 몫이 없다: 같은 -5% 조건 없음 +0.20 > 조건 +0.17. 조건 몫은 early(+0.81)와 late 정규장(+0.18)에서만 보인다.")
    w("  - 무작위 대조(같은 수의 -5% 신호를 무작위로 고름): early p=0.0001이지만 92개 중 고른 값, late p=0.11.")
    w("  - 전체137 late 24시간 +0.006, 이웃 10개 중 6개가 137종목 late24 음수.")
    w("  - late 손절 16건 중 7건은 프리·애프터 얇은 체결로 종가가 -5%를 찍은 뒤 매수가보다 높게 팔림(득실 합 -0.007, 영향은 작음).")
    w("다음 단계: 이 정의를 고정한 채 10/7 이후 데이터로 전방 확인만 한다(문턱·손절폭 다시 고르지 않음).")
    w("함께 지켜볼 단서(선택 절차 밖이라 추천 아님): 종가 -3% & 40분 RSI7 dev>30(+0.60/+0.20/+0.27/+0.11, early 이득 대부분 7/24 하루),")
    w("  종가 -3% & 기준RSI>30 & dev40>35(+0.71/+0.13/+0.32/+0.15, late 손절 26건).")
    w()
    w("정확한 정의(Pine 이식용):")
    w("  매수: 지금 지표 그대로(3분봉 EMA60 x0.98 지정가, 전날 +7% 다음 날 정규장 매수 안 함). 매수가 ent, 매수 봉 ei.")
    w("  손절 판단: 매수 뒤 봉마다 직전 봉 종가 c[i-1]이 ent x 0.95 이하가 '처음' 된 그 순간 한 번만 본다(저가 아님, 종가).")
    w("    그 순간의 완성 40분봉 RSI14(conf)가 40 초과면 다음 봉(i) 시가에 판다.")
    w("    40 이하이거나 값이 없으면 그 거래는 끝까지 손절하지 않는다(뒤에 다시 -5% 아래로 가도 안 봄).")
    w("  conf: 04:00 ET 기준 40분봉(프리·애프터 포함, 04:00~20:00 = 40분봉 24개), RSI 길이 14(Wilder), 봉 i가 시작할 때까지 끝난 마지막 40분봉의 값.")
    w("    Pine: conf40 = request.security(syminfo.tickerid, \"40\", ta.rsi(close, 14)[1], lookahead = barmerge.lookahead_on)을 봉 i-1에서 읽으면")
    w("    '봉 i-1이 속한 40분봉 직전 완성봉' 값이다. 봉 i가 새 40분봉에서 시작하는 경우(40분에 한 번) 연구값보다 40분봉 한 칸 오래된 값이 된다(보수적, 미래 참조 아님).")
    w("  매도 순서: RSI14 > 70 매도 -> 이 손절 -> 다음 아침 8시(KST) 정리. 손절 뒤 그 8시까지 같은 종목 재매수 금지.")
    w("  세션: 연구값은 24시간 모든 세션. 사용자가 깨어 있는 KST 8~24시에만 자르는 형태는 위 4번째 표(첫 신호가 밤이면 8시 정리까지 보유).")
    w()
    w("-" * 72)
    w("5. 대가와 주의")
    w("-" * 72)
    w("  - 조건은 손절 횟수를 크게 줄인다(종가 -5%: early 243 -> 53건, late 39 -> 16건). 손절을 드물게 해서 손해를 줄이는 몫이 크다.")
    w("  - 조건이 막은 손절: early -5% 첫 신호 254건 중 201건(conf<=40·NaN). 이 201건은 잘랐다면 합 -1.33 손해였다(late 23건은 +0.03, 차이 없음).")
    w("    최악 거래(early -18.9%, late -16.3%)는 이 규칙으로도 못 막는다.")
    w("  - 사용자 문장을 그대로 옮긴 'dev(진행 중)>30'보다 '완성 40분봉 > 40'이 나았다. 판 봉의 진행 중 RSI 중앙값은 early 32, late 28")
    w("    -> 실제 뜻은 '빠지기 직전 40분봉 RSI가 높았는데 한 40분봉 안에서 급히 -5%가 난 경우만 자른다'.")
    w("  - early는 5분봉 근사(EMA36·RSI9)라 3분봉 지표와 문턱 뜻이 조금 다르다. 40분봉 RSI는 7/14 첫날 NaN(조건 꺼짐 = 손절 안 함).")
    w("  - 미래 참조: 8계열 모두 독립 재계산으로 40분봉 RSI 불일치 0, 규칙은 c[i-1]과 htf_rsi[i]만 사용.")
    w()
    w("파일: 스크립트 code/stops/g_{fixed,waterfall,grind,structure,comove,htfscan,rsiboth,model}.py, 검증 code/stops/verify_g_*.py,")
    w("      결과 results/stops/g_*.txt, verify_g_*.txt, 이 종합 code/stops/g_summary.py -> results/stops/g_summary.txt")
    w(f"소요 {time.time() - t0:.0f}s")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
