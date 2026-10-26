"""칼손절 연구 — 묶음 동반 하락(섹터·시장) 계열(comove). 2026-10-07
10/6처럼 같은 묶음이 함께 무너질 때 매수를 자르는 규칙을 시험한다(stop_lab 시험대, 기준 = 지금 지표).

신호(모두 봉 i 시가에 팔지 정할 때 i-1 봉까지 이미 끝난 값만 쓴다):
  묶음 중앙  같은 대분류 그룹(fetch.universe) 다른 종목들의 '매수 뒤 수익률' 중앙값.
             기준점 = 매수 봉이 열리기 전 끝난 마지막 봉 종가(시작 <= tk[ei]-봉길이), 현재 = 시작 <= tk[i-1]인 마지막 봉 종가.
             사전에 없는 사용자 종목, 다른 종목이 3개 미만인 그룹(기판·부품, AI 응용)은 QQQ로 대신한다.
  섹터 ETF   칩 설계·파운드리·후공정·메모리·스토리지·장비·소재·테스트 → SMH, 소프트웨어·보안 → IGV, 바이오·유전체 → ARKG,
             나머지(사전에 없는 종목 포함) → QQQ. early는 5m, late는 1m을 3분으로(events.bars). 같은 기준점·현재 정의.
  QQQ        시장 전체.
  창 30분    매수 뒤가 아니라 '지금까지 30분' 수익률(현재 vs 시작 <= tk[i-1]-30분인 마지막 봉 종가). 이미 무너지는 중인지.
  자기 하락  c[i-1] <= 매수가 x (1 - Y) 와 AND.
  정규장만   i-1 봉이 정규장일 때만 규칙을 쓴다.
  나만 빠짐  (역방향 대조) 자기 종가 <= 매수가 x (1 - Y) 인데 묶음 중앙 매수 뒤 수익률은 -X 보다 위일 때만 자른다.
단계: 1단계 격자 46개 → 2단계 신호 종류별 early 24시간 1위를 '정규장만 적용'으로(8개) →
      3단계(1·2단계 결과를 본 뒤 추가, early 기준): 묶음중앙 30분 2.5·3%(24시간·정규장만 적용) 4개 + 나만 빠짐 4개.
선택: early 사용자78 24시간 차이가 가장 큰 규칙(단 early 정규장만 차이 >= 0). late는 고른 뒤 확인만.
사용: py stops/comove.py   → ../results/stops/comove.txt
"""
import io, os, sys, time, warnings
from contextlib import redirect_stdout
import numpy as np, pandas as pd

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
from stop_lab import load, compare, stop_audit, RAW, UNIV
from events import bars
from fetch import universe

OUT = os.path.join(CODE, "..", "results", "stops", "comove.txt")
GRP = universe()
SECTOR = {"칩 설계": "SMH", "파운드리·후공정": "SMH", "메모리·스토리지": "SMH", "장비·소재·테스트": "SMH",
          "소프트웨어·보안": "IGV", "바이오·유전체": "ARKG"}
MINPEER, WIN = 3, 30
NS_MIN = 60 * 10**9
warnings.filterwarnings("ignore", category=RuntimeWarning)

_CTX = {}


def _ctx(mins):
    if mins not in _CTX:
        S = load("early" if mins == 5 else "late")
        etf = {}
        for e in ("QQQ", "SMH", "IGV", "ARKG", "XBI"):
            b = bars(RAW, e, mins)
            etf[e] = (b.index.as_unit("ns").asi8, b.Close.values.astype(float))
        _CTX[mins] = dict(S=S, etf=etf, feat={})
    return _CTX[mins]


def asof(ptk, pc, t):
    """시작 시각 <= t 인 마지막 봉의 종가(없으면 NaN). 그 종가는 t + 봉길이 에 확정된다."""
    j = np.searchsorted(ptk, t, side="right") - 1
    return np.where(j >= 0, pc[np.clip(j, 0, None)], np.nan)


def feats(A):
    """종목 하나의 동반 하락 재료(같은 기간 안에서 한 번만 만든다)."""
    C = _ctx(A["mins"])
    s = A["sym"]
    if s in C["feat"]:
        return C["feat"][s]
    tk, ms = A["tk"], A["mins"] * NS_MIN
    g = GRP.get(s)
    peers = [p for p, v in GRP.items() if g and v == g and p != s and p in C["S"]]
    src = [(C["S"][p]["tk"], C["S"][p]["c"]) for p in peers] if len(peers) >= MINPEER else [C["etf"]["QQQ"]]
    Pc = np.column_stack([asof(a, b, tk) for a, b in src])              # i-1 시점 값으로 쓴다
    Pr = np.column_stack([asof(a, b, tk - ms) for a, b in src])         # 매수 봉 열리기 전 기준점
    Pw = np.column_stack([asof(a, b, tk - WIN * NS_MIN) for a, b in src])
    en = SECTOR.get(g, "QQQ")
    E, Q = C["etf"][en], C["etf"]["QQQ"]
    dead_idx = np.searchsorted(tk, A["dead"], side="left")             # 그 봉에 샀다면 8시 정리 봉
    F = dict(peers=len(peers) if len(peers) >= MINPEER else 0, etf=en, Pc=Pc, Pr=Pr, dead_idx=dead_idx, ms={},
             gwin=np.nanmedian(Pc / Pw - 1, axis=1),
             Ec=asof(*E, tk), Er=asof(*E, tk - ms), Ew=asof(*E, tk - WIN * NS_MIN),
             Qc=asof(*Q, tk), Qr=asof(*Q, tk - ms))
    F["ewin"] = F["Ec"] / F["Ew"] - 1
    C["feat"][s] = F
    return F


def g_since(F, ei):
    """매수 봉 ei 기준 묶음 중앙 수익률 배열: 원소 k = i-1 = ei+k 시점 값."""
    m = F["ms"].get(ei)
    if m is None:
        hi = min(len(F["Pc"]), F["dead_idx"][ei] + 2)
        m = F["ms"][ei] = np.nanmedian(F["Pc"][ei:hi] / F["Pr"][ei] - 1, axis=1)
    return m


def make(kind, X, Y=None, regonly=False):
    """kind: grp(묶음 중앙 매수 뒤) etf(섹터 ETF 매수 뒤) qqq(QQQ 매수 뒤) gwin(묶음 중앙 30분) ewin(섹터 ETF 30분)
    idio(나만 빠짐: Y 필수, 묶음 중앙 매수 뒤 > -X 일 때만)."""
    def make_rule(A):
        F, c, reg = feats(A), A["c"], A["reg"]
        own = (1 - Y) if Y is not None else None
        def rule(i, ei, ent):
            j = i - 1
            if regonly and not reg[j]:
                return False
            if own is not None and c[j] > ent * own:
                return False
            if kind == "idio":
                m = g_since(F, ei)
                k = j - ei
                return k < len(m) and not (m[k] <= -X)
            if kind == "grp":
                m = g_since(F, ei)
                k = j - ei
                return k < len(m) and m[k] <= -X
            if kind == "etf":
                return F["Ec"][j] / F["Er"][ei] - 1 <= -X
            if kind == "qqq":
                return F["Qc"][j] / F["Qr"][ei] - 1 <= -X
            if kind == "gwin":
                return F["gwin"][j] <= -X
            if kind == "ewin":
                return F["ewin"][j] <= -X
            raise ValueError(kind)
        return rule
    return make_rule


def label(kind, X, Y=None, regonly=False):
    if kind == "idio":
        return f"나만 빠짐: 자기 종가 <=-{Y:.0%} & 묶음중앙 매수뒤 >-{X:.0%}" + (" [정규장만 적용]" if regonly else "")
    nm = {"grp": "묶음중앙 매수뒤", "etf": "섹터ETF 매수뒤", "qqq": "QQQ 매수뒤", "gwin": "묶음중앙 30분", "ewin": "섹터ETF 30분"}[kind]
    s = f"{nm} <=-{X:.2%}"
    if Y is not None:
        s += f" & 자기 종가 <=-{Y:.0%}"
    if regonly:
        s += " [정규장만 적용]"
    return s


def lookahead_check():
    """미래 참조 점검: (1) 쓰는 다른 종목·ETF 봉의 종가 확정 시각 <= 결정 봉 시가 시각, (2) 컷오프 뒤 다른 종목 값을
    뒤섞어도 컷오프 전 결정이 그대로인지."""
    msgs = []
    for mins in (5, 3):
        C = _ctx(mins)
        bad = 0
        for s, A in C["S"].items():
            tk, ms = A["tk"], mins * NS_MIN
            g = GRP.get(s)
            peers = [p for p, v in GRP.items() if g and v == g and p != s and p in C["S"]]
            src = [(C["S"][p]["tk"], C["S"][p]["c"]) for p in peers] + [C["etf"][SECTOR.get(g, "QQQ")], C["etf"]["QQQ"]]
            for ptk, _ in src:
                j = np.searchsorted(ptk, tk[:-1], side="right") - 1                  # i-1 = 0..n-2 에서 쓰는 봉
                ok = (j < 0) | (ptk[np.clip(j, 0, None)] + ms <= tk[1:])             # 그 봉 종가 확정 <= 봉 i 시가
                bad += int((~ok).sum())
                jr = np.searchsorted(ptk, tk - ms, side="right") - 1                 # 매수 기준점
                okr = (jr < 0) | (ptk[np.clip(jr, 0, None)] + ms <= tk)
                bad += int((~okr).sum())
        msgs.append(f"{'early' if mins == 5 else 'late'}: 확정 시각 위반 {bad}건")
    # 뒤섞기 시험(late, 사용자 종목 몇 개): 컷오프 이후 다른 종목 종가를 무작위로 바꿔도 컷오프 전 결정 동일한지
    C = _ctx(3)
    rng = np.random.default_rng(0)
    diff = tot = 0
    for s in ["NVDA", "MU", "PLTR", "TEM", "COIN"]:
        A = C["S"][s]
        cut = A["tk"][len(A["tk"]) // 2]
        F0 = feats(A)
        keep = {p: C["S"][p]["c"].copy() for p in C["S"]}
        keepe = {e: C["etf"][e] for e in C["etf"]}
        for p in C["S"]:
            m = C["S"][p]["tk"] > cut - 3 * NS_MIN                                 # 컷오프 봉 이후(그 봉 포함)
            C["S"][p]["c"][m] *= rng.uniform(0.5, 1.5, m.sum())
        for e, (et, ec) in keepe.items():
            ec2 = ec.copy(); m = et > cut - 3 * NS_MIN; ec2[m] *= rng.uniform(0.5, 1.5, m.sum())
            C["etf"][e] = (et, ec2)
        C["feat"].pop(s, None)
        F1 = feats(A)
        n = int(np.searchsorted(A["tk"], cut, side="left"))                         # 결정 봉 i의 시가 <= cut 인 i-1 범위
        for key in ("gwin", "ewin", "Ec", "Qc"):
            a, b = F0[key][:n - 1], F1[key][:n - 1]
            tot += len(a); diff += int((~np.isclose(a, b, equal_nan=True)).sum())
        a, b = F0["Pc"][:n - 1], F1["Pc"][:n - 1]
        tot += a.size; diff += int((~np.isclose(a, b, equal_nan=True)).sum())
        for p in C["S"]:
            C["S"][p]["c"] = keep[p]
        for e in keepe:
            C["etf"][e] = keepe[e]
        C["feat"].pop(s, None)
    msgs.append(f"late 뒤섞기 시험(5종목): 컷오프 전 값 {tot}개 중 바뀐 것 {diff}개")
    return "; ".join(msgs)


def diag(P):
    """기준 거래(137종목)를 stop_paths 갈래로 나눠 매수 뒤 15·30·60분 묶음 중앙·섹터 ETF·자기 수익률 중앙값을 본다(설명용)."""
    from stop_lab import base_trades
    for p in ("early", "late"):
        mins = 5 if p == "early" else 3
        C, b = _ctx(mins), base_trades(p)
        rows = []
        for x in b.itertuples():
            A = C["S"][x.sym]; F = feats(A); ei, xi, ent = x.ei, x.xi, x.ent
            dd = A["l"][ei:xi + 1] / ent - 1; mae = dd.min(); k60 = max(1, 60 // mins)
            fast = dd[:k60 + 1].min() <= mae * 0.5 if mae < 0 else False
            cls = "반등" if x.ret > 0 else ("폭포" if mae <= -0.05 and fast else ("흘러내림" if x.ret <= -0.03 else "작은손실"))
            f = dict(갈래=cls, ret=x.ret)
            m = g_since(F, ei)
            for mm in (15, 30, 60):
                k = max(1, mm // mins)
                ok = ei + k <= xi and k < len(m)
                f[f"묶음{mm}"] = m[k] if ok else np.nan
                f[f"ETF{mm}"] = F["Ec"][ei + k] / F["Er"][ei] - 1 if ok else np.nan
                f[f"자기{mm}"] = A["c"][ei + k] / ent - 1 if ok else np.nan
            rows.append(f)
        d = pd.DataFrame(rows)
        P(f"\n[설명] {p} 기준 거래 {len(d)}건: 매수 뒤 m분 수익률 중앙값(아직 들고 있는 거래)")
        P(d.groupby("갈래")[[f"{a}{m}" for m in (15, 30, 60) for a in ("자기", "묶음", "ETF")]].median()
          .to_string(float_format=lambda v: f"{v:+.2%}"))
        s = d[d.묶음30 <= -0.01]
        P(f"  매수 뒤 30분 묶음 <= -1%: {len(s)}건, 평균 결과 {s.ret.mean():+.2%} (전체 {d.ret.mean():+.2%}), 갈래 비중 "
          + str(s.갈래.value_counts(normalize=True).round(2).to_dict()) + ", 그때 자기 수익률 중앙 " + f"{s.자기30.median():+.2%}")


def grid():
    G = []
    for X in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04):
        G.append(("grp", X, None, False))
    for X in (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03):
        G.append(("etf", X, None, False))
    for X in (0.005, 0.0075, 0.01, 0.015, 0.02):
        G.append(("qqq", X, None, False))
    for X in (0.01, 0.02, 0.03):
        for Y in (0.02, 0.03, 0.05):
            G.append(("grp", X, Y, False))
    for X in (0.005, 0.01, 0.015):
        for Y in (0.02, 0.03, 0.05):
            G.append(("etf", X, Y, False))
    for X in (0.005, 0.01):
        for Y in (0.02, 0.03, 0.05):
            G.append(("qqq", X, Y, False))
    for X in (0.01, 0.015, 0.02):
        G.append(("gwin", X, None, False))
    for X in (0.005, 0.01):
        G.append(("ewin", X, None, False))
    return G


def cell(df, period, scope, un):
    r = df[(df.기간 == period) & (df.범위 == scope) & (df.종목 == un)].iloc[0]
    return r


def main():
    t0 = time.time()
    log = io.StringIO()
    def P(*a):
        print(*a); print(*a, file=log)
    P("칼손절 — 묶음 동반 하락(comove) 시험 결과 (2026-10-07, stop_lab 시험대)")
    P("기준: 사용자78 24시간 early +27.12 · late +17.04, 정규장만 early +1.47 · late +2.28")
    P("미래 참조 점검:", lookahead_check())
    for mins in (5, 3):
        C = _ctx(mins)
        fs = {s: feats(C["S"][s]) for s in sorted(UNIV["사용자78"]) if s in C["S"]}
        if mins == 5:
            P("사용자78 신호 원천: 묶음 동료 3개 이상", sum(1 for f in fs.values() if f["peers"]), "종목, QQQ 대체",
              sum(1 for f in fs.values() if not f["peers"]), "종목; 섹터 ETF",
              pd.Series([f["etf"] for f in fs.values()]).value_counts().to_dict())
    diag(P)

    rows, tabs = [], {}
    G = grid()
    # 1단계 격자 + 2단계(정규장만 적용) — 1단계 early 결과로만 고른다
    def run_one(spec):
        kind, X, Y, ro = spec
        lb = label(kind, X, Y, ro)
        buf = io.StringIO()
        with redirect_stdout(buf):
            df = compare(make(kind, X, Y, ro), lb, show=True)
        tabs[lb] = buf.getvalue()
        u = {(p, sc): cell(df, p, sc, "사용자78") for p in ("early", "late") for sc in ("24시간", "정규장만")}
        a = {(p, sc): cell(df, p, sc, "전체137") for p in ("early", "late") for sc in ("24시간", "정규장만")}
        rows.append(dict(규칙=lb, kind=kind, X=X, Y=Y, regonly=ro,
                         e24=u["early", "24시간"].차이, eR=u["early", "정규장만"].차이,
                         l24=u["late", "24시간"].차이, lR=u["late", "정규장만"].차이,
                         e24t=u["early", "24시간"].t, eRt=u["early", "정규장만"].t,
                         l24t=u["late", "24시간"].t, lRt=u["late", "정규장만"].t,
                         e손절=u["early", "24시간"].손절, l손절=u["late", "24시간"].손절,
                         e6=u["early", "24시간"].차이_10월6일, l6=u["late", "24시간"].차이_10월6일,
                         A_e24=a["early", "24시간"].차이, A_l24=a["late", "24시간"].차이,
                         A_eR=a["early", "정규장만"].차이, A_lR=a["late", "정규장만"].차이))
        r = rows[-1]
        print(f"[{len(rows):2d}] {lb:48s} e24 {r['e24']:+.2f} eR {r['eR']:+.2f} | l24 {r['l24']:+.2f} lR {r['lR']:+.2f} "
              f"손절 {r['e손절']}/{r['l손절']}  ({time.time() - t0:.0f}s)", flush=True)

    for spec in G:
        run_one(spec)
    R = pd.DataFrame(rows)
    # 2단계: 1단계 각 신호 종류(grp/etf/qqq·자기 하락 유무)에서 early 24시간 상위 1개를 '정규장만 적용'으로 다시
    R["fam"] = R.kind + np.where(R.Y.isna(), "", "+자기")
    for fam, g in R.groupby("fam"):
        b = g.sort_values("e24", ascending=False).iloc[0]
        run_one((b.kind, b.X, None if pd.isna(b.Y) else b.Y, True))
    n12 = len(rows)
    # 3단계(1·2단계를 본 뒤 추가): 30분 창을 더 깊게, 그리고 역방향 대조(나만 빠질 때만 자름)
    for X in (0.025, 0.03):
        for ro in (False, True):
            run_one(("gwin", X, None, ro))
    for X in (0.01, 0.02):
        for Y in (0.03, 0.05):
            run_one(("idio", X, Y, False))
    R = pd.DataFrame(rows)

    n = len(R)
    allpos = R[(R.e24 > 0) & (R.eR > 0) & (R.l24 > 0) & (R.lR > 0)]
    cand = R[R.eR >= 0].sort_values("e24", ascending=False)
    best = cand.iloc[0] if len(cand) else R.sort_values("e24", ascending=False).iloc[0]

    P(f"\n시험한 규칙 {n}개 (1단계 격자 {len(G)}개 + 2단계 정규장만 적용 {n12 - len(G)}개 + 3단계 추가 {n - n12}개). "
      f"걸린 시간 {time.time() - t0:.0f}초")
    P(f"사용자78 네 칸(early/late × 24시간/정규장만) 모두 플러스: {len(allpos)}개")
    if len(allpos):
        P("  " + "; ".join(allpos.규칙))
    P(f"early 24시간 플러스: {(R.e24 > 0).sum()}개, early 두 칸 모두 플러스: {((R.e24 > 0) & (R.eR > 0)).sum()}개")

    show = R[["규칙", "e24", "e24t", "eR", "eRt", "l24", "l24t", "lR", "lRt", "e손절", "l손절", "e6", "l6", "A_e24", "A_l24"]]
    P("\n== 전체 규칙 요약(사용자78 차이, early 24시간 순) — l*는 확인용, 선택에 쓰지 않음")
    P(show.sort_values("e24", ascending=False).to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

    P(f"\n== 선택(early 사용자78 24시간 최대, early 정규장만 >= 0): {best.규칙}")
    P(tabs[best.규칙])
    spec = (best.kind, best.X, None if pd.isna(best.Y) else best.Y, bool(best.regonly))
    mk = make(*spec)
    aud = {}
    for p in ("early", "late"):
        for ro in (False, True):
            a = stop_audit(mk, p, reg_only=ro)
            aud[p, ro] = a
            P(f"stop_audit {p} {'정규장만' if ro else '24시간'} 사용자78: " +
              ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))
    P("\n== 차순위(early 사용자78 24시간 순, 선택안 제외 상위 6개)")
    ru = R[R.규칙 != best.규칙].sort_values("e24", ascending=False)
    for _, r in ru.iloc[:6].iterrows():
        P(f"  {r.규칙}: early 24h {r.e24:+.2f} 정규 {r.eR:+.2f} | late 24h {r.l24:+.2f} 정규 {r.lR:+.2f}")
    for _, r in ru.iloc[:2].iterrows():
        mk2 = make(r.kind, r.X, None if pd.isna(r.Y) else r.Y, bool(r.regonly))
        for p in ("early", "late"):
            a = stop_audit(mk2, p)
            P(f"  stop_audit {p} 24시간 [{r.규칙}]: " +
              ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))
    P("\n== 각 규칙 전체 표")
    for k, v in tabs.items():
        P(f"\n--- {k}\n{v}")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(log.getvalue())
    print("저장:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
