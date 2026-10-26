"""이격 % 촘촘히 + '신호 뒤 1~2% 아래 지정가' (2026-10-11, 연구용, dip_by_zone.py 다음 단계).

사용자(10/11): "2.1, 2.2, 2.3, 2.4% 다 시도해도 같나? 그리고 이격 매수 신호 뒤 바로 사는 것보다 거기서 1~2% 정도 아래에
              매수를 미리 걸어 두는 게 나은 것 같기도 해서."
자료·규칙: dip_by_zone.py와 같다(136종목, 창 1 '5분 근사' 7/20~9/8, 창 2 '3분 정확' 9/9~10/9, 24시간, 전날 +7%면 정규장 쉼,
          매도 = RSI > 70 다음 봉 시가 또는 매수 뒤 첫 아침 8시(한국) 이후 첫 봉 시가).
사전 등록
  G(g): 이격 g% 하나로 매수(지금 2.0%). g = 1.5, 1.6, …, 4.0 (26개).
  L(x, 취소): 2% 신호(저가가 EMA[1]×0.98에 닿음, 신호가 P0 = min(시가, 매수선))에서 바로 사지 않고 P0 × (1 − x%)에 지정가.
     그 봉 저가가 지정가 이하면 그 봉에 체결(지정가), 뒤 봉은 시가가 지정가 이하면 시가, 저가가 닿으면 지정가.
     취소 A: 원래 매매가 팔렸을 때(RSI > 70 다음 봉 또는 신호 뒤 첫 8시)까지 안 닿으면 취소.
     취소 B: 신호 뒤 첫 아침 8시까지만 유지(RSI는 무시).
     체결 뒤 매도는 같은 규칙(8시는 체결 시각 기준). 대기 중에는 새 신호를 받지 않는다. x = 0.5, 1, 1.5, 2, 3.
  평가: 수익 합(같은 금액 매매) · 건수 · 건당 평균 · 이긴 비율 · 최악 · 매수 뒤 −5% 더 뚫린 건수 ·
        '날마다 같은 돈'(그날 산 매매의 평균을 날마다 더함 — 자금이 한정된 계좌의 거친 대리) · 정규장 매수만의 평균.
        L은 신호 수, 체결률, 체결 안 된 신호를 바로 샀다면의 수익(놓친 것)도 낸다.
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/trend_zones/code/dip_gap_limit.py
결과: results/dip_gap_limit.txt, dip_gap_limit.csv
"""
import os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zones as Zm  # noqa: E402
import dip_by_zone as DZ  # noqa: E402
from dip_by_zone import bars, prep, hot_days, X, HR, RAW, WIN, NY  # noqa: E402

GAPS = [round(1.5 + 0.1 * k, 1) for k in range(26)]
LIMS = [0.5, 1.0, 1.5, 2.0, 3.0]


def eight_after(kst_i):
    t8 = kst_i.normalize() + pd.Timedelta(hours=HR)
    return (t8 if kst_i < t8 else t8 + pd.Timedelta(days=1)).as_unit("ns").value


def sim_gap(d, g, hot):
    o, l, r, ema = d.o.values, d.l.values, d.r.values, d.ema.values
    kst = d.index.tz_convert("Asia/Seoul")
    tk = kst.as_unit("ns").asi8
    out, hold, ent, ei, dead = [], False, 0.0, 0, 0
    for i in range(61, len(d)):
        if hold:
            why = "RSI" if (i - 1 >= ei and r[i - 1] > X) else ("8시" if tk[i] >= dead else None)
            if why:
                out.append((ei, ent, o[i] / ent - 1, l[ei:i].min() / ent - 1))
                hold = False
        elif not hot[i]:
            lvl = ema[i - 1] * (1 - g)
            if l[i] <= lvl:
                ent, ei, hold, dead = min(o[i], lvl), i, True, eight_after(kst[i])
    return out


def sim_limit(d, x, hot, cancel):
    """반환: 체결 매매 [(신호 봉, 체결 봉, 체결가, 수익, MAE)], 놓친 신호 [(신호 봉, 바로 샀다면 수익)]."""
    o, l, r, ema = d.o.values, d.l.values, d.r.values, d.ema.values
    kst = d.index.tz_convert("Asia/Seoul")
    tk = kst.as_unit("ns").asi8
    fills, missed = [], []
    st, si, p0, lim, sdead, ent, ei, dead = "idle", 0, 0.0, 0.0, 0, 0.0, 0, 0
    for i in range(61, len(d)):
        if st == "hold":
            why = "RSI" if (i - 1 >= ei and r[i - 1] > X) else ("8시" if tk[i] >= dead else None)
            if why:
                fills.append((si, ei, ent, o[i] / ent - 1, l[ei:i].min() / ent - 1))
                st = "idle"
            continue
        if st == "pend":
            # 원래(바로 산) 매매가 팔리는 시점에 취소
            gone = (tk[i] >= sdead) or (cancel == "A" and i - 1 >= si and r[i - 1] > X)
            if gone:
                missed.append((si, o[i] / p0 - 1))
                st = "idle"
            else:
                if o[i] <= lim:
                    ent = o[i]
                elif l[i] <= lim:
                    ent = lim
                else:
                    continue
                ei, dead, st = i, eight_after(kst[i]), "hold"
                continue
        if st == "idle" and not hot[i]:
            lvl = ema[i - 1] * 0.98
            if l[i] <= lvl:
                p0 = min(o[i], lvl)
                lim = p0 * (1 - x / 100)
                si, sdead = i, eight_after(kst[i])
                if l[i] <= lim:                      # 같은 봉에서 더 내려와 닿음(닿는 순서상 신호가 먼저)
                    ent, ei, dead, st = lim, i, eight_after(kst[i]), "hold"
                else:
                    st = "pend"
    return fills, missed


def stats(x: pd.DataFrame):
    if len(x) == 0:
        return dict(n=0, sum=0.0, mean=np.nan, win=np.nan, worst=np.nan, dd5=0, daily=0.0, reg_mean=np.nan, reg_n=0)
    reg = x[x.ses == "정규"]
    return dict(n=len(x), sum=x.ret.sum(), mean=x.ret.mean(), win=(x.ret > 0).mean(), worst=x.ret.min(),
                dd5=int((x.mae <= -0.05).sum()), daily=x.groupby("day").ret.mean().sum(),
                reg_mean=reg.ret.mean() if len(reg) else np.nan, reg_n=len(reg))


def main():
    S = DZ.syms()
    rows, lrows, mrows = [], [], []
    for wname, (mins, lo, hi) in WIN.items():
        ema_len, rsi_len = (60, 14) if mins == 3 else (36, 9)
        for s in S:
            d = prep(bars(RAW, s, mins), ema_len, rsi_len)
            t = d.index.tz_convert(NY)
            ses = d.ses.values
            hot = hot_days(RAW, s, d, 0.07)
            ok = lambda i: lo <= t[i].date() <= hi
            for g in GAPS:
                for ei, ent, ret, mae in sim_gap(d, g / 100, hot):
                    if ok(ei):
                        rows.append((wname, f"G {g:.1f}%", s, t[ei].date(), ses[ei], ret, mae))
            for x in LIMS:
                for c in ("A", "B"):
                    fills, missed = sim_limit(d, x, hot, c)
                    name = f"L {x:.1f}% 아래 · 취소{c}"
                    for si, ei, ent, ret, mae in fills:
                        if ok(si):
                            lrows.append((wname, name, s, t[ei].date(), ses[ei], ret, mae, si))
                    for si, r0 in missed:
                        if ok(si):
                            mrows.append((wname, name, s, t[si].date(), ses[si], r0))
    G = pd.DataFrame(rows, columns=["win", "var", "sym", "day", "ses", "ret", "mae"])
    L = pd.DataFrame(lrows, columns=["win", "var", "sym", "day", "ses", "ret", "mae", "si"])
    M = pd.DataFrame(mrows, columns=["win", "var", "sym", "day", "ses", "ret0"])
    pd.concat([G.assign(kind="G"), L.drop(columns="si").assign(kind="L")]).to_csv(
        os.path.join(Zm.OUT, "dip_gap_limit.csv"), index=False, encoding="utf-8-sig")
    hdr = (f"  {'변형':<22} {'건수':>6} {'합':>9} {'건당':>7} {'이김':>5} {'최악':>7} {'-5%↓':>5} {'날마다같은돈':>11} "
           f"{'정규장 건당':>10}")
    fmt = lambda nm, z, extra="": (f"  {nm:<22} {z['n']:6d} {z['sum'] * 100:+8.1f}% {z['mean'] * 100:+6.2f}% "
                                   f"{z['win'] * 100:4.0f}% {z['worst'] * 100:+6.1f}% {z['dd5']:5d} {z['daily'] * 100:+10.1f}% "
                                   f"{z['reg_mean'] * 100:+7.2f}%({z['reg_n']})" + extra)
    lines = []
    for wname in WIN:
        lines += ["", f"== {wname} — 이격 % 촘촘히 (24시간, 같은 금액 매매)", hdr]
        for g in GAPS:
            nm = f"G {g:.1f}%"
            lines.append(fmt(nm + (" ← 지금" if g == 2.0 else ""), stats(G[(G.win == wname) & (G["var"] == nm)])))
        lines += ["", f"== {wname} — 2% 신호 뒤 x% 아래 지정가", hdr + "  | 신호 수·체결률·놓친 신호의 바로 산 수익 합"]
        base = stats(G[(G.win == wname) & (G["var"] == "G 2.0%")])
        lines.append(fmt("(바로 매수 = 지금)", base))
        for x in LIMS:
            for c in ("A", "B"):
                nm = f"L {x:.1f}% 아래 · 취소{c}"
                f_ = L[(L.win == wname) & (L["var"] == nm)]
                m_ = M[(M.win == wname) & (M["var"] == nm)]
                nsig = len(f_) + len(m_)
                lines.append(fmt(nm, stats(f_), f"  | {nsig}신호 · 체결 {len(f_) / max(nsig, 1) * 100:3.0f}% · 놓친 {m_.ret0.sum() * 100:+.1f}%"))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(Zm.OUT, "dip_gap_limit.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)


if __name__ == "__main__":
    main()
