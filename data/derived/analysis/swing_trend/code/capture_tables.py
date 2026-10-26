"""capture 결과 표 정리 → results/capture_tables.txt, results/capture_key.json (capture_confirm·capture_detect·capture_delaynull 실행 뒤)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import swing

R = swing.HERE / "results"
PERS = ("P1", "P2", "Y25")
CFG_ORDER = ["ZZP_th0.03", "ZZP_th0.05", "ZZP_th0.07", "ZZP_th0.1", "ZZP_th0.15", "ZZA_k1.5", "ZZA_k2.0", "ZZA_k3.0", "ZZA_k4.0"]


def pct(x, nd=1):
    return "" if pd.isna(x) else f"{x * 100:+.{nd}f}"


def main():
    out = []
    key = {}
    # 표 1: 확정 뒤 남은 움직임
    c = pd.read_csv(R / "capture_confirm.csv")
    ev = pd.read_csv(R / "capture_confirm_events.csv")
    ev = ev[ev.ver == "cut"]
    lg = lambda s, t: np.log1p(s) if t == 1 else -np.log1p(-s)
    rem = {}
    for (per, cfg, typ), e in ev.groupby(["per", "cfg", "typ"]):
        gl, cl = lg(e.gone, typ), lg(e.cont, typ)
        rem[(per, cfg, typ)] = float(cl.sum() / (gl + cl).sum())
    st = pd.read_csv(R / "capture_confirm_stock.csv")
    st = st[st.n >= 1]
    beat = st.assign(b=st.capt > st.capt_null).groupby(["per", "cfg", "typ"]).b.agg(["mean", "size"])
    out.append("표 1. 실시간 지그재그 확정 뒤 남은 움직임(확정일 종가 기준, %, 구간 끝에서 절단). 무작위 = 봉 섞기 200개(모든 종목 같은 날 순서)")
    out.append("  방향 매수 = 상승 확정(먹은 몫 = 반대 확정 종가/확정 종가 −1), 매도 = 하락 확정(먹은 몫 = 피한 손실 = 1 − 재매수 확정 종가/매도 확정 종가)")
    out.append("  구간 설정 방향 n 보유중앙 | 이미지남 이어짐 되돌림 남은비율 | 먹음평균 먹음중앙 승률 | 무작위평균 [5%,95%] p위 | 종목이김비율(사건 1개 이상, 보조) 절단비율")
    for per in PERS:
        for cfg in CFG_ORDER:
            for typ in (1, -1):
                r = c[(c.per == per) & (c.cfg == cfg) & (c.typ == typ)].iloc[0]
                b = beat.loc[(per, cfg, typ)] if (per, cfg, typ) in beat.index else None
                out.append(f"  {per} {cfg:11s} {'매수' if typ == 1 else '매도'} {int(r.n):5d} {r.hold_med:5.0f}일 | {pct(r.gone_mean):>6} {pct(r.cont_mean):>6} {pct(r.retr_mean):>6} {rem[(per, cfg, typ)] * 100:5.0f}% | "
                           f"{pct(r.capt_mean):>6} {pct(r.capt_med):>6} {r.win * 100:4.0f}% | {pct(r.capt_mean_null):>6} [{pct(r.capt_mean_null05)},{pct(r.capt_mean_null95)}] {r.p_hi_capt_mean:.2f} | "
                           f"{'' if b is None else f'{b['mean'] * 100:3.0f}% ({int(b['size'])})'} {r.cens * 100:3.0f}%")
            key[f"confirm_{per}_{cfg}"] = dict(buy_capt_mean=float(c[(c.per == per) & (c.cfg == cfg) & (c.typ == 1)].capt_mean.iloc[0]),
                                               buy_null=float(c[(c.per == per) & (c.cfg == cfg) & (c.typ == 1)].capt_mean_null.iloc[0]),
                                               sell_capt_mean=float(c[(c.per == per) & (c.cfg == cfg) & (c.typ == -1)].capt_mean.iloc[0]),
                                               sell_null=float(c[(c.per == per) & (c.cfg == cfg) & (c.typ == -1)].capt_mean_null.iloc[0]))
    nsig = c[(c.p_hi_capt_mean <= 0.05) | (c.p_lo_capt_mean <= 0.05)][["per", "cfg", "typ", "capt_mean", "capt_mean_null", "p_hi_capt_mean", "p_lo_capt_mean"]]
    out.append(f"  무작위와 5% 수준에서 다른 칸(평균 먹은 몫, 양쪽 각 5%): {len(nsig)} / {len(c)}")
    for _, r in nsig.iterrows():
        out.append(f"    {r.per} {r.cfg} {'매수' if r.typ == 1 else '매도'} 실제 {pct(r.capt_mean)} 무작위 {pct(r.capt_mean_null)} p위 {r.p_hi_capt_mean:.3f} p아래 {r.p_lo_capt_mean:.3f}")
    out.append("")

    # 표 2: 판정 28개 대 진실 구간
    d = pd.read_csv(R / "capture_detect.csv")
    for tn in ("T15", "TK4"):
        out.append(f"표 2-{tn}. 판정 28개 포착(진실 {'지그재그 15%' if tn == 'T15' else '4 × ATR20 지그재그'}, 예지자 = 100%, 로그 합 풀링). 기술 = 상승포착 + 하락회피 − 1(같은 노출 무작위 = 0), 귀무 = 상태 원형 이동")
        out.append("  구간 판정 | 상승포착 하락회피 기술 귀무95 p | 진입지연중앙 놓침 그대로보유 | 청산지연중앙 그대로보유 | 헛신호 월매수")
        for per in PERS:
            x = d[(d.truth == tn) & (d.per == per)].sort_values("skill", ascending=False)
            for _, r in x.iterrows():
                out.append(f"  {per} {r.det:17s} | {r.cap_up * 100:4.0f}% {r.avoid * 100:4.0f}% {r.skill * 100:+4.0f}%p {r.skill_null95 * 100:+4.0f}%p {r.p_skill:.2f} | "
                           f"{r.entry_med:4.0f}일 {r.entry_missed * 100:3.0f}% {r.entry_held * 100:3.0f}% | {r.exit_med:4.0f}일 {r.exit_held * 100:3.0f}% | "
                           f"{'' if pd.isna(r.whip) else f'{r.whip * 100:3.0f}%'} {'' if pd.isna(r.entries_per_month) else f'{r.entries_per_month:.2f}'}")
        out.append("")
    t = d[d.truth == "T15"]
    key["detect_T15_skill_range"] = {p: [float(t[t.per == p].skill.min()), float(t[t.per == p].skill.max())] for p in PERS}
    key["detect_T15_min_p"] = {p: float(t[t.per == p].p_skill.min()) for p in PERS}
    key["detect_T15_cap_up_range"] = {p: [float(t[t.per == p].cap_up.min()), float(t[t.per == p].cap_up.max())] for p in PERS}
    key["detect_T15_avoid_range"] = {p: [float(t[t.per == p].avoid.min()), float(t[t.per == p].avoid.max())] for p in PERS}

    # 표 3: 늦음의 값 — 예지자 지연 곡선(실제 대 무작위)
    dn = pd.read_csv(R / "capture_delay_null.csv")
    out.append("표 3. 늦음의 값: 예지자를 d일 늦게 사면(진입) 남는 몫, d일 늦게 팔면(청산) 상승 몫 중 돌려주는 몫. 실제 / 무작위 평균 [5%,95%] p아래")
    for tn in ("T15", "TK4"):
        for per in PERS:
            m = dn[(dn.per == per) & (dn.truth == tn)]
            g = lambda k: m[(m.metric == k) & (m.d == -1)].iloc[0]
            n_, om, ab, ae = g("n"), g("orc_mean"), g("ab_start"), g("ab_end")
            out.append(f"  {tn} {per}: 상승 구간 {int(n_.real)}개(무작위 {n_.null_mean:.0f}), 예지자 평균 로그수익 {om.real * 100:.1f}%(무작위 {om.null_mean * 100:.1f}), "
                       f"갑작 시작 {ab.real * 100:.0f}%(무작위 {ab.null_mean * 100:.0f} [{ab.null05 * 100:.0f},{ab.null95 * 100:.0f}]), 갑작 끝 {ae.real * 100:.0f}%(무작위 {ae.null_mean * 100:.0f})")
            for kind in ("entry", "exit", "entry_abrupt", "exit_abrupt"):
                q = m[m.metric == kind].set_index("d")
                cells = " ".join(f"d{dd} {q.loc[dd].real * 100:5.1f}/{q.loc[dd].null_mean * 100:5.1f}[{q.loc[dd].null05 * 100:4.0f},{q.loc[dd].null95 * 100:4.0f}]p{q.loc[dd].p_lo:.2f}" for dd in (1, 2, 3, 5, 10))
                out.append(f"    {kind:12s} {cells}")
            if tn == "T15":
                q = m[m.metric == "entry"].set_index("d"); x = m[m.metric == "exit"].set_index("d")
                key[f"delay_T15_{per}"] = dict(entry_d1_lost=float(1 - q.loc[1].real), entry_d1_lost_null=float(1 - q.loc[1].null_mean), entry_d1_p=float(q.loc[1].p_lo),
                                               entry_d5_lost=float(1 - q.loc[5].real), exit_d1_given=float(-x.loc[1].real), exit_d5_given=float(-x.loc[5].real),
                                               exit_d5_given_null=float(-x.loc[5].null_mean))
    out.append("")
    rg = pd.read_csv(R / "capture_delay_reg.csv")
    out.append("표 3b. 판정 28개 × 진실 상승 구간: 한 바퀴(상승 + 이어진 하락) 먹은 몫 / 예지자 상승 몫, 진입 지연 묶음별(풀링)")
    for _, r in rg.iterrows():
        ce = " ".join(f"{a}:{r[f'cyc_entry_{a}'] * 100:+.0f}%({int(r[f'n_entry_{a}'])})" for a in ("-99_0", "1_2", "3_5", "6_10", "11_999"))
        cx = " ".join(f"{a}:{r[f'cyc_exit_{a}'] * 100:+.0f}%({int(r[f'n_exit_{a}'])})" for a in ("-99_-1", "0_2", "3_5", "6_10", "11_999"))
        out.append(f"  {r.per} {r.truth} 진입 {ce}")
        out.append(f"  {r.per} {r.truth} 청산 {cx}")
        out.append(f"  {r.per} {r.truth} 회귀(하루당, 상승 몫 대비): 늦은 진입 {r.b_entry_late * 100:+.1f}%p 이른 진입 {r.b_entry_early * 100:+.1f}%p 늦은 청산 {r.b_exit_late * 100:+.1f}%p 이른 청산 {r.b_exit_early * 100:+.1f}%p")
    out.append("")
    sg = pd.read_csv(R / "capture_detect_segments.csv")
    sg = sg[(sg.truth == "T15") & sg.orc.notna() & sg.earn_cyc.notna()]
    cyc = sg.groupby(["per", "det"]).apply(lambda q: pd.Series(dict(up=q.earn.sum() / q.orc.sum(), cyc=q.earn_cyc.sum() / q.orc.sum())), include_groups=False)
    out.append("표 3c. 진실 15% 상승 구간 단위: 상승 구간 안 포착 대 한 바퀴(이어진 하락 끝까지) 포착, 판정별 범위")
    for per in PERS:
        q = cyc.loc[per]
        out.append(f"  {per}: 상승 안 {q.up.min() * 100:.0f}~{q.up.max() * 100:.0f}% → 한 바퀴 {q.cyc.min() * 100:+.0f}~{q.cyc.max() * 100:+.0f}% (ZZA k3 {q.loc['ZZA_k3.0'].up * 100:.0f}% → {q.loc['ZZA_k3.0'].cyc * 100:+.0f}%)")
        key[f"cycle_T15_{per}"] = dict(up_min=float(q.up.min()), up_max=float(q.up.max()), cyc_min=float(q.cyc.min()), cyc_max=float(q.cyc.max()))
    out.append("")

    # 표 4: 갑작스러운 반전 대 아닌 구간 포착
    out.append("표 4. 갑작스러운 반전(반전 첫날 새 방향 |수익| ≥ 2 ATR 또는 갭 ≥ 1 ATR) 구간 대 아닌 구간 — 상승 포착 / 하락 회피(진실 15%)")
    a = pd.read_csv(R / "capture_detect_stock.csv"); a = a[a.truth == "T15"]
    sh = a[a.det == "ZZA_k3.0"].groupby("per")[["UPa", "UP", "DNa", "DN"]].sum()
    for per in PERS:
        out.append(f"  {per}: 갑작 구간 몫 — 상승 로그수익의 {sh.loc[per].UPa / sh.loc[per].UP * 100:.0f}%, 하락의 {sh.loc[per].DNa / sh.loc[per].DN * 100:.0f}%")
    t15 = d[d.truth == "T15"]
    for per in PERS:
        x = t15[t15.per == per]
        dif_up = (x.cap_up_ab - x.cap_up_nab); dif_dn = (x.avoid_ab - x.avoid_nab)
        out.append(f"  {per}: 판정 28개 중앙 — 상승 포착 갑작 {x.cap_up_ab.median() * 100:.0f}% 대 보통 {x.cap_up_nab.median() * 100:.0f}% (차 중앙 {dif_up.median() * 100:+.0f}%p, 범위 {dif_up.min() * 100:+.0f}~{dif_up.max() * 100:+.0f}), "
                   f"하락 회피 갑작 {x.avoid_ab.median() * 100:.0f}% 대 보통 {x.avoid_nab.median() * 100:.0f}% (차 중앙 {dif_dn.median() * 100:+.0f}%p, 범위 {dif_dn.min() * 100:+.0f}~{dif_dn.max() * 100:+.0f})")
        key[f"abrupt_T15_{per}"] = dict(cap_ab=float(x.cap_up_ab.median()), cap_nab=float(x.cap_up_nab.median()), av_ab=float(x.avoid_ab.median()), av_nab=float(x.avoid_nab.median()))
    for det in ("ZZA_k3.0", "ZZP_th0.15", "ZZA_k1.5", "ZZP_th0.05", "CR_n10_b0.5_s0.5", "EMA_f10_s30", "SLP_n20"):
        cells = []
        for per in PERS:
            r = t15[(t15.per == per) & (t15.det == det)].iloc[0]
            cells.append(f"{per} 상승 {r.cap_up_ab * 100:.0f}/{r.cap_up_nab * 100:.0f} 하락 {r.avoid_ab * 100:.0f}/{r.avoid_nab * 100:.0f}")
        out.append(f"  {det:17s} " + " | ".join(cells))
    out.append("")

    # 표 5: 사례
    cs = pd.read_csv(R / "capture_cases.csv")
    out.append("표 5. MU·BE·NBIS 2026 진실 15% 구간(사후 반전점), 실시간 15% 확정일, 판정이 먹은 몫(구간 날만, 단순 %, 2026-01-02 이전 부분 제외). 판정 신호일은 결정일(체결은 다음 날 시가)")
    out.append("  종목 방향 시작→끝(거래일) 종가변화 예지자 갑작 | 15%확정일(+일) 확정때지난몫 | ZZA_k3 [신호] | ZZP15 | CR10 | SLP20")
    for _, r in cs.iterrows():
        out.append(f"  {r.sym:4s} {r.dir} {r.start}→{r.end}({r.days}) {pct(r.move_close, 0):>5} {pct(r.oracle, 0):>5} {'예' if r.abrupt else ''} | {r.conf_date}(+{r.conf_lag:.0f}) {r.gone_at_conf * 100:3.0f}% | "
                   f"{pct(r['ZZA_k3.0'], 0):>5} [{r['ZZA_k3.0_sig']}] | {pct(r['ZZP_th0.15'], 0):>5} | {pct(r['CR_n10_b0.5_s0.5'], 0):>5} | {pct(r['SLP_n20'], 0):>5}")
    (R / "capture_tables.txt").write_text("\n".join(out), encoding="utf-8")
    (R / "capture_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
