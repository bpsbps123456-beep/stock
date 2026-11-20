"""추가 요인 두 묶음 → results/features_extra.csv

1) Astra 사전 등록 정의 그대로(astra/HYPOTHESES.md, 13:12 잠금) — Astra가 모은 자료로 Claude가 계산
   a_h01_wiki     위키 일평균 조회 최근 7일 / 직전 28일 (발표일 2일 전까지)
   a_h07_wiki_post 반응일 다음 날 조회 / 발표 전 28일 평균 (사후 설명용)
   a_h08_dtc      반응일 전날까지 공개된 최신 FINRA days_to_cover
   a_h09_si_ch    공개된 최신 두 회차 short_qty 비율 − 1 (분할 표시 회차 제외)
   a_h10_ins_buy  과거 90일 Form 4 매수(P) 금액 / (P+S), 접수 다음 날부터 앎
   a_h11_ins_n    과거 90일 매수(P) 신고자 수
   a_h14_gap_keep 양의 갭 사건의 (종가−전날 종가)/(시가−전날 종가)
   a_h16_soxx     반도체 분류의 발표 전 20일 SOXX − SPY
   a_h18_vcomp    발표 전 5일 일수익 표준편차 / 그 전 20일
2) 결과를 본 뒤 만든 탐색 요인(Claude, 2026-11-20 13:25) — 오른 종목이 시즌마다 한 테마로 몰린 것을 보고 추가
   t_peer_r0      같은 업종 묶음에서 이 사건 전 30거래일 안에 먼저 발표한 종목들의 반응일 QQQ 대비 평균
   t_peer_drift   그 종목들의 '반응일 다음 날 시가 → 이 사건 전날 종가' QQQ 대비 평균(먼저 발표한 동료가 실적 뒤 계속 갔나)
   t_grp_pre20    같은 업종 묶음 다른 종목(유니버스 전체)의 발표 전 20일 QQQ 대비 평균(업종 추세)
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CACHE = BASE / "cache"
AST = BASE / "astra/data"


def main() -> None:
    ev = pd.read_csv(BASE / "results/events.csv")
    ft = pd.read_csv(BASE / "results/features.csv")[["ticker", "reaction_date", "group"]]
    ev = ev.merge(ft, on=["ticker", "reaction_date"])
    px = {k: pd.read_parquet(CACHE / f"px_{k}.parquet") for k in ("open", "close", "adjclose")}
    idx = px["close"]["QQQ"].dropna().index
    px = {k: v.reindex(idx) for k, v in px.items()}
    ac = px["adjclose"]
    ao = px["open"] * ac / px["close"]
    n = len(idx)

    wiki = pd.read_parquet(AST / "wiki_views.parquet") if (AST / "wiki_views.parquet").exists() else pd.DataFrame()
    wk = {t: g.set_index("date").views.sort_index() for t, g in wiki.groupby("ticker")} if len(wiki) else {}
    si = pd.read_parquet(AST / "short_interest.parquet")
    si["pub"] = pd.to_datetime(si.publish_date)
    si = si[si.stockSplitFlag.isna()]
    sik = {t: g.sort_values("pub") for t, g in si.groupby("ticker")}
    ins = pd.read_parquet(AST / "insider.parquet")
    ins["fd"] = pd.to_datetime(ins.filing_date)
    insk = {t: g for t, g in ins[ins.code.isin(["P", "S"])].groupby("ticker")}
    grp_members = ev.groupby("group").ticker.unique().to_dict()
    uni_groups = pd.read_csv(BASE / "results/features.csv").drop_duplicates("ticker").set_index("ticker").group

    rows = []
    for e in ev.itertuples():
        k, tk = int(e.k), e.ticker
        f = {"ticker": tk, "reaction_date": e.reaction_date}
        ann = pd.Timestamp(e.ann_date)
        # H01·H07 위키
        s = wk.get(tk)
        if s is not None and len(s) > 40:
            pre = s[s.index <= ann - pd.Timedelta(days=2)]
            if len(pre) >= 35:
                b28 = pre.iloc[-35:-7].mean()
                f["a_h01_wiki"] = pre.iloc[-7:].mean() / b28 if b28 > 0 else np.nan
                nxt = s[s.index == pd.Timestamp(e.reaction_date) + pd.Timedelta(days=1)]
                f["a_h07_wiki_post"] = nxt.iloc[0] / b28 if len(nxt) and b28 > 0 else np.nan
        # H08·H09 공매도(반응일 전날까지 공개)
        g = sik.get(tk)
        if g is not None:
            pub = g[g.pub <= idx[k - 1]]
            if len(pub):
                f["a_h08_dtc"] = pub.days_to_cover.iloc[-1]
            if len(pub) >= 2 and pub.short_qty.iloc[-2] > 0:
                f["a_h09_si_ch"] = pub.short_qty.iloc[-1] / pub.short_qty.iloc[-2] - 1
        # H10·H11 내부자(접수 다음 날부터 → 반응일 전날 이전 접수분)
        g = insk.get(tk)
        lo, hi = idx[k - 1] - pd.Timedelta(days=90), idx[k - 1] - pd.Timedelta(days=1)
        if g is not None:
            w = g[(g.fd > lo) & (g.fd <= hi)]
            p, s_ = w[w.code == "P"].value.sum(), w[w.code == "S"].value.sum()
            f["a_h10_ins_buy"] = p / (p + s_) if (p + s_) > 0 else np.nan
            f["a_h11_ins_n"] = w[w.code == "P"].owner_ciks.nunique()
        else:
            f["a_h11_ins_n"] = 0
        # H14 갭 보존, H16 업종, H18 변동성 압축
        c = ac[tk]
        o_raw, c_raw = px["open"][tk], px["close"][tk]
        gp = o_raw.iloc[k] - c_raw.iloc[k - 1]
        if gp > 0 and gp / c_raw.iloc[k - 1] > 0.005:
            f["a_h14_gap_keep"] = (c_raw.iloc[k] - c_raw.iloc[k - 1]) / gp
        if e.group in ("칩 설계", "장비·소재·테스트", "파운드리·후공정", "메모리·스토리지"):
            f["a_h16_soxx"] = 100 * ((ac["SOXX"].iloc[k - 1] / ac["SOXX"].iloc[k - 21]) - (ac["SPY"].iloc[k - 1] / ac["SPY"].iloc[k - 21]))
        r = c.pct_change()
        sd20 = r.iloc[k - 25:k - 5].std()
        f["a_h18_vcomp"] = r.iloc[k - 5:k].std() / sd20 if sd20 > 0 else np.nan
        # 탐색: 먼저 발표한 동료
        peers = ev[(ev.group == e.group) & (ev.ticker != tk) & (ev.k < k) & (ev.k >= k - 30)]
        if len(peers):
            f["t_peer_r0"] = peers.r0_x.mean()
            dr = []
            for p in peers.itertuples():
                if p.k + 1 <= k - 1:
                    a, b = ao[p.ticker].iloc[p.k + 1], ac[p.ticker].iloc[k - 1]
                    qa, qb = ao["QQQ"].iloc[p.k + 1], ac["QQQ"].iloc[k - 1]
                    if a > 0 and qa > 0:
                        dr.append(100 * ((b / a - 1) - (qb / qa - 1)))
            f["t_peer_drift"] = np.mean(dr) if dr else np.nan
            f["t_peer_n"] = len(peers)
        mates = [t for t in uni_groups.index if uni_groups[t] == e.group and t != tk and t in ac.columns]
        if mates:
            rr = (ac[mates].iloc[k - 1] / ac[mates].iloc[k - 21] - 1) - (ac["QQQ"].iloc[k - 1] / ac["QQQ"].iloc[k - 21] - 1)
            f["t_grp_pre20"] = 100 * rr.mean()
        rows.append(f)
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "results/features_extra.csv", index=False)
    print(out.shape)
    print(out.notna().mean().round(2).to_string())


if __name__ == "__main__":
    main()
