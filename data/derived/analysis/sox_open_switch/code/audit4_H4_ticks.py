"""BRIEF4 C3 H4 감사 — 체결 단위(틱) 점검. Alpaca SIP 체결·호가를 사건 25개(2026 9개 + 2023~2025 16개)의 몇 분만 받는다.

1) 신호 봉 SOXL 윗꼬리: 모든 체결(Alpaca 봉과 같아야 함) / 100주 미만 체결 뺌 / 100주 미만 + 고가·저가에 안 드는 조건 뺌.
   차트 출처(증권사 화면)마다 고가가 다를 수 있어 0.5 문턱이 그대로인지 본다.
2) +1% 지정가가 실제로 채워졌나(2026 도달 8건): 목표 봉에서 목표가 이상 체결 주수, 목표가 초과(엄격히 위) 체결 주수.
3) 진입 순간 SOXS 호가: 체결 봉 마지막 5초 NBBO 반 호가 중앙값(엔진 가정 반 호가와 비교).

사용: cd code && py audit4_H4_ticks.py → results/audit4_H4_ticks.json (키는 저장소 루트 .env)
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "../../hatta_bar/code"))
from fetch_alpaca import keys, Pace  # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
TR = "https://data.alpaca.markets/v2/stocks/{sym}/trades"
QT = "https://data.alpaca.markets/v2/stocks/{sym}/quotes"
NO_HL = set("W47CNRTUPZMQ9HVGB")       # 고가·저가 갱신에 안 드는(또는 의심) 조건 — 대략값
COSTS = json.load(open(os.path.normpath(os.path.join(HERE, "..", "cost_table.json")), encoding="utf-8"))


def hb(i):
    """lib.bucket과 같은 반 호가 칸(SOXS)."""
    k = "pre" if i < 330 else "open5" if i < 335 else "early" if i < 360 else "late"
    return COSTS["SOXS"][k]


def get(url, params, head, pace):
    rows, tok = [], None
    while True:
        p = dict(params)
        if tok:
            p["page_token"] = tok
        for a in range(8):
            pace.wait()
            try:
                r = requests.get(url, headers=head, params=p, timeout=30)
            except requests.RequestException:
                time.sleep(3 + 3 * a)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 + a)
                continue
            r.raise_for_status()
            break
        js = r.json()
        key = "trades" if "trades" in js else "quotes"
        rows += js.get(key) or []
        tok = js.get("next_page_token")
        if not tok or p.get("sort") == "desc":
            return rows


def ts(day, i, sec=0):
    t = pd.Timestamp(f"{day} 04:00", tz="America/New_York") + pd.Timedelta(minutes=int(i), seconds=sec)
    return t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def wick_of(df):
    if df.empty:
        return None
    o, c, h, l = df["p"].iloc[0], df["p"].iloc[-1], df["p"].max(), df["p"].min()
    return {"o": o, "h": h, "l": l, "c": c, "wick": (h - max(o, c)) / (h - l) if h > l else 0.0, "n": int(len(df))}


def main():
    ev = pd.read_parquet(os.path.join(RES, "ev4_events.parquet"))
    u = ev[(ev["side"] == "user") & np.isclose(ev["x"], 0.03)]
    h = u[u["c10_wick"] >= 0.5 - 1e-9].sort_values("day")
    head, pace = keys(), Pace(3.0)
    out = []
    for _, e in h.iterrows():
        day = e["day"].strftime("%Y-%m-%d")
        dec, fill, ex = int(e["dec_i"]), int(e["fill_i"]), int(e["exit_i"])
        rec = {"day": day, "year": int(e["year"]), "bar_wick": float(e["c10_wick"]), "r_w90": float(e["r_w90"]), "hit": int(e["hit"])}
        # 1) 신호 봉 SOXL 체결
        t = get(TR.format(sym="SOXL"), {"start": ts(day, dec), "end": ts(day, dec + 1), "limit": 10000, "feed": "sip", "sort": "asc"}, head, pace)
        df = pd.DataFrame(t)
        if not df.empty:
            df["c"] = df["c"].apply(lambda v: v or [])
            df = df.sort_values("t")
            allw = wick_of(df)
            rl = df[df["s"] >= 100]
            rl = rl[~rl["c"].apply(lambda cs: "I" in cs)]
            rlw = wick_of(rl)
            el = rl[~rl["c"].apply(lambda cs: any(x in NO_HL for x in cs))]
            elw = wick_of(el)
            # 고가를 만든 체결
            top = df[df["p"] == df["p"].max()]
            rec["soxl_signal_bar"] = {"all": allw, "round_lot": rlw, "round_lot_eligible": elw,
                                      "high_trades": [{"s": int(r.s), "c": r.c, "t": r.t[11:23]} for r in top.itertuples()][:5],
                                      "split_factor_adj_over_raw": None}
        # 2) 목표 봉 SOXS 체결(도달한 매매)
        if int(e["hit"]) == 1:
            t2 = get(TR.format(sym="SOXS"), {"start": ts(day, ex), "end": ts(day, ex + 1), "limit": 10000, "feed": "sip", "sort": "asc"}, head, pace)
            d2 = pd.DataFrame(t2)
            # 진입 체결 봉 SOXS 마지막 체결로 분할 비율(조정 봉 ÷ 원 체결) 추정
            t3 = get(TR.format(sym="SOXS"), {"start": ts(day, fill, 50), "end": ts(day, fill + 1), "limit": 10000, "feed": "sip", "sort": "asc"}, head, pace)
            d3 = pd.DataFrame(t3)
            entry_close_adj = float(e["entry_px"]) / (1 + hb(fill))
            raw_last = float(d3.sort_values("t")["p"].iloc[-1]) if not d3.empty else np.nan
            fac = entry_close_adj / raw_last if np.isfinite(raw_last) else np.nan
            tgt_raw = float(e["entry_px"]) * 1.01 / fac
            if not d2.empty:
                d2["c"] = d2["c"].apply(lambda v: v or [])
                ge = d2[d2["p"] >= tgt_raw - 1e-9]
                gt = d2[d2["p"] > tgt_raw + 0.005]          # 목표보다 1센트 이상 위(대기열과 무관하게 채워졌을 값)
                ge_rl = ge[(ge["s"] >= 100) & ~ge["c"].apply(lambda cs: "I" in cs)]
                rec["soxs_target_bar"] = {"target_raw": round(tgt_raw, 4), "split_factor": round(fac, 4),
                                          "max_px": float(d2["p"].max()), "shares_at_or_above": int(ge["s"].sum()),
                                          "trades_at_or_above": int(len(ge)), "round_lot_shares_at_or_above": int(ge_rl["s"].sum()),
                                          "shares_1c_above": int(gt["s"].sum()), "bar_shares": int(d2["s"].sum())}
        # 3) 진입 순간 SOXS 호가(체결 봉 마지막 5초)
        q = get(QT.format(sym="SOXS"), {"start": ts(day, fill, 55), "end": ts(day, fill + 1), "limit": 10000, "feed": "sip", "sort": "asc"}, head, pace)
        dq = pd.DataFrame(q)
        if not dq.empty:
            dq = dq[(dq["ap"] > 0) & (dq["bp"] > 0) & (dq["ap"] >= dq["bp"])]
            hs = (dq["ap"] - dq["bp"]) / ((dq["ap"] + dq["bp"]) / 2) / 2
            rec["soxs_entry_half_spread"] = {"median": float(hs.median()), "p90": float(hs.quantile(0.9)), "n": int(len(dq)),
                                             "engine_assumed": hb(fill)}
        out.append(rec)
        print(day, json.dumps({k: v for k, v in rec.items() if k != "soxl_signal_bar"}, ensure_ascii=False)[:400],
              (rec.get("soxl_signal_bar") or {}).get("round_lot", {}), flush=True)
    with open(os.path.join(RES, "audit4_H4_ticks.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v))


if __name__ == "__main__":
    main()
