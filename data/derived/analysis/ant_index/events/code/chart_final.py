"""종합 보고서 그림용 자료 chart_final.json 만들기 (설명용, 계산 새로 하지 않음).

입력은 이미 나온 조합 실행 결과(run_*.json, *_profiles, *_tests, *_lead)와
prices_close.csv 뿐이다. 분석·문턱·귀무는 다시 돌리지 않는다.

- curves: 자산·문턱·바닥/고점별 핵심 특징의 사건 전후(오프셋 -10~+10) 평균 z
  (사건 평균, 가짜 대표 평균, 위약 평균과 95% 띠) — 조합 profiles 그대로.
- price: 사건·가짜 대표 평균 가격 경로(사건일 종가 대비 %) — prices_close.csv로
  다시 계산(조합 JSON 값과 대조), 위약은 조합 JSON 값.
- bars: 바닥 대 가짜 바닥(고점 대 가짜 고점도 함께) 오프셋 0·+1 평균 z 막대와
  앞섬 닿은 비율 막대 — 조합 tests·lead 그대로.

실행: cd events/code && py -B chart_final.py
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUT = RES / "chart_final.json"

COMBOS = {
    "KOSPI_8": dict(asset="KOSPI", th=8, json="run_KOSPI_8.json", prof="run_KOSPI_8_profiles.csv",
                    tests="run_KOSPI_8_tests.csv", lead="run_KOSPI_8_lead.csv", window="W_KR"),
    "KOSPI_12": dict(asset="KOSPI", th=12, json="run_KOSPI_12.json", prof="run_KOSPI_12/profiles.csv",
                     tests="run_KOSPI_12/tests.csv", lead="run_KOSPI_12/lead.csv", window="W_KR"),
    "SKHYNIX_8": dict(asset="SKHYNIX", th=8, json="run_SKHYNIX_8.json", prof="run_SKHYNIX_8/profiles.parquet",
                      tests="run_SKHYNIX_8/tests.csv", lead="run_SKHYNIX_8/lead.csv", window="W_KR"),
    "SKHYNIX_12": dict(asset="SKHYNIX", th=12, json="run_SKHYNIX_12.json", prof="run_SKHYNIX_12_profiles.csv",
                       tests="run_SKHYNIX_12_tests.csv", lead="run_SKHYNIX_12_lead.csv", window="W_KR"),
    "SOXX_8": dict(asset="SOXX", th=8, json="run_SOXX_8.json", prof="run_SOXX_8_profiles.csv",
                   tests="run_SOXX_8_tests.csv", lead="run_SOXX_8_lead.csv", window="W_U"),
    "SOXX_12": dict(asset="SOXX", th=12, json="run_SOXX_12.json", prof="run_SOXX_12_profiles.csv",
                    tests="run_SOXX_12_tests.csv", lead="run_SOXX_12_lead.csv", window="W_U"),
}

# 자산을 가로질러 같은 특징을 보이기 위한 핵심 특징(조합 후보 + 근접 칸에서 자주 나온 것)
CORE = [
    "act_ln_posts_per_h", "act_ln_saved_per_h", "act_new30_share", "act_share_night",
    "tp_MC_A", "tp_MEM_A", "tp_TILT", "tp_LL_US_A", "tp_LS_A", "tp_ANG_A",
    "pr_P_share", "vw_ln_b_median",
]
KO = {
    "act_ln_posts_per_h": "글 수 ÷ 시간(ln)",
    "act_ln_saved_per_h": "저장 글 ÷ 시간(ln)",
    "act_ln_comments_pre_per_h": "마감 전 댓글 ÷ 시간(ln)",
    "act_comments_per_post": "글당 마감 전 댓글",
    "act_new30_share": "새 작성자 몫(직전 30일 이력 없음)",
    "act_share_night": "새벽 01~05시 글 몫",
    "tp_MC_A": "반대매매·마진콜 작성자 몫",
    "tp_MEM_A": "메모리 글 작성자 몫",
    "tp_SEMI_A": "반도체 글 작성자 몫",
    "tp_TILT": "미장/국장 글 비",
    "tp_KX_A": "국장 이탈 작성자 몫",
    "tp_LL_US_A": "미국 레버리지 롱 작성자 몫",
    "tp_LS_A": "인버스·숏 작성자 몫",
    "tp_ANG_A": "분노 단어 작성자 몫",
    "pr_P_share": "예측 몫(예측 층)",
    "pr_P_bar": "군중 예측 P̄(예측 층)",
    "pr_P_strong1": "강한 하락 몫(예측 층)",
    "pr_P_strong5": "강한 상승 몫(예측 층)",
    "vw_ln_b_median": "조회 ÷ 마감 전 댓글(ln, 수집 시점 누적)",
    "vw_up_gini": "추천 지니(수집 시점 누적)",
    "vw_top10_share": "조회 상위 10글 몫(수집 시점 누적)",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def clean(x):
    if x is None:
        return None
    if isinstance(x, float):
        return None if math.isnan(x) else round(x, 4)
    if hasattr(x, "item"):
        return clean(x.item())
    return x


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def kind_col(df: pd.DataFrame, combo: str) -> pd.Series:
    if "kind" in df.columns:
        return df["kind"].astype(str)
    return df["contrast"].astype(str).str.replace(f"{combo}_", "", regex=False)


def lists_from_json(combo: str, d: dict) -> dict:
    """{kind: (사건 날짜 목록, 가짜 대표 날짜 목록, 위약 가격 경로 {오프셋: %})}"""
    out = {}
    if combo == "KOSPI_12":
        pp = read_any(RES / "run_KOSPI_12/price_paths.csv")
        for kind in ("bottom", "top"):
            ev = [e["date"] for e in d["events"][kind]["events"]]
            fk = [e["date"] for e in d["events"][kind]["fake_reps"]]
            q = pp[(pp.kind == kind) & (pp.group == "placebo")]
            pl = {int(r.offset): float(r.mean_ret) * 100 for r in q.itertuples()}
            out[kind] = (ev, fk, pl)
        return out
    cs = d["contrasts"]
    items = cs.items() if isinstance(cs, dict) else [(c["kind"], c) for c in cs]
    for kind, c in items:
        if combo == "SKHYNIX_12":
            ev = [e["date"] for e in c["lists"]["events"]]
            fk = [e["date"] for e in c["lists"]["fake_reps"]]
            path = c["price_mean_path_pct"]
        else:
            ev = [e["date"] for e in c["events"]]
            fk = [e["date"] for e in c["fake_reps"]]
            path = c.get("price_path_mean_pct") or c.get("price_paths_mean_pct")
        plv = path["placebo"]
        if isinstance(plv, dict):
            pl = {int(k): float(v) for k, v in plv.items()}
        else:
            pl = {o: float(v) for o, v in zip(range(-10, 11), plv)}
        out[kind] = (ev, fk, pl, path["events"])
    return out


def price_path(px: pd.Series, dates: list[str]) -> dict:
    idx = {d: i for i, d in enumerate(px.index)}
    vals = px.values
    res = {}
    for o in range(-10, 11):
        xs = []
        for d in dates:
            i = idx.get(d)
            if i is None or not (0 <= i + o < len(vals)):
                continue
            xs.append((vals[i + o] / vals[i] - 1) * 100)
        res[o] = (sum(xs) / len(xs)) if xs else None
    return res


def realtime_note(window: str, feature: str, offset: int) -> str:
    if feature.startswith("vw_"):
        return "실시간 측정 불가(조회·추천 수집 시점 누적값)"
    if window == "W_U" and offset == 1:
        return "마감 초과(U(D+1)은 D+1 21:30 KST에 닫힘)"
    return "마감 안(D+1 07:59까지 쓴 글)"


def main() -> None:
    prices = pd.read_csv(RES / "prices_close.csv", encoding="utf-8-sig")
    ev_json = json.loads((RES / "events.json").read_text(encoding="utf-8"))
    kst = timezone(timedelta(hours=9))
    out = {
        "meta": {
            "made": datetime.now(kst).isoformat(timespec="seconds"),
            "title": "폭락장 바닥·고점과 개미 게시판 — 그림용 자료(설명용)",
            "note": ("탐색 연구의 그림 자료다. 결과는 '후보'이고 매매 규칙이 아니며 '지금이 바닥/고점'이라는 "
                     "판단이 아니다. 사건은 events.json(2026-10-22 종가)의 확정 꺾임점만, 끝 극값은 미확정으로 뺐다. "
                     "오프셋 +2 뒤는 사건 뒤 글이라 '알아챔' 근거가 아니다. 새로 계산한 통계는 없다(가격 경로만 "
                     "prices_close.csv로 다시 셈)."),
            "z": "직전 20창 기준 z(창 종류 보정). 오프셋 0 = 사건일 창(W_KR: D 08:00 마감, W_U: D 21:30 KST 마감), +1 = 다음 창.",
            "placebo_band": "위약 날(직전 5·20일 수익률이 비슷한 20일) z 평균 ± t × SD ÷ √(사건 수)",
            "core_features": CORE,
            "feature_ko": KO,
            "inputs": {},
        },
        "tail_unconfirmed": [],
        "combos": {},
    }
    for p in ["events.json", "prices_close.csv"]:
        out["meta"]["inputs"][p] = sha(RES / p)

    for combo, cfg in COMBOS.items():
        th = str(cfg["th"])
        tail = ev_json["assets"][cfg["asset"]]["thresholds"][th]["tail_unconfirmed"]
        out["tail_unconfirmed"].append({
            "combo": combo, "kind": tail["kind"], "date": tail["date"], "close": tail["close"],
            "status": tail["status"], "confirm_level": round(tail["confirm_level"], 2), "win_id": tail["win_id"],
        })
        for key in ("json", "prof", "tests", "lead"):
            out["meta"]["inputs"][cfg[key]] = sha(RES / cfg[key])

        d = json.loads((RES / cfg["json"]).read_text(encoding="utf-8"))
        lists = lists_from_json(combo, d)
        prof = read_any(RES / cfg["prof"])
        prof["k"] = kind_col(prof, combo)
        tests = read_any(RES / cfg["tests"])
        tests["k"] = kind_col(tests, combo)
        lead = read_any(RES / cfg["lead"])
        lead["k"] = kind_col(lead, combo)
        px = prices[prices.asset == cfg["asset"]].set_index("date")["close"].astype(float)

        cands = set(tests.loc[tests.candidate == True, "feature"]) | set(lead.loc[lead.candidate == True, "feature"])
        feats = CORE + sorted(f for f in cands if f not in CORE)
        cmb = {"asset": cfg["asset"], "threshold_pct": cfg["th"], "window": cfg["window"], "kinds": {}}
        for kind in ("bottom", "top"):
            item = lists[kind]
            ev_dates, fk_dates, pl_path = item[0], item[1], item[2]
            ev_pp = price_path(px, ev_dates)
            fk_pp = price_path(px, fk_dates)
            # 조합 JSON의 사건 가격 경로와 대조(오프셋 -1·+1)
            check = None
            if len(item) > 3:
                rep = item[3]
                rep = {int(k): v for k, v in rep.items()} if isinstance(rep, dict) else dict(zip(range(-10, 11), rep))
                check = max(abs(ev_pp[o] - rep[o]) for o in (-1, 1) if rep.get(o) is not None and ev_pp[o] is not None)
            price = {
                "offsets": list(range(-10, 11)),
                "events": [clean(ev_pp[o]) for o in range(-10, 11)],
                "fakes": [clean(fk_pp[o]) for o in range(-10, 11)],
                "placebo": [clean(pl_path.get(o)) for o in range(-10, 11)],
                "n_events": len(ev_dates), "n_fakes": len(fk_dates),
                "check_vs_run_json_max_abs_pct": clean(check),
            }
            curves = {}
            pk = prof[prof.k == kind]
            for f in feats:
                q = pk[pk.feature == f].sort_values("offset")
                if q.empty:
                    continue
                curves[f] = {
                    "offsets": [int(o) for o in q.offset],
                    "ev_mean": [clean(v) for v in q.ev_mean],
                    "ev_n": [clean(v) for v in q.ev_n],
                    "fk_mean": [clean(v) for v in q.fk_mean],
                    "pl_mean": [clean(v) for v in q.pl_mean],
                    "pl_lo": [clean(v) for v in q.pl_lo],
                    "pl_hi": [clean(v) for v in q.pl_hi],
                }
            bars_cls = []
            tk = tests[tests.k == kind]
            for f in feats:
                for off in (0, 1):
                    r = tk[(tk.feature == f) & (tk.offset == off)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    pr = pk[(pk.feature == f) & (pk.offset == off)]
                    bars_cls.append({
                        "feature": f, "offset": off,
                        "mean_z_events": clean(r.mean_z_events), "mean_z_fakes": clean(r.mean_z_fakes),
                        "mean_z_placebo": clean(pr.pl_mean.iloc[0]) if not pr.empty else None,
                        "n_events": clean(r.n_events), "n_fakes": clean(r.n_fakes),
                        "ba": clean(r.ba), "p": clean(r.p), "status_feature": r.status_feature,
                        "leak_suspect": bool(r.leak_suspect), "candidate": bool(r.candidate),
                        "realtime": realtime_note(cfg["window"], f, off),
                    })
            bars_lead = []
            lk = lead[lead.k == kind]
            for f in feats:
                for _, r in lk[lk.feature == f].iterrows():
                    bars_lead.append({
                        "feature": f, "side": r.side,
                        "rate_events": clean(r.rate_events), "rate_fakes": clean(r.rate_fakes),
                        "rate_placebo_mean": clean(r.rate_placebo_mean), "rate_placebo_q975": clean(r.rate_placebo_q975),
                        "n_events": clean(r.n_events), "n_fakes": clean(r.n_fakes),
                        "p": clean(r.p), "p_vs_placebo": clean(r.p_vs_placebo),
                        "status_feature": r.status_feature, "candidate": bool(r.candidate),
                    })
            cmb["kinds"][kind] = {
                "events": ev_dates, "fake_reps": fk_dates,
                "price_path_pct": price, "curves": curves,
                "bars_cls_offset0_1": bars_cls, "bars_lead": bars_lead,
            }
        out["combos"][combo] = cmb

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    checks = {c: {k: v["price_path_pct"]["check_vs_run_json_max_abs_pct"] for k, v in x["kinds"].items()}
              for c, x in out["combos"].items()}
    print("wrote", OUT, OUT.stat().st_size, "bytes")
    print("price check (max |diff| %, offsets -1/+1):", checks)


if __name__ == "__main__":
    main()
