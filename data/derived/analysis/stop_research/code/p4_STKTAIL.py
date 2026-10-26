"""4단계 갈래 STKTAIL(2026-10-11): 손절이 '어떤 종목'에서 통하나.

문헌(results/p2_LIT.md 5·6·7번): 손절은 고변동·복권형·소형·성장주에서 효과가 있었다(Lei-Li 2009, Dai 외 2021·2023).
여기서는 일반 주식 2,441종목(trade_lab)을 무작위 날짜(10거래일마다)에 사서 60거래일 들고 있는 거래 1건마다
손절 규칙 5개를 걸고, 진입 직전(t−1)까지의 정보로 나눈 종목 묶음별로 '같은 거래를 그냥 들고 있기'와 짝 비교한다.

시점
  - 신호일 s(= 진입일 e − 1) 종가까지의 값으로 묶음을 정한다. 묶음 경계는 그날 전체 종목 단면 백분위(그날 알 수 있음).
  - 거래·손절 체결은 trade_lab.trade 그대로(진입 e일 시가, 장중 손절선은 t−1 정보, SMA는 종가 판단 → 다음 시가).
  - 명목 주가(복권 지수의 '주가 하위 1/3')는 분할 수정 종가 × 그 뒤 분할 비율 곱(yfinance 분할 이력)으로 되돌린다.
    분할 수정 가격을 그대로 쓰면 '나중에 분할한(= 많이 오른) 종목'이 과거에 싸 보이는 미래 정보가 들어간다.
  - 섹터는 종목표(2026-10) 분류라 사후 값 → 설명용, 고르기 대상에서 뺀다.

실행: PYTHONIOENCODING=utf-8 py p4_STKTAIL.py          (전체, 약 5~10분)
      PYTHONIOENCODING=utf-8 py p4_STKTAIL.py quick    (종목 1/8, 점검용)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import stock_lab as S
import trade_lab as T
from trade_lab import Leg

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
NAME = "p4_STKTAIL"

# ============================================================================ 사전 고정 격자(뒤·2025를 본 뒤 바꾸지 않음)
ENTRY, EVERY, GAP = "random", 10, 10           # 10거래일마다 한 번(위상은 종목 이름으로 고정)
H_MAIN = 60
H_SENS = {"2026앞": 20, "2026뒤": 20, "2025": 120, "2022": 120}   # 보유 기간 민감도(대표 후보·요약 표에만)
RULES = {
    "추적-20%": (Leg(1.0, "trail", 0.20),),
    "추적-25%": (Leg(1.0, "trail", 0.25),),
    "SMA50 이탈": (Leg(1.0, "sma", 50),),
    "고정-15%": (Leg(1.0, "fixed", 0.15),),
    "추적-20% 절반": (Leg(0.5, "trail", 0.20), Leg(0.5, "hold")),
}
RULE_NAMES = list(RULES)
# 묶음 정의(신호일 s 단면 백분위). 3분위: 1 = 아래 1/3, 3 = 위 1/3.
DIMS = {"유동성": "dv60", "변동성": "rv60", "모멘텀": "mom120", "고점거리": "dist"}
DIM_LABEL = {"유동성": ("저유동", "중유동", "고유동"), "변동성": ("저변동", "중변동", "고변동"),
             "모멘텀": ("약모멘텀", "중모멘텀", "강모멘텀"), "고점거리": ("고점먼", "고점중간", "고점가까운")}
LOTTERY_MAX_TOP = 0.20                          # 직전 20일 최대 일수익 상위 1/5
LOTTERY_PX_BOT = 1 / 3                           # 명목 주가 하위 1/3
COMBOS = {                                       # 사전 선언 조합 5개
    "고변동&복권": lambda d: (d["변동성"] == 3) & d["복권"],
    "고변동&저유동": lambda d: (d["변동성"] == 3) & (d["유동성"] == 1),
    "고변동&약모멘텀": lambda d: (d["변동성"] == 3) & (d["모멘텀"] == 1),
    "고변동&강모멘텀": lambda d: (d["변동성"] == 3) & (d["모멘텀"] == 3),
    "고변동&고점먼": lambda d: (d["변동성"] == 3) & (d["고점거리"] == 1),
}
MIN_N = 30
SLIP = 0.005
N_NULL = 200
CUT_FRONT = "2026-05-29"                         # 앞 구간은 청산(보유 끝)이 이 날 안에 끝나는 거래만
WIN = {"2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09"), "2025": ("2025-01-02", "2025-12-31")}
for y in range(2015, 2025):
    WIN[str(y)] = (f"{y}-01-01", f"{y}-12-31")
MAIN_W = ["2026앞", "2026뒤", "2025", "2022"]
CTX_W = [str(y) for y in range(2015, 2025)]
CRASH = {"2018 4분기 진입": ("2018", "2018-10-01", "2018-12-31"), "2020 1분기 진입": ("2020", "2020-01-02", "2020-03-31"),
         "2022 상반기 진입": ("2022", "2022-01-03", "2022-06-30")}   # 맥락용(폭락 시기는 사후로 아는 것)
SECTOR_KO = {"Technology": "기술", "Health Care": "헬스케어", "Finance": "금융", "Industrials": "산업재",
             "Consumer Discretionary": "경기소비재", "Consumer Staples": "필수소비재", "Energy": "에너지", "Utilities": "유틸리티",
             "Real Estate": "부동산", "Telecommunications": "통신", "Basic Materials": "소재", "Miscellaneous": "기타", "": "미상"}


def single_buckets():
    out = {"전체": lambda d: np.ones(len(d), bool)}
    for dim, labels in DIM_LABEL.items():
        for k, lab in enumerate(labels, 1):
            out[lab] = (lambda dim, k: (lambda d: d[dim] == k))(dim, k)
    out["복권형"] = lambda d: d["복권"]
    out["비복권"] = lambda d: (~d["복권"]) & (d["변동성"] > 0)
    return out


POOL_BUCKETS = {**single_buckets(), **COMBOS}  # 고르기 대상 20묶음(전체 1 + 3분위 12 + 복권 2 + 조합 5)
CORE_ROWS = [("아무 종목", "전체", "추적-20%"), ("아무 종목", "전체", "추적-25%"), ("아무 종목", "전체", "SMA50 이탈"),
             ("아무 종목", "전체", "고정-15%"), ("아무 종목, 절반만", "전체", "추적-20% 절반"),
             ("60일 변동성 위 1/3", "고변동", "추적-20%"), ("복권형(급등 이력 + 저가)", "복권형", "추적-20%"),
             ("고변동 + 복권형", "고변동&복권", "추적-20%"), ("거래대금 아래 1/3(소형 대리)", "저유동", "추적-20%"),
             ("120일 수익 아래 1/3", "약모멘텀", "추적-20%"), ("52주 고점에서 먼 1/3", "고점먼", "추적-20%")]
POOL_N = len(POOL_BUCKETS) * len(RULES)


# ============================================================================ 자료
def get_splits(syms):
    p = CACHE / f"{NAME}_splits.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    import yfinance as yf
    out, missing = {}, []
    for i in range(0, len(syms), 200):
        chunk = syms[i:i + 200]
        f = None
        for attempt in range(3):
            try:
                f = yf.download(chunk, start="2014-01-01", end="2026-10-10", actions=True, auto_adjust=False,
                                progress=False, threads=True, group_by="column")
                break
            except Exception as ex:
                print("재시도", i, type(ex).__name__, ex, flush=True)
                time.sleep(5)
        if f is None or "Stock Splits" not in f:
            missing += chunk
            continue
        sp = f["Stock Splits"]
        if isinstance(sp, pd.Series):
            sp = sp.to_frame(chunk[0])
        got = set()
        for sym in sp.columns:
            x = sp[sym].dropna()
            if f["Close"][sym].notna().sum() == 0:
                continue
            got.add(sym)
            x = x[x != 0]
            out[sym] = [[d.strftime("%Y-%m-%d"), float(v)] for d, v in x.items()]
        missing += [s for s in chunk if s not in got]
        print(f"분할 이력 {i + len(chunk)}/{len(syms)}", flush=True)
    out["_missing"] = missing
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def build_features(data: dict, syms: list[str], splits: dict):
    """종목별 신호일 지표와 그날 단면 백분위. 반환: (pct 패널 dict, 날짜 인덱스)."""
    raw = pd.read_parquet(CACHE / "p4_stocks.parquet", columns=["date", "sym", "Close", "Volume"])
    raw = raw[raw.sym.isin(set(syms))]
    rawg = {s: g.set_index("date") for s, g in raw.groupby("sym")}
    cols = {k: {} for k in ("dv60", "rv60", "mom120", "max20", "dist", "px")}
    for s in syms:
        b = data[s]
        g = rawg[s].reindex(b.days)
        rc, vol = g["Close"].to_numpy(float), g["Volume"].to_numpy(float)
        dv = pd.Series(rc * vol).rolling(60, min_periods=60).mean().to_numpy()
        r = b.ret()
        mom = np.full(len(b.c), np.nan)
        mom[120:] = b.c[120:] / b.c[:-120] - 1
        mx = pd.Series(r).rolling(20, min_periods=20).max().to_numpy()
        fac = np.ones(len(b.c))                       # 그 날 뒤에 일어난 분할 비율의 곱(그날 명목 주가 = 수정 종가 × fac)
        for d, ratio in splits.get(s, []):
            if ratio > 0:
                fac[b.days < d] *= ratio
        idx = pd.Index(b.days)
        cols["dv60"][s] = pd.Series(dv, idx)
        cols["rv60"][s] = pd.Series(b.rvol(60), idx)
        cols["mom120"][s] = pd.Series(mom, idx)
        cols["max20"][s] = pd.Series(mx, idx)
        cols["dist"][s] = pd.Series(b.dd252(), idx)
        cols["px"][s] = pd.Series(rc * fac, idx)
    pct = {}
    for k, d in cols.items():
        P = pd.DataFrame(d).sort_index()
        pct[k] = P.rank(axis=1, pct=True)
    return pct


def label_trades(df: pd.DataFrame, pct: dict, meta: dict) -> pd.DataFrame:
    if not len(df):
        return df
    dates = pct["dv60"].index
    ri = dates.get_indexer(df["sday"])
    out = {}
    for k, P in pct.items():
        ci = P.columns.get_indexer(df["sym"])
        arr = P.to_numpy()
        v = np.where((ri >= 0) & (ci >= 0), arr[np.clip(ri, 0, None), np.clip(ci, 0, None)], np.nan)
        out[k] = v
    for dim, key in DIMS.items():
        v = out[key]
        df[dim] = np.where(np.isnan(v), 0, np.where(v <= 1 / 3, 1, np.where(v <= 2 / 3, 2, 3))).astype(int)
    df["복권"] = (out["max20"] > 1 - LOTTERY_MAX_TOP) & (out["px"] <= LOTTERY_PX_BOT)
    df["섹터"] = [SECTOR_KO.get(meta.get(s, {}).get("sector", ""), "미상") for s in df["sym"]]
    return df


# ============================================================================ 거래 생성
def run_window(data, syms, d0, d1, H, cut=None, slip=False, paths=False, cal_pos=None):
    rows, P, POS = [], [], []
    for s in syms:
        b = data[s]
        for e in T.entries(b, ENTRY, d0, d1, GAP, EVERY):
            z = e + H - 1
            if z >= len(b.c):
                continue
            if cut and b.days[z] > cut:
                continue
            row = {"sym": s, "entry": b.days[e], "sday": b.days[e - 1], "exit_end": b.days[z]}
            entry = b.o[e]
            rel = b.c[e:z + 1] / entry
            V = [rel] if paths else None
            for j, (rn, legs) in enumerate(RULES.items()):
                r = T.trade(b, e, H, legs)
                if j == 0:
                    row["bh"], row["mae"] = r["bh"], r["mae"]
                row[f"r{j}"] = r["ret"]
                row[f"d{j}"] = r["days_in"]
                row[f"x{j}"] = any(w != "기간끝" for w, _, _ in r["legs"])          # 손절이 한 다리라도 걸렸나
                if slip:
                    rs = T.trade(b, e, H, legs, SLIP)
                    row[f"s{j}"] = rs["ret"]
                    if j == 0:
                        row["bhs"] = rs["bh"]
                if paths:
                    v = np.zeros(H)
                    for leg, (w, day, rr) in zip(legs, r["legs"]):
                        lv = rel.copy()
                        lv[day:] = 1 + rr
                        v += leg.frac * lv
                    V.append(v)
            rows.append(row)
            if paths:
                Vm = np.vstack(V)                                         # (6, H): 보유 + 규칙 5
                prev = np.concatenate([np.ones((Vm.shape[0], 1)), Vm[:, :-1]], axis=1)
                P.append((Vm / prev - 1).astype(np.float32))
                POS.append(cal_pos.get_indexer(b.days[e:z + 1]).astype(np.int32))
    df = pd.DataFrame(rows)
    if paths:
        return df, (np.stack(P) if P else np.zeros((0, 6, H), np.float32)), (np.stack(POS) if POS else np.zeros((0, H), np.int32))
    return df


# ============================================================================ 통계
def _cluster_t(d: np.ndarray, keys: np.ndarray):
    g = pd.Series(d).groupby(keys).mean()
    if len(g) < 3:
        return float("nan"), int(len(g))
    sd = g.std(ddof=1)
    return (float(g.mean() / (sd / np.sqrt(len(g)))) if sd > 0 else float("nan")), int(len(g))


def cell_stats(df: pd.DataFrame, j: int, H: int, m_day: float, slip: bool) -> dict:
    x, bh = df[f"r{j}"].to_numpy(), df["bh"].to_numpy()
    n = len(x)
    if n == 0:
        return {"n": 0}
    d = x - bh
    tm, nm = _cluster_t(d, df["entry"].str[:7].to_numpy())
    wk = pd.to_datetime(df["entry"]).dt.strftime("%G-%V").to_numpy()
    tw, nw = _cluster_t(d, wk)
    days = df[f"d{j}"].to_numpy()

    def wl(v):
        win, loss = v[v > 0], v[v <= 0]
        return (float(len(win) / len(v)), float(win.mean()) if len(win) else float("nan"),
                float(loss.mean()) if len(loss) else float("nan"),
                float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"))

    wr, aw, al, pf = wl(x)
    wrb, awb, alb, pfb = wl(bh)
    out = {"n": n, "종목수": int(df["sym"].nunique()), "보유_기대값": float(bh.mean()), "규칙_기대값": float(x.mean()),
           "짝차": float(d.mean()), "월t": tm, "월수": nm, "주t": tw, "주수": nw,
           "짝차_중앙": float(np.median(d)), "규칙이_나은비율": float((d > 1e-12).mean()),
           "발동률": float(df[f"x{j}"].mean()),
           "규칙_승률": wr, "규칙_평균이익": aw, "규칙_평균손실": al, "규칙_손익비": pf,
           "보유_승률": wrb, "보유_평균이익": awb, "보유_평균손실": alb, "보유_손익비": pfb,
           "규칙_P30": float((x <= -0.30).mean()), "보유_P30": float((bh <= -0.30).mean()),
           "규칙_P50": float((x <= -0.50).mean()), "보유_P50": float((bh <= -0.50).mean()),
           "규칙_하위5": float(np.quantile(x, 0.05)), "보유_하위5": float(np.quantile(bh, 0.05)),
           "규칙_상위5": float(np.quantile(x, 0.95)), "보유_상위5": float(np.quantile(bh, 0.95)),
           "규칙_보유일당": float(x.sum() / days.sum()), "보유_보유일당": float(bh.sum() / (n * H)),
           "평균보유일": float(days.mean()),
           "재사용차": float((x + (H - days) * m_day - bh).mean())}
    if slip and f"s{j}" in df:
        out["짝차_미끄러짐"] = float((df[f"s{j}"] - df["bhs"]).mean())
        out["규칙_기대값_미끄러짐"] = float(df[f"s{j}"].mean())
    return out


def basket(Rd: np.ndarray, pos: np.ndarray, mask: np.ndarray, ncal: int):
    """묶음 거래를 '거래마다 한 칸' 동일가중으로 매일 재조정한 바구니. 반환 {보유, 규칙0..4: 일수익 배열}."""
    if mask.sum() == 0:
        return None
    pm = pos[mask].ravel()
    C = np.bincount(pm, minlength=ncal)
    live = C > 0
    out = {}
    for k in range(Rd.shape[1]):
        Ssum = np.bincount(pm, weights=Rd[mask, k, :].ravel().astype(float), minlength=ncal)
        out[k] = Ssum[live] / C[live]
    return out


def curve_stats(daily: np.ndarray) -> dict:
    v = np.concatenate([[1.0], np.cumprod(1 + daily)])
    mdd = float((v / np.maximum.accumulate(v) - 1).min())
    n = len(daily)
    cagr = float(v[-1] ** (252 / n) - 1) if n > 0 and v[-1] > 0 else float("nan")
    return {"수익": float(v[-1] - 1), "낙폭": mdd, "연환산": cagr, "칼마": float(cagr / -mdd) if mdd < 0 else float("nan"),
            "일수": n}


# ============================================================================ 본체
def main(quick=False):
    t0 = time.time()
    data = T.load_stocks()
    meta = T.meta()["stocks"]
    allsyms = sorted(data)
    bad = sorted(s for s in allsyms if T.bad_bars(data[s]))
    syms = [s for s in allsyms if s not in set(bad)]
    if quick:
        syms = syms[::8]
    print("종목", len(allsyms), "자료 오류 의심 제외", len(bad), "→", len(syms), flush=True)
    splits = get_splits(sorted(s for s in allsyms if s not in set(bad)))
    n_split = sum(1 for s in syms if splits.get(s))
    print("분할 이력 있는 종목", n_split, "· 못 받은 종목", len(splits.get("_missing", [])), flush=True)
    pct = build_features(data, syms, splits)
    cal = pct["dv60"].index
    ncal = len(cal)
    print("지표·단면 백분위", round(time.time() - t0), "초", flush=True)

    # ---- 거래(H 60, 경로 포함)
    trades, cells, baskets = {}, [], {}
    for w, (d0, d1) in WIN.items():
        df, Rd, pos = run_window(data, syms, d0, d1, H_MAIN, cut=CUT_FRONT if w == "2026앞" else None,
                                 slip=w in MAIN_W, paths=True, cal_pos=cal)
        df = label_trades(df, pct, meta)
        trades[w] = df
        m_day = float(df["bh"].mean() / H_MAIN)
        bucket_defs = dict(POOL_BUCKETS)
        for sec in sorted(df["섹터"].unique()):
            bucket_defs["섹터:" + sec] = (lambda sec: (lambda d: d["섹터"] == sec))(sec)
        for bname, f in bucket_defs.items():
            mask = np.asarray(f(df), bool)
            sub = df[mask]
            bk = basket(Rd, pos, mask, ncal) if len(sub) else None
            hold_c = curve_stats(bk[0]) if bk else None
            for j, rn in enumerate(RULE_NAMES):
                st = cell_stats(sub, j, H_MAIN, m_day, w in MAIN_W)
                st.update({"창": w, "H": H_MAIN, "묶음": bname, "규칙": rn, "고르기대상": bname in POOL_BUCKETS})
                if bk:
                    rc = curve_stats(bk[j + 1])
                    st.update({f"바구니_{k}": v for k, v in rc.items()})
                    st.update({f"바구니보유_{k}": v for k, v in hold_c.items()})
                    exc, e_eq = S.same_mdd_excess(np.concatenate([[1.0], np.cumprod(1 + bk[j + 1])]), bk[0])
                    st["덜들기초과"], st["덜들기_e"] = exc, e_eq
                cells.append(st)
        print(w, "거래", len(df), "종목", df["sym"].nunique(), round(time.time() - t0), "초", flush=True)
        del Rd, pos

    # ---- 폭락 시기 진입(맥락, 사후 정의) — 그해 거래에서 진입일로 자른다
    for cname, (w, a, b2) in CRASH.items():
        df = trades[w]
        sub0 = df[(df["entry"] >= a) & (df["entry"] <= b2)]
        m_day = float(df["bh"].mean() / H_MAIN)
        for bname, f in POOL_BUCKETS.items():
            sub = sub0[np.asarray(f(sub0), bool)]
            for j, rn in enumerate(RULE_NAMES):
                st = cell_stats(sub, j, H_MAIN, m_day, False)
                st.update({"창": cname, "H": H_MAIN, "묶음": bname, "규칙": rn, "고르기대상": False})
                cells.append(st)

    # ---- 보유 기간 민감도
    sens_cells = []
    for w, H in H_SENS.items():
        d0, d1 = WIN[w]
        df = run_window(data, syms, d0, d1, H, cut=CUT_FRONT if w == "2026앞" else None)
        df = label_trades(df, pct, meta)
        m_day = float(df["bh"].mean() / H)
        for bname, f in POOL_BUCKETS.items():
            sub = df[np.asarray(f(df), bool)]
            for j, rn in enumerate(RULE_NAMES):
                st = cell_stats(sub, j, H, m_day, False)
                st.update({"창": w, "H": H, "묶음": bname, "규칙": rn, "고르기대상": True})
                sens_cells.append(st)
        print("민감도", w, "H", H, "거래", len(df), round(time.time() - t0), "초", flush=True)

    C = pd.DataFrame(cells)
    CS = pd.DataFrame(sens_cells)
    res = analyze(C, CS, trades)
    res["rel"] = drift_relation(C)
    res["meta"] = {"종목_전체": len(allsyms), "자료오류_제외": len(bad), "자료오류_목록": bad, "사용종목": len(syms),
                   "분할이력_종목": n_split, "분할이력_못받음": len(splits.get("_missing", [])),
                   "거래수": {w: int(len(d)) for w, d in trades.items()},
                   "격자": {"규칙": len(RULES), "고르기묶음": len(POOL_BUCKETS), "고르기칸": POOL_N,
                          "섹터묶음": int(sum(1 for b in C["묶음"].unique() if str(b).startswith("섹터:"))), "창": list(WIN),
                          "H_주": H_MAIN, "H_민감도": H_SENS, "null판": N_NULL},
                   "quick": quick, "초": round(time.time() - t0)}
    out = {"cells": C.to_dict("records"), "sens": CS.to_dict("records"), **res}
    tag = "_quick" if quick else ""
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{NAME}{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_js), encoding="utf-8")
    md = render(C, CS, res, out["meta"])
    (RESULTS / f"{NAME}{tag}.md").write_text(md, encoding="utf-8")
    print("저장", RESULTS / f"{NAME}{tag}.md", round(time.time() - t0), "초")


def rerender(tag=""):
    """저장된 json에서 보고서만 다시 쓴다(계산 반복 없음)."""
    out = json.loads((RESULTS / f"{NAME}{tag}.json").read_text(encoding="utf-8"))
    C, CS = pd.DataFrame(out["cells"]), pd.DataFrame(out["sens"])
    res = {k: out[k] for k in ("picks", "pass", "rank_corr", "spread", "null")}
    res["rel"] = drift_relation(C)
    out["rel"] = res["rel"]
    (RESULTS / f"{NAME}{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_js), encoding="utf-8")
    (RESULTS / f"{NAME}{tag}.md").write_text(render(C, CS, res, out["meta"]), encoding="utf-8")
    print("다시 씀", RESULTS / f"{NAME}{tag}.md")


def drift_relation(C: pd.DataFrame) -> dict:
    """왜 묶음마다 손절 값이 다른가(설명용): 고르기 묶음 × 창(2015~2026, H60)마다 짝차를 그 칸의 보유 기대값에 회귀.
    본전 = 짝차가 0이 되는 보유 기대값(60일). 이 값보다 그 묶음의 뒤 60일 수익이 낮을 때만 손절이 이긴다."""
    out = {}
    for rn in RULE_NAMES:
        q = C[(C["규칙"] == rn) & C["고르기대상"] & (C["n"] >= MIN_N) & C["창"].isin(list(WIN))]
        x, y = q["보유_기대값"].to_numpy(float), q["짝차"].to_numpy(float)
        b, a = np.polyfit(x, y, 1)
        out[rn] = {"칸": int(len(q)), "절편": float(a), "기울기": float(b), "본전_보유기대값": float(-a / b) if b else float("nan"),
                   "상관": float(np.corrcoef(x, y)[0, 1]), "짝차>0칸": int((y > 0).sum()),
                   "보유<0칸": int((x < 0).sum()), "보유<0칸중_짝차>0": int(((x < 0) & (y > 0)).sum())}
    return out


def _js(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return str(x)


# ============================================================================ 분석(고르기·확인·위약)
def analyze(C: pd.DataFrame, CS: pd.DataFrame, trades: dict) -> dict:
    pool = C[C["고르기대상"] & (C["n"] >= MIN_N)]
    piv = pool.pivot_table(index=["묶음", "규칙"], columns="창", values="짝차")
    pivE = pool.pivot_table(index=["묶음", "규칙"], columns="창", values="규칙_기대값")
    pivC = pool.pivot_table(index=["묶음", "규칙"], columns="창", values="바구니_칼마")
    pivCH = pool.pivot_table(index=["묶음", "규칙"], columns="창", values="바구니보유_칼마")
    front = piv["2026앞"].dropna()
    g1 = front.idxmax()
    g2 = pivE["2026앞"].dropna().idxmax()
    g3 = pivC["2026앞"].dropna().idxmax()

    def row(key):
        r = {}
        for w in WIN:
            q = C[(C["묶음"] == key[0]) & (C["규칙"] == key[1]) & (C["창"] == w)]
            if len(q):
                r[w] = q.iloc[0].to_dict()
        for w in H_SENS:
            q = CS[(CS["묶음"] == key[0]) & (CS["규칙"] == key[1]) & (CS["창"] == w)]
            if len(q):
                r[f"{w}_H{H_SENS[w]}"] = q.iloc[0].to_dict()
        return r

    picks = {"G1": {"key": list(g1), "기준": "앞 짝차 최대"}, "G2": {"key": list(g2), "기준": "앞 규칙 기대값 최대(30건 이상)"},
             "G3": {"key": list(g3), "기준": "앞 바구니 칼마 최대"}}
    for g in picks.values():
        g["rows"] = row(tuple(g["key"]))

    # 통과 집합 → 확인 성공 비율(무조건 비율과 나란히)
    pass_tab = {}
    for metric, P, cond in (("짝차>0", piv, lambda s: s > 0), ("기대값>0", pivE, lambda s: s > 0)):
        f = P["2026앞"]
        sel = cond(f)
        r = {"통과칸": int(sel.sum()), "전체칸": int(f.notna().sum())}
        for w in ("2026뒤", "2025", "2022", "2020", "2018"):
            if w in P:
                ok = cond(P[w])
                r[w] = {"통과→성공": float(ok[sel].mean()) if sel.sum() else float("nan"), "무조건": float(ok.mean())}
        pass_tab[metric] = r
    d = pivC - pivCH
    sel = d["2026앞"] > 0
    r = {"통과칸": int(sel.sum()), "전체칸": int(d["2026앞"].notna().sum())}
    for w in ("2026뒤", "2025", "2022", "2020", "2018"):
        ok = d[w] > 0
        r[w] = {"통과→성공": float(ok[sel].mean()) if sel.sum() else float("nan"), "무조건": float(ok.mean())}
    pass_tab["칼마(규칙>보유)"] = r

    # 묶음 순서가 해마다 유지되나: 같은 규칙 안에서 묶음 짝차 순위 상관(앞 vs 다른 창)
    rank_corr = {}
    for rn in RULE_NAMES:
        sub = piv.xs(rn, level="규칙")
        rank_corr[rn] = {w: float(sub["2026앞"].corr(sub[w], method="spearman")) for w in sub.columns if w != "2026앞"}
    # 3분위 위−아래 차(손절 효과가 묶음 성격에 따라 갈리는지): 해마다 부호
    spread = {}
    for dim, labs in DIM_LABEL.items():
        for rn in RULE_NAMES:
            s = {}
            for w in WIN:
                hi = C[(C["묶음"] == labs[2]) & (C["규칙"] == rn) & (C["창"] == w)]["짝차"]
                lo = C[(C["묶음"] == labs[0]) & (C["규칙"] == rn) & (C["창"] == w)]["짝차"]
                if len(hi) and len(lo):
                    s[w] = float(hi.iloc[0] - lo.iloc[0])
            spread[f"{dim}|{rn}"] = s
    for rn in RULE_NAMES:
        s = {}
        for w in WIN:
            a = C[(C["묶음"] == "복권형") & (C["규칙"] == rn) & (C["창"] == w)]["짝차"]
            b = C[(C["묶음"] == "비복권") & (C["규칙"] == rn) & (C["창"] == w)]["짝차"]
            if len(a) and len(b):
                s[w] = float(a.iloc[0] - b.iloc[0])
        spread[f"복권|{rn}"] = s

    # 위약: 같은 크기의 무작위 종목 묶음(종목 단위로 뽑아 같은 종목 거래끼리 엮임 유지) 105칸의 앞 짝차 최대
    df = trades["2026앞"]
    D = np.vstack([df[f"r{j}"].to_numpy() - df["bh"].to_numpy() for j in range(len(RULES))]).T
    usyms, inv = np.unique(df["sym"].to_numpy(), return_inverse=True)
    fr = {b: float(np.asarray(f(df), bool).mean()) for b, f in POOL_BUCKETS.items()}
    rng = np.random.default_rng(20261011)
    mx = []
    for _ in range(N_NULL):
        best = -np.inf
        for b, q in fr.items():
            u = rng.random(len(usyms)) < q
            m = u[inv]
            if m.sum() >= MIN_N:
                best = max(best, float(D[m].mean(axis=0).max()))
        mx.append(best)
    mx = np.array(mx)
    null = {"앞_G1_짝차": float(front.max()), "위약_최대_중앙": float(np.median(mx)), "위약_최대_95%": float(np.quantile(mx, 0.95)),
            "위약_최대가_G1이상_비율": float((mx >= front.max()).mean())}
    return {"picks": picks, "pass": pass_tab, "rank_corr": rank_corr, "spread": spread, "null": null}


# ============================================================================ 보고서
def pc(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def pp(x, d=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x * 100:+.{d}f}%p"


def tt(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.1f}"


def render(C, CS, res, meta) -> str:
    def get(b, r, w, col, src=C):
        q = src[(src["묶음"] == b) & (src["규칙"] == r) & (src["창"] == w)]
        return q.iloc[0][col] if len(q) and col in q else float("nan")

    g1 = res["picks"]["G1"]["key"]
    L = []
    A = L.append
    A("# 4단계 STKTAIL: 손절이 '어떤 종목'에서 통하나 (2026-10-11)")
    A("")
    A("> 사용자: \"7종목만 하는 건 아니고 일반 주식도 한다.\" 일반 주식 거래 1건(무작위 날짜에 사서 60거래일)에 손절 5개를 걸고, "
      "진입 직전 정보로 나눈 종목 묶음마다 '같은 거래를 그냥 들고 있기'와 짝 비교했다. 코드 `code/p4_STKTAIL.py`, 숫자 `results/p4_STKTAIL.json`.")
    A("")
    A("## 결론")
    A("")
    ps = res["pass"]["짝차>0"]
    rel = res.get("rel", {})
    gv = lambda b, r, w, c="짝차": get(b, r, w, c)
    A(f"1. **손절이 기대값까지 올리는 종목 묶음은 2026에 없었다.** 고르기 {ps['전체칸']}칸(묶음 {len(POOL_BUCKETS)} × 규칙 {len(RULES)}) 중 "
      f"2026 앞 구간에서 짝차(손절 − 그냥 보유)가 플러스인 칸은 {ps['통과칸']}칸이다. 확인 구간에서도 플러스 칸은 뒤 {pc(ps['2026뒤']['무조건'], 0, False)}, "
      f"2025 {pc(ps['2025']['무조건'], 0, False)}, 2022 {pc(ps['2022']['무조건'], 0, False)}였다. "
      f"문헌과 반대로 고변동·복권형에서 손해가 가장 컸다: 복권형 × 추적 −20%는 앞 {pp(gv('복권형', '추적-20%', '2026앞'))}, 2025 {pp(gv('복권형', '추적-20%', '2025'))}"
      f"(아무 종목 {pp(gv('전체', '추적-20%', '2026앞'))}, {pp(gv('전체', '추적-20%', '2025'))}). 이 순서(변동이 클수록 손해)는 해마다 거의 같았다.")
    A(f"2. **손절은 꼬리 보험이고, 보험료는 그 종목이 앞으로 오를 몫에 비례한다.** 아무 종목 추적 −20%로 거래 1건이 −30% 이하로 끝날 확률은 "
      f"{pc(gv('전체', '추적-20%', '2026앞', '보유_P30'), 1, False)} → {pc(gv('전체', '추적-20%', '2026앞', '규칙_P30'), 1, False)}"
      f"(복권형 {pc(gv('복권형', '추적-20%', '2026앞', '보유_P30'), 1, False)} → {pc(gv('복권형', '추적-20%', '2026앞', '규칙_P30'), 1, False)})로 줄었다. "
      + (f"묶음별 짝차는 그 묶음의 보유 기대값과 상관 {rel['추적-20%']['상관']:+.2f}였다. 본전은 보유 60일 기대값 {pc(rel['추적-20%']['본전_보유기대값'])} 근처다. "
         if rel else "")
      + "즉 앞으로 빠질 종목에서만 이긴다. 진입 전 정보(변동성·거래대금·모멘텀·고점 거리·복권형)로는 그런 묶음을 골라내지 못했다.")
    A(f"3. **쓴다면 덜 아픈 형태가 있다.** 추적 −25%나 절반만 손절하면 아무 종목 앞 {pp(gv('전체', '추적-25%', '2026앞'))}·{pp(gv('전체', '추적-20% 절반', '2026앞'))}, "
      f"2025 {pp(gv('전체', '추적-25%', '2025'))}·{pp(gv('전체', '추적-20% 절반', '2025'))}로 비용이 줄었다. "
      f"그래도 −30% 꼬리는 {pc(gv('전체', '추적-25%', '2026앞', '규칙_P30'), 1, False)}·{pc(gv('전체', '추적-20% 절반', '2026앞', '규칙_P30'), 1, False)}로 남았다. "
      f"대표 후보 G1(저변동 × 추적 −25%, 앞 {pp(gv('저변동', '추적-25%', '2026앞'))}, 뒤 {pp(gv('저변동', '추적-25%', '2026뒤'))}, 2025 {pp(gv('저변동', '추적-25%', '2025'))})은 "
      f"이긴 규칙이 아니다. 손절이 거의 걸리지 않아(발동 {pc(gv('저변동', '추적-25%', '2026앞', '발동률'), 0, False)}) 덜 손해 본 칸이다. "
      f"SMA50 이탈은 오래 끈 폭락 진입(2020 1분기 {pp(gv('전체', 'SMA50 이탈', '2020 1분기 진입'))}, 2022 상반기 {pp(gv('전체', 'SMA50 이탈', '2022 상반기 진입'))})에서만 이겼다. "
      f"V자였던 2018 4분기({pp(gv('전체', 'SMA50 이탈', '2018 4분기 진입'))})와 2026({pp(gv('전체', 'SMA50 이탈', '2026앞'))}·{pp(gv('전체', 'SMA50 이탈', '2026뒤'))})에서는 졌다.")
    A("")
    # ---- 격자
    A("## 격자(사전 고정, 뒤·2025를 본 뒤 바꾸지 않음)")
    A("")
    A(f"- 대상: 일반 주식 {meta['종목_전체']:,}종목 중 자료 오류 의심 {meta['자료오류_제외']}종목을 빼고 {meta['사용종목']:,}종목. "
      f"분할 이력은 {meta['분할이력_종목']}종목에서 받았다(못 받음 {meta['분할이력_못받음']}).")
    A(f"- 진입: 무작위(종목마다 10거래일에 한 번, 위상은 이름으로 고정), 신호일 종가 판단 → 다음 날 시가. 보유 {H_MAIN}거래일(민감도: 2026 {H_SENS['2026앞']}일, 2025·2022 {H_SENS['2025']}일).")
    A("- 손절 규칙 5개: 추적 −20%(산 뒤 최고 종가 대비, 장중), 추적 −25%, 종가 SMA50 이탈(다음 시가), 고정 −15%(진입가 대비, 장중), 추적 −20%를 절반에만. 나온 뒤 현금 0%, 다시 사지 않음.")
    A("- 묶음(신호일 단면 백분위): 유동성(60일 평균 거래대금, 시총 대리) 3분위, 60일 실현변동성 3분위, 120일 수익 3분위, 252일 최고 종가 대비 거리 3분위, "
      "복권형(직전 20일 최대 일수익 상위 1/5 + 명목 주가 하위 1/3) 예·아니오, 전체 → 16묶음. 사전 선언 조합 5개(고변동 × 복권·저유동·약모멘텀·강모멘텀·고점먼).")
    A(f"- 고르기 대상: {len(POOL_BUCKETS)}묶음 × 규칙 {len(RULES)} = **{POOL_N}칸**(거래 {MIN_N}건 이상). 섹터 {sum(1 for b in C['묶음'].unique() if str(b).startswith('섹터:'))}묶음 × 5는 사후 분류라 설명용. "
      f"창 13개(2026 앞·뒤, 2025, 2015~2024) + 폭락 시기 진입 3개(맥락).")
    A(f"- 창: 2026 앞(진입 1/2~, 보유 끝이 5/29 안), 2026 뒤(진입 6/1~, 10/9까지 60일을 채운 거래 = 7월 중순 진입까지), 2025(보조), 2015~2024 맥락. 위약 {N_NULL}판.")
    A("")
    A("거래 수: " + ", ".join(f"{w} {n:,}" for w, n in meta["거래수"].items()))
    A("")
    # ---- 변수 시점
    A("## 변수 시점")
    A("")
    A("| 변수 | 언제 아는가 | 쓰는 곳 |")
    A("|---|---|---|")
    A("| 60일 평균 거래대금(종가×거래량) | 신호일 s 종가 | 유동성 3분위(시총 대리. 종목표 시총은 2026-10 값이라 안 씀) |")
    A("| 60일 실현변동성, 120일 수익, 252일 고점 대비 거리, 20일 최대 일수익 | 신호일 s 종가 | 묶음 |")
    A("| 명목 주가 = 분할 수정 종가 × 그 뒤 분할 비율 곱 | 신호일 s 종가(분할은 공시된 사건이라 거꾸로 되돌릴 뿐) | 복권형 '주가 하위 1/3' |")
    A("| 단면 백분위 경계 | 신호일 s(그날 전체 종목 값) | 3분위·1/5 |")
    A("| 섹터 | 2026-10 종목표(사후) | 설명용, 고르기 제외 |")
    A("| 손절선 | 추적·고정: t−1 종가까지, 장중 체결(시가가 선 아래면 시가) / SMA50: t일 종가 판단 → t+1 시가 | 청산 |")
    A("")
    # ---- 핵심 표
    A("## 핵심 표: 상황 → 규칙 → 숫자 (짝차 = 같은 거래에서 손절 − 그냥 보유, 60거래일 거래 1건 평균)")
    A("")
    A("| 상황(진입 전날 알 수 있음) | 규칙 | 2026 앞 | 2026 뒤 | 2025 | 2022 | 2020 1분기 진입 | 2018 4분기 진입 | P(−30%) 앞 보유→규칙 | P(−30%) 2025 보유→규칙 | 보유일당 앞 보유→규칙 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    core = list(CORE_ROWS)
    if tuple(g1) not in [(b, r) for _, b, r in core]:
        core.append((f"G1: {g1[0]}", g1[0], g1[1]))
    for lab, b, rn in core:
        A(f"| {lab} | {rn} | " + " | ".join(pp(get(b, rn, w, "짝차")) for w in ("2026앞", "2026뒤", "2025", "2022", "2020 1분기 진입", "2018 4분기 진입"))
          + f" | {pc(get(b, rn, '2026앞', '보유_P30'), 1, False)}→{pc(get(b, rn, '2026앞', '규칙_P30'), 1, False)}"
          + f" | {pc(get(b, rn, '2025', '보유_P30'), 1, False)}→{pc(get(b, rn, '2025', '규칙_P30'), 1, False)}"
          + f" | {pc(get(b, rn, '2026앞', '보유_보유일당'), 2)}→{pc(get(b, rn, '2026앞', '규칙_보유일당'), 2)} |")
    A("")
    A("- 짝차는 %p(거래 1건, 60거래일). 플러스면 손절이 그냥 들고 있기보다 더 벌었다. 보유일당은 손절로 일찍 나와 생긴 현금을 다른 데 쓸 수 있을 때의 효율(돈이 묶여 있던 하루당 수익).")
    A("")
    # ---- 묶음 × 규칙 앞/뒤/2025/2022
    A("## 묶음별 짝차(손절 − 보유), 거래 1건 평균")
    A("")
    for rn in RULE_NAMES:
        A(f"### {rn}")
        A("")
        A("| 묶음 | 앞 n | 앞 짝차(월t/주t) | 뒤 짝차(주t) | 2025 짝차(월t) | 2022 짝차 | 2020 | 2018 | 앞 P(−30%) 보유→규칙 | 2025 P(−30%) 보유→규칙 |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for b in POOL_BUCKETS:
            A(f"| {b} | {int(get(b, rn, '2026앞', 'n')) if np.isfinite(get(b, rn, '2026앞', 'n')) else 0:,} | {pp(get(b, rn, '2026앞', '짝차'))} ({tt(get(b, rn, '2026앞', '월t'))}/{tt(get(b, rn, '2026앞', '주t'))}) | "
              f"{pp(get(b, rn, '2026뒤', '짝차'))} ({tt(get(b, rn, '2026뒤', '주t'))}) | {pp(get(b, rn, '2025', '짝차'))} ({tt(get(b, rn, '2025', '월t'))}) | "
              f"{pp(get(b, rn, '2022', '짝차'))} | {pp(get(b, rn, '2020', '짝차'))} | {pp(get(b, rn, '2018', '짝차'))} | "
              f"{pc(get(b, rn, '2026앞', '보유_P30'), 1, False)}→{pc(get(b, rn, '2026앞', '규칙_P30'), 1, False)} | "
              f"{pc(get(b, rn, '2025', '보유_P30'), 1, False)}→{pc(get(b, rn, '2025', '규칙_P30'), 1, False)} |")
        A("")
    # ---- 현금 재사용·덜 들기
    A("## 일찍 나온 현금을 쓰면, 그리고 '같은 낙폭만큼 덜 들기'와 비교하면")
    A("")
    A("- 재사용 가정 짝차: 손절로 나온 뒤 남은 날에 그 창 전체 종목 평균 하루 수익(보유 기준)을 번다고 치고 다시 잰 짝차.")
    A("- 덜 들기 대비: 묶음 거래를 한 칸씩 매일 같은 비중으로 들고 가는 바구니에서, 손절 바구니와 최대 낙폭이 같아지도록 보유 바구니 비중을 줄였을 때(0~100%, 0.5% 간격) 대비 초과 수익.")
    A("")
    A("| 묶음 × 규칙 | 재사용 앞 | 재사용 뒤 | 재사용 2025 | 재사용 2022 | 덜 들기 대비 앞 | 뒤 | 2025 | 2022 | 2020 | 2018 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, b, rn in list(CORE_ROWS) + [(None, g1[0], g1[1])]:
        A(f"| {b} × {rn} | " + " | ".join(pp(get(b, rn, w, "재사용차")) for w in ("2026앞", "2026뒤", "2025", "2022")) + " | "
          + " | ".join(pp(get(b, rn, w, "덜들기초과")) for w in ("2026앞", "2026뒤", "2025", "2022", "2020", "2018")) + " |")
    A("")
    # ---- 왜 묶음마다 다른가
    if "rel" in res:
        A("## 왜 묶음마다 손절 값이 다른가(설명용): 짝차 = 절편 + 기울기 × 그 묶음의 보유 기대값")
        A("")
        A("고르기 묶음 × 창(2015~2026, 60거래일) 칸마다 짝차를 보유 기대값에 회귀했다. '본전'은 짝차가 0이 되는 보유 기대값이다. "
          "그 묶음의 다음 60일 평균 수익이 본전보다 낮을 때만 손절이 이긴다. 그런데 이 값은 진입 때 모른다.")
        A("")
        A("| 규칙 | 칸 | 상관 | 기울기 | 본전(보유 60일 기대값) | 짝차 > 0 칸 | 보유 < 0 칸 중 짝차 > 0 |")
        A("|---|---|---|---|---|---|---|")
        for rn, r in res["rel"].items():
            A(f"| {rn} | {r['칸']} | {r['상관']:+.2f} | {r['기울기']:+.2f} | {pc(r['본전_보유기대값'])} | {r['짝차>0칸']} | {r['보유<0칸중_짝차>0']}/{r['보유<0칸']} |")
        A("")
    # ---- 섹터(설명용)
    A("## 섹터(사후 분류, 설명용): 추적 −20% 짝차")
    A("")
    secs = sorted(b for b in C["묶음"].unique() if str(b).startswith("섹터:"))
    A("| 섹터 | 앞 n | 앞 | 뒤 | 2025 | 2022 | 2020 | 2018 |")
    A("|---|---|---|---|---|---|---|---|")
    for b in secs:
        A(f"| {b[3:]} | {int(get(b, '추적-20%', '2026앞', 'n')) if np.isfinite(get(b, '추적-20%', '2026앞', 'n')) else 0:,} | "
          + " | ".join(pp(get(b, "추적-20%", w, "짝차")) for w in ("2026앞", "2026뒤", "2025", "2022", "2020", "2018")) + " |")
    A("")
    # ---- 고르기 표
    A("## 고르기(2026 앞으로 고르고 뒤·2025·2022로 확인)")
    A("")
    A("| 고르기 | 기준 | 고른 칸 | 앞 | 뒤 | 2025 | 2022 | 2020 | 2018 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for g, v in res["picks"].items():
        b, rn = v["key"]
        if g == "G3":
            vals = [f"칼마 {get(b, rn, w, '바구니_칼마'):.2f} (보유 {get(b, rn, w, '바구니보유_칼마'):.2f})" for w in ("2026앞", "2026뒤", "2025", "2022", "2020", "2018")]
        elif g == "G2":
            vals = [f"{pc(get(b, rn, w, '규칙_기대값'))} (보유 {pc(get(b, rn, w, '보유_기대값'))})" for w in ("2026앞", "2026뒤", "2025", "2022", "2020", "2018")]
        else:
            vals = [pp(get(b, rn, w, "짝차")) for w in ("2026앞", "2026뒤", "2025", "2022", "2020", "2018")]
        A(f"| {g} | {v['기준']} | {b} × {rn} | " + " | ".join(vals) + " |")
    A("")
    A("앞 통과 칸 → 확인 구간 성공 비율(무조건 비율):")
    A("")
    A("| 지표 | 앞 통과/전체 | 뒤 | 2025 | 2022 | 2020 | 2018 |")
    A("|---|---|---|---|---|---|---|")
    for k, r in res["pass"].items():
        A(f"| {k} | {r['통과칸']}/{r['전체칸']} | " + " | ".join(
            f"{pc(r[w]['통과→성공'], 0, False)} ({pc(r[w]['무조건'], 0, False)})" if w in r else "—" for w in ("2026뒤", "2025", "2022", "2020", "2018")) + " |")
    A("")
    sv = CS[CS["n"] >= MIN_N]
    A("- 보유 기간 민감도(같은 100칸, 짝차 > 0 칸 수): " + ", ".join(
        f"{w} {H}일 {int((sv[(sv['창'] == w) & (sv['H'] == H)]['짝차'] > 0).sum())}/{int(((sv['창'] == w) & (sv['H'] == H)).sum())}"
        for w, H in H_SENS.items()) + ".")
    nl = res["null"]
    A(f"- 위약: 같은 크기 무작위 종목 묶음 {POOL_N}칸의 앞 짝차 최대는 중앙 {pp(nl['위약_최대_중앙'])}, 95% {pp(nl['위약_최대_95%'])}. "
      f"실제 G1 {pp(nl['앞_G1_짝차'])}보다 크거나 같은 판 {pc(nl['위약_최대가_G1이상_비율'], 0, False)}.")
    A(f"- 묶음 짝차 순위 상관(앞 vs 다른 창, 같은 규칙 {len(POOL_BUCKETS)}묶음, 스피어만):")
    A("")
    rcw = ("2026뒤", "2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018", "2017", "2016", "2015")
    A("| 규칙 | " + " | ".join(w.replace("2026뒤", "뒤") for w in rcw) + " |")
    A("|---|" + "---|" * len(rcw))
    for rn, r in res["rank_corr"].items():
        A(f"| {rn} | " + " | ".join(f"{r.get(w, float('nan')):+.2f}" for w in rcw) + " |")
    A("")
    # ---- 대표 후보 상세
    b, rn = g1
    A(f"## 대표 후보 G1: {b} × {rn}")
    A("")
    A("| 창 | H | n | 보유 기대값 | 규칙 기대값 | 짝차(월t·주t) | 규칙이 나은 거래 | 발동률 | P(−30%) 보유→규칙 | 하위5% 보유→규칙 | 보유일당 보유→규칙 | 재사용 가정 차 | 미끄러짐 0.5% 짝차 | 바구니 수익/낙폭 보유→규칙 | 덜 들기 대비 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for w, s in res["picks"]["G1"]["rows"].items():
        A(f"| {w} | {int(s['H'])} | {int(s['n']):,} | {pc(s['보유_기대값'])} | {pc(s['규칙_기대값'])} | {pp(s['짝차'])} ({tt(s['월t'])}·{tt(s['주t'])}) | "
          f"{pc(s['규칙이_나은비율'], 0, False)} | {pc(s['발동률'], 0, False)} | {pc(s['보유_P30'], 1, False)}→{pc(s['규칙_P30'], 1, False)} | "
          f"{pc(s['보유_하위5'])}→{pc(s['규칙_하위5'])} | {pc(s['보유_보유일당'], 2)}→{pc(s['규칙_보유일당'], 2)} | {pp(s.get('재사용차', float('nan')))} | "
          f"{pp(s.get('짝차_미끄러짐', float('nan')))} | "
          + (f"{pc(s.get('바구니보유_수익'))}/{pc(s.get('바구니보유_낙폭'))} → {pc(s.get('바구니_수익'))}/{pc(s.get('바구니_낙폭'))} | {pp(s.get('덜들기초과'))} |"
             if isinstance(s.get('바구니_수익'), float) and np.isfinite(s.get('바구니_수익', np.nan)) else "— | — |"))
    A("")
    # ---- 승률·손익비·꼬리
    A("## 승률·손익비·꼬리 (전체 종목과 고변동·복권형, 2026 앞 / 뒤 / 2025)")
    A("")
    A("| 묶음 | 규칙 | 창 | 승률 보유→규칙 | 평균이익 | 평균손실 | 손익비 보유→규칙 | P(−30%) | P(−50%) | 하위5% | 상위5% |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for bb in ("전체", "고변동", "복권형", "고변동&복권"):
        for rr in ("추적-20%", "SMA50 이탈", "고정-15%"):
            for w in ("2026앞", "2026뒤", "2025"):
                g = lambda col: get(bb, rr, w, col)
                if not np.isfinite(g("n")):
                    continue
                A(f"| {bb} | {rr} | {w} | {pc(g('보유_승률'), 0, False)}→{pc(g('규칙_승률'), 0, False)} | {pc(g('보유_평균이익'))}→{pc(g('규칙_평균이익'))} | "
                  f"{pc(g('보유_평균손실'))}→{pc(g('규칙_평균손실'))} | {g('보유_손익비'):.2f}→{g('규칙_손익비'):.2f} | "
                  f"{pc(g('보유_P30'), 1, False)}→{pc(g('규칙_P30'), 1, False)} | {pc(g('보유_P50'), 1, False)}→{pc(g('규칙_P50'), 1, False)} | "
                  f"{pc(g('보유_하위5'))}→{pc(g('규칙_하위5'))} | {pc(g('보유_상위5'))}→{pc(g('규칙_상위5'))} |")
    A("")
    # ---- 맥락
    A("## 맥락: 해마다 짝차(진입 연도, 60거래일)")
    A("")
    A("| 묶음 × 규칙 | " + " | ".join(CTX_W + ["2025", "2026앞", "2026뒤"]) + " | 플러스 해(2015~2025) |")
    A("|---|" + "---|" * (len(CTX_W) + 4))
    ctx_rows = [("전체", "추적-20%"), ("전체", "SMA50 이탈"), ("전체", "고정-15%"), ("고변동", "추적-20%"), ("저변동", "추적-20%"),
                ("복권형", "추적-20%"), ("저유동", "추적-20%"), ("약모멘텀", "추적-20%"), ("고점먼", "추적-20%"), ("고변동&복권", "추적-20%"),
                tuple(g1)]
    seen = set()
    for bb, rr in ctx_rows:
        if (bb, rr) in seen:
            continue
        seen.add((bb, rr))
        vals = [get(bb, rr, w, "짝차") for w in CTX_W + ["2025", "2026앞", "2026뒤"]]
        pos = sum(1 for v in vals[:len(CTX_W) + 1] if np.isfinite(v) and v > 0)
        A(f"| {bb} × {rr} | " + " | ".join(pp(v) for v in vals) + f" | {pos}/{len(CTX_W) + 1} |")
    A("")
    A("위 1/3 − 아래 1/3 짝차 차이(손절 효과가 그 성격 쪽에서 더 큰가), 추적 −20%:")
    A("")
    A("| 성격 | " + " | ".join(CTX_W + ["2025", "2026앞", "2026뒤"]) + " | 플러스 해 |")
    A("|---|" + "---|" * (len(CTX_W) + 4))
    for key in ("변동성|추적-20%", "유동성|추적-20%", "모멘텀|추적-20%", "고점거리|추적-20%", "복권|추적-20%"):
        s = res["spread"][key]
        vals = [s.get(w, float("nan")) for w in CTX_W + ["2025", "2026앞", "2026뒤"]]
        pos = sum(1 for v in vals[:len(CTX_W) + 1] if np.isfinite(v) and v > 0)
        name = {"변동성": "고변동 − 저변동", "유동성": "고유동 − 저유동", "모멘텀": "강모멘텀 − 약모멘텀",
                "고점거리": "고점가까운 − 고점먼", "복권": "복권형 − 비복권"}[key.split("|")[0]]
        A(f"| {name} | " + " | ".join(pp(v) for v in vals) + f" | {pos}/{len(CTX_W) + 1} |")
    A("")
    A("폭락 시기 진입(사후 정의, 맥락), 짝차와 P(−30%) 보유→규칙:")
    A("")
    A("| 묶음 × 규칙 | " + " | ".join(CRASH) + " |")
    A("|---|" + "---|" * len(CRASH))
    for bb, rr in [("전체", "추적-20%"), ("전체", "추적-25%"), ("전체", "SMA50 이탈"), ("전체", "고정-15%"), ("고변동", "추적-20%"),
                   ("복권형", "추적-20%"), ("고변동&복권", "추적-20%"), tuple(g1)]:
        A(f"| {bb} × {rr} | " + " | ".join(
            f"{pp(get(bb, rr, w, '짝차'))} ({pc(get(bb, rr, w, '보유_P30'), 0, False)}→{pc(get(bb, rr, w, '규칙_P30'), 0, False)})" for w in CRASH) + " |")
    A("")
    A("## 선택편향·한계")
    A("")
    nl = res["null"]
    rc = res["rank_corr"]
    A(f"- **생존편향(가장 큰 한계)**: 종목표는 2026-10에 살아 있고 시총 10억 달러 이상인 종목이다. 문헌이 손절 효과를 찾은 곳(복권형·소형·고변동, Dai 외 2023은 CRSP 전 종목)은 "
      "망하거나 작아져 빠진 종목이 가장 많은 곳이다. 그래서 이 표본의 고변동·복권형 보유 수익은 부풀려졌고, 손절 값은 깎였다. 2026 안에서도 올해 10억 달러 아래로 떨어진 종목은 빠져 있다. "
      "'복권형에서 손절이 더 손해'는 생존 종목에 한정된 결론이다.")
    A(f"- **고르기 정보**: 앞 통과(짝차 > 0) 칸이 0개라 '통과 → 확인' 비율을 낼 수 없다. 위약(같은 크기의 무작위 종목 묶음 {POOL_N}칸 × {N_NULL}판)에서 앞 짝차 최대는 "
      f"중앙 {pp(nl['위약_최대_중앙'])}, 95% {pp(nl['위약_최대_95%'])}였다. G1 {pp(nl['앞_G1_짝차'])}은 우연보다 낫지만, 이는 저변동주에서 손절이 덜 걸리는 구조 때문이고 이긴 것은 아니다.")
    yrs = [str(y) for y in range(2016, 2026)]
    A(f"- **묶음 순서의 안정성**: 추적 −20%의 묶음 짝차 순위는 2026 앞과 2016~2025 각 해의 스피어만 상관이 "
      f"{min(rc['추적-20%'][y] for y in yrs):+.2f}~{max(rc['추적-20%'][y] for y in yrs):+.2f}였다(뒤 {rc['추적-20%']['2026뒤']:+.2f}). "
      f"예외는 2015({rc['추적-20%']['2015']:+.2f})다. 그해(8월 급락·에너지 폭락)에는 고변동·복권형에서 손절이 조금 이겼다. "
      f"SMA50은 {min(rc['SMA50 이탈'].values()):+.2f}~{max(rc['SMA50 이탈'].values()):+.2f}로 불안정하다. '어디서 덜 아픈가'는 안정적이고, '어디서 이기나'는 없다.")
    A("- **표본 겹침**: 2026 앞은 진입 월이 3개(1월~3월 초), 뒤는 2개(6월~7월 중순)라 월 묶음 t가 약하다. 주 묶음 t는 참고용이다(60일 보유가 겹쳐 독립이 아니다). "
      "같은 종목을 10거래일마다 사서 60일 들기 때문에 거래가 6겹으로 겹친다. 거래 수는 실제 정보량을 부풀린다.")
    A(f"- **미끄러짐**: 엔진은 진입과 손절 체결에 0.5%를 물리고, 보유 끝 매도에는 물리지 않는다. 그래서 손절 쪽이 0.5% × 발동률만큼 더 불리하다(손절 체결이 더 나쁘다는 가정). "
      f"아무 종목 추적 −20% 앞 짝차: {pp(gv('전체', '추적-20%', '2026앞'))} → {pp(gv('전체', '추적-20%', '2026앞', '짝차_미끄러짐'))}. 수수료는 0이다(사용자 매매 조건).")
    A("- **SMA50 규칙의 실체**: 무작위 진입의 93~98%에서 걸려 평균 10~18일만 들고 있었다. 50일선 아래에서 산 거래는 다음 날 판다. 그래서 사실상 '50일선 위에서만 보유'하는 추세 필터다.")
    A("- **덜 들기 비교**: 바구니(거래 한 칸씩 매일 같은 비중)의 최대 낙폭 한 숫자에 기댄다. 2022처럼 보유 바구니가 마이너스인 창은 효율선 최선이 현금(0%)이라, '덜 들기 대비'가 곧 손절 바구니 수익이다.")
    A(f"- **자료**: 자료 오류 의심 {meta['자료오류_제외']}종목(하루 +150% 이상 또는 −70% 이하)을 뺐다. 명목 주가는 yfinance 분할 이력({meta['분할이력_종목']}종목)으로 되돌렸다. "
      "분사 등 분할 외 조정(예: GE 1.04)도 비율로 들어가 조금 틀릴 수 있다. 유동성(거래대금)은 시총의 대리이고, 섹터는 사후 분류다.")
    A("- **창 경계**: 2025 진입 중 10~12월분은 2026년 1~3월까지 들고 있어 앞 구간과 기간이 겹친다. 2026 뒤는 10/9까지 60일을 채운 7월 중순 진입까지라 8~10월 진입은 없다. "
      "폭락 시기 창(2018 4분기·2020 1분기·2022 상반기)은 사후로 정한 맥락이다. 현금 이자는 0이다.")
    A("")
    A("## 다시 돌리기")
    A("")
    A("```bash")
    A("cd data/derived/analysis/stop_research/code")
    A("PYTHONIOENCODING=utf-8 py p4_STKTAIL.py        # 분할 이력은 cache/p4_STKTAIL_splits.json(처음 한 번 yfinance)")
    A("```")
    return "\n".join(L)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg.startswith("render"):
        rerender("_quick" if arg == "render_quick" else "")
    else:
        main(quick=arg == "quick")
