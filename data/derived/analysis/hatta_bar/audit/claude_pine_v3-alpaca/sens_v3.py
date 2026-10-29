"""감도 점검: 직역의 한 줄만 바꾼 변형이 신호를 몇 건 바꾸는지 센다(비교가 차이를 잡을 힘이 있는지, Pine 의미가 애매한 곳이 결과에 닿는지).
성과 수치는 계산하지 않는다. 기준 = pine_signals.csv(엔진과 100% 같은 집합).
사용: py sens_v3.py
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parity_v3 as PV  # noqa: E402
import pine_v3 as P  # noqa: E402

SRC = (HERE / "pine_v3.py").read_text(encoding="utf-8")
TMP = HERE / "_variants"
TMP.mkdir(exist_ok=True)

SIG_OLD = "def run(o, h, l, c, v, t, dayNY, hrNY, hrKST,"
SIG_NEW = "def run(o, h, l, c, v, t, dayNY, hrNY, hrKST, ntr,"
VARIANTS = {
    # ta.lowest 가 length 봉 전에도 있는 봉만으로 최저가를 낸다고 가정
    "lowest_partial": [("    if j < 0 or j < length - 1:\n        return NAN\n    mn = l[j]\n    for q in range(j - length + 1, j + 1):",
                        "    if j < 0:\n        return NAN\n    mn = l[j]\n    for q in range(max(0, j - length + 1), j + 1):")],
    # v2 의 사슬 묶음(바로 앞 점과 0.15%)
    "level_chain": [("                    if lc == 0 or x > anchor * (1 + 0.0015):\n                        lc += 1\n                        anchor = x",
                     "                    if lc == 0 or x > anchor * (1 + 0.0015):\n                        lc += 1\n                    anchor = x")],
    # 직전 20봉 평균 0 이면 무한대(거래량 > 0)
    "volR_inf": [("        volR[bi] = v[bi] / vAvg if vAvg > 0 else NAN",
                  "        volR[bi] = v[bi] / vAvg if vAvg > 0 else (np.inf if (vAvg == 0 and v[bi] > 0) else NAN)")],
    # 한 가격 봉 거르기를 뺌(봉 길이 0.2% 조건이 이미 거르는지)
    "no_single": [("and rng >= minRng / 100 * c[bi] and not single", "and rng >= minRng / 100 * c[bi]")],
    # 스윙 왼쪽 조건 < 를 <= 로
    "swing_left_le": [("isSw = lowp < loR_prev and", "isSw = lowp <= loR_prev and")],
    # 30분 밀도 창을 (t-30, t] 대신 [t-30, t] 로
    "dens_closed": [("            if t[bi - i] > t[bi] - 30 * 60000:", "            if t[bi - i] >= t[bi] - 30 * 60000:")],
    # 백테스트 위생(체결 5건 이상)을 Pine 에 더하면 — 트레이딩뷰 차트가 백테스트보다 더 보이는 신호 수
    "trades5": [(SIG_OLD, SIG_NEW),
                ("sane[bi] = v[bi] * c[bi] >= minNot and", "sane[bi] = ntr[bi] >= 5 and v[bi] * c[bi] >= minNot and")],
}


def build(name, reps):
    s = SRC.replace("@njit(cache=True)", "@njit(cache=False)")
    for a, b in reps:
        assert s.count(a) == 1, (name, a)
        s = s.replace(a, b)
    f = TMP / f"pine_{name}.py"
    f.write_text(s, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"pine_{name}", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    base = pd.read_csv(HERE / "pine_signals.csv")
    key = ["cell", "sym", "ts_et"]
    mods = {k: build(k, r) for k, r in VARIANTS.items()}
    data = {sym: PV.load(sym) for sym in PV.universe()}
    trades = {sym: pd.read_parquet(PV.CACHE / "alp1m" / f"{sym}.parquet")["Trades"].to_numpy(float) for sym in data}
    res = []
    for vn, mod in mods.items():
        rows = []
        for sym, (o, h, l, c, v, t, day, hr, hk, ts) in data.items():
            extra = (trades[sym],) if vn == "trades5" else ()
            for cell, ov in PV.CELLS.items():
                prm = dict(P.DEFAULTS); prm.update(ov); prm["dedup30"] = False
                kA, kB, *_ = mod.run(o, h, l, c, v, t, day, hr, hk, *extra, *[prm[k] for k in P.ORDER])
                sig = kA if PV.PAT[cell] == "A" else kB
                rows += [(cell, sym, ts[i]) for i in np.flatnonzero(sig)]
        var = pd.DataFrame(rows, columns=key)
        m = base[key].merge(var, on=key, how="outer", indicator=True)
        m["pre"] = m.ts_et.str[11:] < "09:30"
        for cell in PV.CELLS:
            mm = m[m.cell == cell]
            res.append(dict(variant=vn, cell=cell, removed=int((mm._merge == "left_only").sum()),
                            added=int((mm._merge == "right_only").sum()),
                            removed_pre=int(((mm._merge == "left_only") & mm.pre).sum()),
                            added_pre=int(((mm._merge == "right_only") & mm.pre).sum())))
        print(vn, "done", flush=True)
    r = pd.DataFrame(res)
    r.to_csv(HERE / "sensitivity.csv", index=False)
    print(r.pivot_table(index="variant", values=["removed", "added", "removed_pre", "added_pre"], aggfunc="sum").to_string())
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
