"""보조: 2025-07~10월에 원안·앞 1위 칸을 그대로 적용(판정 모집단·E2). 장세 전환 위험 확인용.
사용: py build.py alp1m_2025 2025-07-01 2025-10-31 → py aux2025.py → results/aux2025.json"""
import os, json
import numpy as np
import evaluate as V


def main():
    picks = json.load(open(os.path.join(V.RES, "pick.json"), encoding="utf-8"))
    cells = {"orig_A": ("A", V.ORIG["A"]), "orig_B": ("B", V.ORIG["B"])}
    for pat in ("A", "B"):
        if picks.get(pat):
            cells[f"top_{pat}"] = (pat, V.spec_of(picks[pat], pat))
    rng = np.random.default_rng(2025)
    out = {}
    for name, (pat, spec) in cells.items():
        sig, ctx = V.cell_rows("cand_alp1m_2025", pat, spec)
        out[name] = {"sig": V.summary(sig), "ctx": V.summary(ctx), "boot": V.boot(sig, ctx, rng)}
        print(name, out[name]["sig"].get("n"), out[name]["sig"].get("r1net_e2"))
    json.dump(out, open(os.path.join(V.RES, "aux2025.json"), "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
