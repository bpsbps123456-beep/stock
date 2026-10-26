"""같은 문턱 방식(a: +-0.5, b: 애매 비율 맞춤) 안에서 K0 계열 경계선(바뀜->지표, 선형 보간)을 K3 각 점의 바뀜에서 읽어 차이를 낸다.
범위 밖은 끝 두 점 직선 연장(표시 *). 실행: PYTHONIOENCODING=utf-8 py matched.py"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
m = json.load(open(os.path.join(HERE, "frontier.json"), encoding="utf-8"))["main"]
KEYS = ["dn_down", "up_up", "lag_dn", "rev_share", "dn_down_26", "up_up_26", "lag_dn_26"]
out = {}
for kind in ("a", "b"):
    k0 = sorted([v for v in m.values() if v["fam"] == "K0" and v["kind"] == kind], key=lambda v: v["flips"])
    xs = np.array([v["flips"] for v in k0])
    for v in sorted([v for v in m.values() if v["fam"] == "K3" and v["kind"] == kind], key=lambda v: v["N"]):
        x = v["flips"]
        row = {}
        ext = not (xs[0] <= x <= xs[-1])
        for key in KEYS:
            ys = np.array([w[key] for w in k0])
            if ext:
                i = (0, 1) if x < xs[0] else (-2, -1)
                y = ys[i[0]] + (ys[i[1]] - ys[i[0]]) * (x - xs[i[0]]) / (xs[i[1]] - xs[i[0]])
            else:
                y = float(np.interp(x, xs, ys))
            row[key] = (v[key], y, v[key] - y)
        out[f"K3-{v['N']}-{kind}"] = row
        print(f"K3-{v['N']}-{kind} flips {x:.1f}{' *' if ext else ''}: " + ", ".join(
            f"{k} {a:.3f} vs K0 {b:.3f} ({d:+.3f})" for k, (a, b, d) in row.items()))
json.dump(out, open(os.path.join(HERE, "matched.json"), "w", encoding="utf-8"), indent=1)
# 분해: K0-20 -> K3-20 차이 중 '기간 단축'(K0 경계선 이동) 몫과 '기준점 방식' 몫
k020 = m["K0-20-a"]
for name in ("K3-20-a", "K3-20-b"):
    for key in ("dn_down", "up_up", "dn_down_26"):
        tot = m[name][key] - k020[key]
        meth = out[name][key][2]
        print(f"{name} {key}: 전체 {tot*100:+.1f}%p = 기간(같은 바뀜의 K0) {(tot-meth)*100:+.1f} + 기준점 {meth*100:+.1f}")
