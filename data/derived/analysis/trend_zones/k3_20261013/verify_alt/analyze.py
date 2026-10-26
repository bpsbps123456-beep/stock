"""frontier.json -> 인지-깜빡임 경계선 비교(보조 훑기 포함). 실행: PYTHONIOENCODING=utf-8 py analyze.py"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "frontier.json"), encoding="utf-8"))
sw = r["sweep"]
fam = {"K0": [v for v in sw.values() if v["col"].startswith("T0")], "K3": [v for v in sw.values() if v["col"].startswith("T3")]}
print("훑기 점 수:", {k: len(v) for k, v in fam.items()})

def best_at(pts, x, key, lower_better=False):
    c = [p for p in pts if p["flips"] <= x]
    if not c:
        return None, None
    p = (min if lower_better else max)(c, key=lambda p: p[key])
    return p[key], f'{p["col"]}@{p["tau"]}'

for key, lb in (("dn_down", False), ("up_up", False), ("lag_dn", True), ("rev_share", True)):
    print(f"\n[{key}] 바뀜 상한 x에서 각 계열 최선(훑기 105점씩)")
    for x in (50, 55, 60, 65, 67.63, 70, 75, 80, 85, 90):
        a, an = best_at(fam["K0"], x, key, lb)
        b, bn = best_at(fam["K3"], x, key, lb)
        fa = "-" if a is None else f"{a:.4f} ({an})"
        fb = "-" if b is None else f"{b:.4f} ({bn})"
        print(f"  x<={x:6}: K0 {fa:32s} K3 {fb}")

# K3n 대비: K0 계열에서 K3n의 인지를 맞추는 최소 바뀜
k3n = r["main"]["K3-20-a"]
for key in ("dn_down", "up_up"):
    c = [p for p in fam["K0"] if p[key] >= k3n[key]]
    p = min(c, key=lambda p: p["flips"]) if c else None
    print(f"\nK3n {key}={k3n[key]:.4f}, flips={k3n['flips']:.1f}; K0 계열이 같은 인지에 필요한 최소 바뀜:",
          None if p is None else (round(p["flips"], 1), p["col"], p["tau"], round(p["rev_share"], 4), p["lag_dn"]))
# 지연 6일 이하 K0 최소 바뀜
c = [p for p in fam["K0"] if p["lag_dn"] <= k3n["lag_dn"]]
p = min(c, key=lambda p: p["flips"])
print("K0 계열 급락 지연<=6 최소 바뀜:", round(p["flips"], 1), p["col"], p["tau"], round(p["dn_down"], 4))
c = [p for p in fam["K3"] if p["lag_dn"] <= k3n["lag_dn"]]
p = min(c, key=lambda p: p["flips"])
print("K3 계열 급락 지연<=6 최소 바뀜:", round(p["flips"], 1), p["col"], p["tau"], round(p["dn_down"], 4))
# 역행 비율 범위
for f, pts in fam.items():
    v = [p["rev_share"] for p in pts]
    print(f, "역행 비율 범위(훑기 전체):", round(min(v), 4), round(max(v), 4))
