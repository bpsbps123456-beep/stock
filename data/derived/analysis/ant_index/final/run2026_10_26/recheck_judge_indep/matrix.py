"""명세 선택(동결 문서가 정하지 않은 실행 세부)별로 recheck.py를 돌려 판정 범주를 모은다."""
import itertools, json, subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
combos = []
for samp, rv, m0c, cal, ic in itertools.product(["own", "harness"], ["rms", "std"], ["base", "flags4", "all24"], ["after", "same_day", "none"], ["partial", "plain"]):
    tag = f"m_{samp}_{rv}_{m0c}_{cal}_{ic}"
    a = [sys.executable, os.path.join(HERE, "recheck.py"), "--tag", tag, "--rv", rv, "--ic_mode", ic,
         "--placebo_mode", "same_test" if ic == "partial" else "plain"]
    if samp == "harness": a.append("--harness_sample")
    if m0c == "flags4": a.append("--m0_extra_cal")
    if m0c == "all24": a.append("--m0_all24")
    if cal == "none": a.append("--no_cal")
    else: a += ["--cal_map", cal]
    combos.append((tag, a, dict(sample=samp, rv=rv, m0_cal_flags=m0c, cal=cal, ic=ic)))
def run(c):
    tag, a, spec = c
    subprocess.run(a, cwd=HERE, capture_output=True, text=True, check=True)
    d = json.load(open(os.path.join(HERE, f"recheck_{tag}.json"), encoding="utf-8"))
    os.remove(os.path.join(HERE, f"recheck_{tag}.json"))
    row = dict(spec)
    for cid, c_ in d["cells"].items():
        row[cid] = {"verdict": c_["verdict"], "t": round(c_["t"], 3), "p_an": round(c_["p_an"], 4), "why": c_["why"], "n": c_["n"], "est": round(c_["est"], 4), "p_final": round(c_["p_final"], 4),
                    "placebo_ratio": None if c_["placebo"]["ratio"] is None else round(c_["placebo"]["ratio"], 3),
                    "robust": {k: round(v["ratio"], 3) for k, v in c_["robust"].items()}, "holm_ok": c_.get("holm_ok"),
                    "front": None if c_["est_front"] is None else round(c_["est_front"], 4)}
    return row
with ThreadPoolExecutor(4) as ex:
    rows = list(ex.map(run, combos))
json.dump(rows, open(os.path.join(HERE, "matrix_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
from collections import Counter
for cid in ("H03", "H08", "H70", "H66"):
    print(cid, Counter(r[cid]["verdict"] for r in rows))
