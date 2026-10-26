"""verify_gate_2026.py 보조: 16개 전부를 결정 요일 월~금(skip/roll)으로 다시 돌려 앞·전체 1위가 요일에 따라 바뀌는지 본다.
실행: $env:PYTHONIOENCODING='utf-8'; py data/derived/analysis/rsi_breakout/code/verify_gate_2026_weekday16.py
결과: data/derived/analysis/rsi_breakout/results/gate/verify_gate_2026_weekday16.json
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_gate_2026 as G  # noqa: E402

OUT = os.path.join(HERE, "../results/gate/verify_gate_2026_weekday16.json")


def main():
    D = G.Data()
    res = {}
    for sched in ("skip", "roll"):
        for wd in range(5):
            tab = {}
            for base in ("live50", "c774", "nocap", "sub5eq"):
                for gate in ("none", "soxx_w", "soxx_d", "qqq_w"):
                    v, W, ST, sw, rebs, _ = G.simulate(G_D := D, base, gate, wd=wd, sched=sched)
                    m = G.metrics(D, v, W, ST)
                    tab[f"{base}·{gate}"] = {k: m[k] for k in ("앞", "뒤", "전체", "MDD", "9월", "10월")}
            res[f"{sched}·요일{wd}"] = tab
            f = sorted(tab, key=lambda n: -tab[n]["앞"])
            t = sorted(tab, key=lambda n: -tab[n]["전체"])
            print(f"[{sched}] 요일{wd} 앞 1~3: " + ", ".join(f"{n} {tab[n]['앞']*100:+.1f}" for n in f[:3])
                  + f" | live50·none 앞 순위 {f.index('live50·none')+1}"
                  + " || 전체 1~3: " + ", ".join(f"{n} {tab[n]['전체']*100:+.1f}" for n in t[:3])
                  + f" | live50·none 전체 순위 {t.index('live50·none')+1}")
            worse = [f"{b}·{g}" for b in ("live50", "c774", "nocap", "sub5eq") for g in ("soxx_d", "qqq_w")
                     if not all(tab[f"{b}·{g}"][k] < tab[f"{b}·none"][k] for k in ("앞", "뒤", "전체"))]
            print(f"      C4 예외(soxx_d·qqq_w가 앞·뒤·전체 중 하나라도 none 이상): {worse or '없음'}")
            res[f"{sched}·요일{wd}"]["_c4_exceptions"] = worse
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print("저장:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
