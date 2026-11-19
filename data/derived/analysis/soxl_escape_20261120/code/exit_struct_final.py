"""갈래 exit_struct 최종 정리: TRAIN primary 순 상위 3(신호가 같은 중복 설정은 하나로) + PRE 귀무 보정 (2026-11-20).
실행 순서: exit_struct_scan.py → exit_struct_retest.py → exit_struct_null.py → exit_struct_final.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402
import exit_struct_scan as S  # noqa: E402
import exit_struct_null as N  # noqa: E402


def main():
    with open(S.OUT, encoding="utf-8") as fh:
        res = json.load(fh)
    allc = {}
    for st in ("stage1", "stage2", "stage3", "stage4"):
        allc.update(res[st])
    res["configs_by_stage"] = {st: len(res[st]) for st in ("stage1", "stage2", "stage3", "stage4")}
    res["configs_tested"] = len(allc)
    elig = sorted([k for k, v in allc.items() if S.ok_freq(v)], key=lambda k: -allc[k]["train"]["primary"])
    picked, seen = [], set()
    for k in elig:
        sig = json.dumps([allc[k]["train"], allc[k]["test"]], sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        picked.append(k)
        if len(picked) == 3:
            break
    base = N.signals()
    final = {}
    for k in picked:
        bars, sig = N.resolve(k, base)
        pos = H.exec_index(sig, bars)
        pos = pos[pos >= 0]
        nl = dict(res["null"].get(k, {}))
        nl["pre"] = N.null_stats(pos, H.PRE, allc[k]["pre"]["primary"])
        final[k] = {**allc[k], "null": nl}
    res["final_top3"] = final
    res["final_rule"] = "TRAIN hold_sim x_rand_mean% 내림차순, 조건 TRAIN n>=10·하루 0.3~6회, 같은 신호 중복 제거. TEST로 재선택 없음."
    with open(S.OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    for k, v in final.items():
        print(k)
        for nm in ("train", "test", "pre"):
            print("  ", nm, json.dumps(v[nm]), v["null"].get(nm))


if __name__ == "__main__":
    main()
