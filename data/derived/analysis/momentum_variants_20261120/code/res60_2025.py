"""2025 하반기 보조: 잔차 60일 vs 120일 수익을 펀드 밖 종목군(AI·WIDE200·WIDETECH50, 지금 목록이라 둘 다 같은 생존 편향)에서 짝 비교.
결과: results/res60_2025.json
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "code"))
sys.path.insert(0, str(BASE / "branches/composite"))
import engine as E  # noqa: E402
import composite as C  # noqa: E402


def main():
    e = E.Engine()
    res60 = C.residual(e, 60)
    out = {}
    for u in ("AI", "WIDE200", "WIDETECH50"):
        for nm, sc in (("잔차60", res60), ("수익120", e.ret(120)), ("수익60", e.ret(60))):
            r = e.run(sc, universe=u, n=7, buffer=5, weight="equal", caps=True, start="2025-07-01", end="2025-12-31")
            out[f"{u}|{nm}"] = {k: round(v, 4) for k, v in e.metrics_2025(r).items()}
    q = e.bench("QQQ", start="2025-07-01", end="2025-12-31")
    out["QQQ"] = {k: round(v, 4) for k, v in e.metrics_2025(q).items()}
    (BASE / "results/res60_2025.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
