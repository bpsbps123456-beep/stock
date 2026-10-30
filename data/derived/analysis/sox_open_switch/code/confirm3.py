"""BRIEF3 확인 단계 — 조건 겹치기(L1·L2·L3)에서 얼린 후보 전부를 한 번 잰다. 세 묶음이 모두 얼린 뒤에만 실행한다.

무엇을 재나
- 후보: results/lay_L1_frozen.json, lay_L2_frozen.json, lay_L3_frozen.json 의 frozen 항목 전부(6개).
  factory 문자열 'lay_Lk.make({...})'을 ast.parse 로 풀어(호출 꼴·모듈 이름 확인) 인자 dict 는 상수 마디(문자열·숫자·None·
  참거짓)만 받아 파이썬 값으로 옮기고, 그 모듈의 make(dict)를 그대로 부른다(코드 실행 없음, 얼린 코드·매개변수 그대로).
  factory 오류는 고치지 않고 적는다.
- 구간 sel·conf·aux × 마감 w90(매일 11:00)·std(매일 10:00) × 판
  base = 비용 1배·다음 봉 종가 체결, cost2x = 비용 2배·종가 체결, open = 비용 1배·다음 봉 시가 체결(지연 0, 감도).
  conf 는 base 를 두 반 h1(2026-07-01~08-31)·h2(2026-09-01~10-28)로 따로 잰다.
- 거울: MirrorWrap(안쪽 전략에는 ctx.pos 를 뒤집어 보여 주고, 낸 Order.pos 부호를 바꾼다. 목표·손절 비율 그대로).
  세 묶음 전략은 ctx.pos 를 참·거짓으로만 보고 ctx.entry_px 를 쓰지 않아(L2 시간 손절은 ctx.entry_i) 래퍼가 뜻을 지킨다.
  각 모듈의 make(p, mirror=True)(고유 거울)도 base 로 재서 래퍼와 날별 최대 차를 적는다(같아야 한다).
  방향 성분 = (전략 − 거울)/2, 날별.

결과 보기 전에 정한 해석(바꾸지 않는다)
1. 판정(마감 방식마다, BRIEF3 '판정'): 지표는 conf base 날별 수익(%, 84일 전부, 매매 없는 날 0).
   후보 = 평균 > 0 & 날 부트스트랩(10,000회, seed 20261029) 95% 하한 > 0 & Holm 보정 단측 p ≤ 0.05
          & h1·h2 평균 모두 > 0 & cost2x 평균 > 0.
   관찰 = 후보가 아니고 평균 > 0 & h1·h2 평균 모두 > 0.
   탈락 = 나머지.
2. Holm: 마감 방식별로 얼린 후보 전체(m = 6, 그 방식으로 얼리지 않은 후보도 넣는다)의 conf base 단측 부트스트랩 p
   (lib.summarize p_one_sided = 부트스트랩 평균 ≤ 0 비율)를 step-down 보정한다.
3. open(시가 체결)·거울·aux(2025)·sel 은 판정에 쓰지 않는다(감도·장세 전환 위험·재현 점검).
4. sel 재현: 얼린 파일의 sel 평균과 다시 잰 값의 차가 1e-9 넘으면 sel_repro 에 적는다(고치지 않는다).

사용: cd code && py confirm3.py → results/confirm3.json, results/confirm3_daily_<id>_<scheme>.csv,
      results/confirm3_trades_conf_<id>_<scheme>.csv
"""
import ast
import hashlib
import json
import os
import re
import sys
import time
import traceback

os.environ["SOS_UNLOCK"] = "frozen"          # 확인 단계만 conf·aux 를 연다(L1·L2·L3 모두 얼린 뒤)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                            # noqa: E402
import pandas as pd                           # noqa: E402

from lib import Data, Engine, Order, INST, PERIODS, cutoff_idx, summarize   # noqa: E402
import lay_L1, lay_L2, lay_L3                 # noqa: E402,E401

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
FILES = ["lay_L1", "lay_L2", "lay_L3"]
MODS = {"lay_L1": lay_L1, "lay_L2": lay_L2, "lay_L3": lay_L3}
SCHEMES = ["w90", "std"]
RUN_PERIODS = ["sel", "conf", "aux"]
VARIANTS = {"base": dict(cost_mult=1.0, fill="close"), "cost2x": dict(cost_mult=2.0, fill="close"),
            "open": dict(cost_mult=1.0, fill="open")}
HALVES = {"h1": ("2026-07-01", "2026-08-31"), "h2": ("2026-09-01", "2026-10-28")}
ALPHA = 0.05
NBOOT, SEED = 10000, 20261029


def const_value(node):
    """상수 마디만 파이썬 값으로(문자열·숫자·None·참거짓, 음수 부호 허용). 다른 마디는 거부."""
    if isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, (str, int, float, bool))):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) \
            and isinstance(node.operand.value, (int, float)) and not isinstance(node.operand.value, bool):
        return -node.operand.value
    raise ValueError(f"상수가 아닌 마디: {ast.dump(node)}")


def build_factory(s):
    """'lay_Lk.make({...})' → (그 모듈의 make(dict), dict, 모듈). 코드는 실행하지 않는다."""
    tree = ast.parse(s.strip(), mode="eval")
    call = tree.body
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "make"
            and isinstance(call.func.value, ast.Name) and call.func.value.id in MODS
            and len(call.args) == 1 and not call.keywords and isinstance(call.args[0], ast.Dict)):
        raise ValueError(f"factory 꼴이 아니다: {s!r}")
    dn = call.args[0]
    params = {}
    for k, v in zip(dn.keys, dn.values):
        key = const_value(k)
        if not isinstance(key, str):
            raise ValueError(f"dict 키가 문자열이 아니다: {s!r}")
        params[key] = const_value(v)
    mod = MODS[call.func.value.id]
    return mod.make(params), params, mod


class MirrorWrap:
    """SOXL↔SOXS 뒤집기. 안쪽 전략에는 ctx.pos 를 뒤집어 보여 주고 Order.pos 부호를 바꾼다(목표·손절 비율 그대로)."""

    def __init__(self, inner):
        self.inner = inner
        self.start = getattr(inner, "start", "09:30")
        self.dropped = 0

    def on_day(self, ctx):
        if hasattr(self.inner, "on_day"):
            ctx.pos = -ctx.pos
            try:
                self.inner.on_day(ctx)
            finally:
                ctx.pos = -ctx.pos

    def on_bar(self, ctx, i):
        ctx.pos = -ctx.pos
        try:
            o = self.inner.on_bar(ctx, i)
        finally:
            ctx.pos = -ctx.pos
        if o is None:
            return None
        if o.pos != 0 and not np.isfinite(ctx.bars(INST[-o.pos], "Close")[i]):
            self.dropped += 1
            return None
        return Order(-o.pos, stop=o.stop, target=o.target, keep=o.keep)


def wrap_mirror(make):
    return lambda: MirrorWrap(make())


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def safe_id(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def load_candidates():
    cands, errors, fams = [], [], {}
    for fn in FILES:
        path = os.path.join(RES, f"{fn}_frozen.json")
        j = json.load(open(path, encoding="utf-8"))
        fams[fn] = {"family": j["family"], "n_configs": j["n_configs"], "file": f"results/{fn}_frozen.json",
                    "sha256_16": sha(path)}
        for e in j["frozen"]:
            try:
                make, fp, mod = build_factory(e["factory"])
                make()
                params_match = fp == {k: v for k, v in e["params"].items() if k in fp} and set(fp) == set(e["params"])
            except Exception as ex:                                 # 고치지 않고 기록만 한다
                errors.append({"file": fn, "id": e.get("id"), "factory": e.get("factory"),
                               "error": repr(ex), "trace": traceback.format_exc(limit=3)})
                continue
            cands.append({"family": j["family"], "module": fn, "id": str(e["id"]), "file_id": safe_id(str(e["id"])),
                          "picked_for": str(e["scheme"]), "params": e["params"], "factory": e["factory"],
                          "factory_params": fp, "params_match": bool(params_match),
                          "make": make, "mirror": wrap_mirror(make), "native_mirror": mod.make(fp, mirror=True),
                          "frozen_sel_mean": {s: (e.get(f"sel_{s}") or {}).get("mean_pct") for s in SCHEMES},
                          "n_configs": j["n_configs"]})
    return cands, errors, fams


def holm(ps):
    """ps: {id: p}. 반환 {id: Holm 보정 p}(step-down, 단조)."""
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for j, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - j) * p))
        out[k] = run
    return out


def slim(s):
    keys = ["days", "active_days", "trades", "mean_pct", "sd_pct", "hit_days", "sum_pct", "compound_pct", "t",
            "ci95", "p_one_sided", "max_dd_pct"]
    return {k: s.get(k) for k in keys if k in s}


def hhmm_min(x):
    return int(x[:2]) * 60 + int(x[3:])


def trade_mix(tr, daily):
    if not len(tr):
        return {"trades": 0}
    miss = tr[tr["why"] != "target"]
    mins = [hhmm_min(b) - hhmm_min(a) for a, b in zip(tr["entry"], tr["exit"])]
    r = daily["ret"].values * 100
    top3 = np.sort(r)[::-1][:3].sum()
    return {"trades": int(len(tr)), "why": {k: int(v) for k, v in tr["why"].value_counts().items()},
            "by_sym": {k: int(v) for k, v in tr["sym"].value_counts().items()},
            "trade_mean_pct": float(tr["ret"].mean() * 100),
            "target_rate": float((tr["why"] == "target").mean()),
            "miss_mean_pct": float(miss["ret"].mean() * 100) if len(miss) else None,
            "worst_trade_pct": float(tr["ret"].min() * 100), "best_trade_pct": float(tr["ret"].max() * 100),
            "hold_min_mean": float(np.mean(mins)),
            "trade_mean_by_sym_pct": {k: float(g["ret"].mean() * 100) for k, g in tr.groupby("sym")},
            "top3_days_sum_pct": float(top3), "total_sum_pct": float(r.sum()),
            "top3_share_of_sum": float(top3 / r.sum()) if r.sum() > 0 else None,
            "sum_without_top3_pct": float(r.sum() - top3)}


def in_half(dl, h):
    a, b = HALVES[h]
    return (dl["day"] >= pd.Timestamp(a).date()) & (dl["day"] <= pd.Timestamp(b).date())


def boot_ci(x, nboot=NBOOT, seed=SEED):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (nboot, len(x)))].mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], float((bs <= 0).mean())


def main():
    t0 = time.time()
    hashes = {f: sha(os.path.join(HERE, f)) for f in ["lib.py", "lay_L1.py", "lay_L2.py", "lay_L3.py", "confirm3.py"]}
    D = Data()
    print(f"자료 {time.time() - t0:.1f}s, 날 {len(D.days)}", flush=True)
    cands, errors, fams = load_candidates()
    print(f"후보 {len(cands)}개, factory 오류 {len(errors)}", flush=True)
    for c in cands:
        print("  ", c["module"], c["id"], c["picked_for"], c["factory_params"], "params_match", c["params_match"], flush=True)
    E = {(s, v): Engine(D, scheme=s, **kw) for s in SCHEMES for v, kw in VARIANTS.items()}
    pdays = {p: D.days_in(p) for p in RUN_PERIODS}
    for p in RUN_PERIODS:
        print(f"  {p}: {len(pdays[p])}일 {D.days[pdays[p][0]]}~{D.days[pdays[p][-1]]}", flush=True)

    def run(make, s, v, p):
        dl, tr = E[(s, v)].run(make, period=p)
        bad = int((~np.isfinite(dl["ret"].values)).sum())
        assert bad == 0, f"nan 수익 {bad}일"
        return dl, tr

    out = []
    for c in cands:
        rec = {k: c[k] for k in ["family", "module", "id", "file_id", "picked_for", "params", "factory",
                                 "factory_params", "params_match", "n_configs"]}
        rec.update(results={}, halves={}, mirror={}, mirror_native_check={}, direction={}, trades={}, sel_repro={})
        for s in SCHEMES:
            R = rec["results"][s] = {}
            keep, trs = {}, {}
            for p in RUN_PERIODS:
                R[p] = {}
                for v in VARIANTS:
                    dl, tr = run(c["make"], s, v, p)
                    R[p][v] = slim(summarize(dl, NBOOT, SEED))
                    keep[(p, v)] = dl
                    trs[(p, v)] = tr
            rec["trades"][s] = {p: trade_mix(trs[(p, "base")], keep[(p, "base")]) for p in RUN_PERIODS}
            fm = c["frozen_sel_mean"][s]
            rr = R["sel"]["base"]["mean_pct"]
            rec["sel_repro"][s] = {"frozen_mean_pct": fm, "rerun_mean_pct": rr,
                                   "abs_diff": None if fm is None else abs(fm - rr),
                                   "ok": None if fm is None else abs(fm - rr) <= 1e-9}
            dc, c2 = keep[("conf", "base")], keep[("conf", "cost2x")]
            rec["halves"][s] = {}
            for h in HALVES:
                hs = slim(summarize(dc[in_half(dc, h)].reset_index(drop=True), NBOOT, SEED))
                hs["cost2x_mean_pct"] = float(c2[in_half(c2, h)]["ret"].mean() * 100)
                rec["halves"][s][h] = hs
            mir_d = {}
            rec["mirror"][s], rec["mirror_native_check"][s], rec["direction"][s] = {}, {}, {}
            for p in RUN_PERIODS:
                dm, trm = run(c["mirror"], s, "base", p)
                mir_d[p] = dm
                ms = slim(summarize(dm, NBOOT, SEED))
                ms["trade_mean_pct"] = float(trm["ret"].mean() * 100) if len(trm) else None
                rec["mirror"][s][p] = ms
                dn, _ = run(c["native_mirror"], s, "base", p)
                rec["mirror_native_check"][s][p] = {
                    "native_mean_pct": float(dn["ret"].mean() * 100),
                    "max_abs_daily_diff_vs_wrapper_pct": float(np.max(np.abs(dn["ret"].values - dm["ret"].values)) * 100)}
                b0 = keep[(p, "base")]
                assert list(b0["day"]) == list(dm["day"])
                dirv = (b0["ret"].values - dm["ret"].values) / 2 * 100
                ci, pp = boot_ci(dirv)
                rec["direction"][s][p] = {"mean_pct": float(dirv.mean()), "ci95": ci, "p_one_sided": pp,
                                          "nondir_mean_pct": float(((b0["ret"].values + dm["ret"].values) / 2 * 100).mean())}
            # 날별 CSV(sel·conf·aux)
            rows = []
            for p in RUN_PERIODS:
                b0 = keep[(p, "base")]
                assert list(b0["day"]) == list(keep[(p, "cost2x")]["day"]) == list(keep[(p, "open")]["day"])
                for k in range(len(b0)):
                    day = b0["day"].iloc[k]
                    ci_ = cutoff_idx(day, s)
                    half = ""
                    if p == "conf":
                        half = "h1" if day <= pd.Timestamp(HALVES["h1"][1]).date() else "h2"
                    rows.append({"day": day, "period": p, "half": half,
                                 "cutoff_et": f"{(ci_ + 240) // 60:02d}:{(ci_ + 240) % 60:02d}",
                                 "ret_pct": b0["ret"].iloc[k] * 100, "n": int(b0["n"].iloc[k]),
                                 "ret_cost2x_pct": keep[(p, "cost2x")]["ret"].iloc[k] * 100,
                                 "ret_open_pct": keep[(p, "open")]["ret"].iloc[k] * 100,
                                 "mirror_ret_pct": mir_d[p]["ret"].iloc[k] * 100,
                                 "direction_pct": (b0["ret"].iloc[k] - mir_d[p]["ret"].iloc[k]) / 2 * 100})
            pd.DataFrame(rows).to_csv(os.path.join(RES, f"confirm3_daily_{c['file_id']}_{s}.csv"), index=False,
                                      encoding="utf-8-sig")
            trs[("conf", "base")].to_csv(os.path.join(RES, f"confirm3_trades_conf_{c['file_id']}_{s}.csv"),
                                         index=False, encoding="utf-8-sig")
        out.append(rec)
        print(f"  {c['id']} 끝 {time.time() - t0:.0f}s", flush=True)

    # ── Holm·판정 ──
    holm_tab, verdicts = {}, {}
    for s in SCHEMES:
        ps = {r["id"]: r["results"][s]["conf"]["base"]["p_one_sided"] for r in out}
        adj = holm(ps)
        holm_tab[s] = {k: {"p_one_sided": ps[k], "holm_p": adj[k]} for k in ps}
        for r in out:
            cb = r["results"][s]["conf"]["base"]
            chk = {"mean_gt0": cb["mean_pct"] > 0, "ci_lo_gt0": cb["ci95"][0] > 0, "holm_le_0.05": adj[r["id"]] <= ALPHA,
                   "h1_gt0": r["halves"][s]["h1"]["mean_pct"] > 0, "h2_gt0": r["halves"][s]["h2"]["mean_pct"] > 0,
                   "cost2x_gt0": r["results"][s]["conf"]["cost2x"]["mean_pct"] > 0}
            if all(chk.values()):
                v = "후보"
            elif chk["mean_gt0"] and chk["h1_gt0"] and chk["h2_gt0"]:
                v = "관찰"
            else:
                v = "탈락"
            r.setdefault("verdict", {})[s] = v
            r.setdefault("verdict_checks", {})[s] = {**chk, "holm_p": adj[r["id"]], "m": len(ps)}
            verdicts.setdefault(r["id"], {})[s] = v

    res = {"stage": "확인(BRIEF3 '조건 겹치기' 판정)", "brief": "BRIEF3.md", "unlock": "SOS_UNLOCK=frozen(이 스크립트 안에서만)",
           "periods": {p: PERIODS[p] for p in RUN_PERIODS}, "period_days": {p: len(pdays[p]) for p in RUN_PERIODS},
           "conf_halves": HALVES, "variants": VARIANTS, "bootstrap": {"n": NBOOT, "seed": SEED},
           "conf_second_use": "conf 구간은 앞 연구(BRIEF2)가 되돌림 규칙 F4를 한 번 잰 2차 사용 구간",
           "rules_fixed_before_results": __doc__, "code_sha256_16": hashes, "frozen_files": fams,
           "factory_errors": errors, "n_frozen": len(out),
           "holm": holm_tab, "verdicts": verdicts, "candidates": out,
           "daily_files": {r["id"]: {s: f"results/confirm3_daily_{r['file_id']}_{s}.csv" for s in SCHEMES} for r in out},
           "runtime_s": round(time.time() - t0, 1)}
    with open(os.path.join(RES, "confirm3.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # ── 화면 요약 ──
    for r in out:
        print(f"\n{r['module']} {r['id']} 얼린 방식 {r['picked_for']} sel 재현 "
              f"{ {s: r['sel_repro'][s]['abs_diff'] for s in SCHEMES} }")
        for s in SCHEMES:
            R = r["results"][s]
            cb = R["conf"]["base"]
            print(f"  [{s}] 판정 {r['verdict'][s]}  conf mean {cb['mean_pct']:+.3f} t {cb['t']:+.2f} "
                  f"ci [{cb['ci95'][0]:+.3f},{cb['ci95'][1]:+.3f}] p {cb['p_one_sided']:.4f} "
                  f"holm {holm_tab[s][r['id']]['holm_p']:.4f} act {cb['active_days']}/{cb['days']} n {cb['trades']} "
                  f"| h1 {r['halves'][s]['h1']['mean_pct']:+.3f} h2 {r['halves'][s]['h2']['mean_pct']:+.3f} "
                  f"| 2x {R['conf']['cost2x']['mean_pct']:+.3f} open {R['conf']['open']['mean_pct']:+.3f} "
                  f"| mirror {r['mirror'][s]['conf']['mean_pct']:+.3f} dir {r['direction'][s]['conf']['mean_pct']:+.3f}")
            print(f"         sel {R['sel']['base']['mean_pct']:+.3f} t {R['sel']['base']['t']:+.2f} "
                  f"| aux mean {R['aux']['base']['mean_pct']:+.3f} t {R['aux']['base']['t']:+.2f} "
                  f"ci [{R['aux']['base']['ci95'][0]:+.3f},{R['aux']['base']['ci95'][1]:+.3f}] "
                  f"2x {R['aux']['cost2x']['mean_pct']:+.3f} dd {R['aux']['base'].get('max_dd_pct', 0):+.1f} "
                  f"mirror {r['mirror'][s]['aux']['mean_pct']:+.3f} "
                  f"| native 거울 차 {max(v['max_abs_daily_diff_vs_wrapper_pct'] for v in r['mirror_native_check'][s].values()):.2e}")
            tm = r["trades"][s]["conf"]
            if tm.get("trades"):
                print(f"         conf 매매 {tm['trades']} 목표 {tm['target_rate']:.2f} 매매 평균 {tm['trade_mean_pct']:+.3f} "
                      f"못 닿음 {tm['miss_mean_pct']} 최악 {tm['worst_trade_pct']:+.2f} 상위3일 몫 {tm['top3_share_of_sum']} "
                      f"by_sym {tm['by_sym']}")
    print(f"\n끝 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
