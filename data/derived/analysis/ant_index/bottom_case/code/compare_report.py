"""바닥 사례 비교(설명용) 보고서: results/compare_tables.json → results/compare.md. 판정 아님.

실행: py compare_report.py (compare.py 다음)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BC = HERE.parent
ANT = BC.parent
OUT = BC / "results"

PCT = {"new30_share", "missing_rate", "P_share", "P_strong1", "neutral_share", "mixed_share"}
SIGNED = {"P_bar"}
WD = "월화수목금토일"


def fmt(k: str, v) -> str:
    if v is None:
        return "—"
    if k.endswith("_A") or k.startswith("share_") or k in PCT:
        return f"{v * 100:.1f}%"
    if k in SIGNED:
        return f"{v:+.2f}"
    return f"{v:.2f}"


def md_date(s: str) -> str:
    m, d = int(s[5:7]), int(s[8:10])
    return f"{m}/{d}"


def yn(v) -> str:
    return "—" if v is None else ("예" if v else "아니오")


def disp(w) -> str:
    """'바닥 쪽'은 '7월 바닥과 같은 방향'이라는 뜻이다. 그날이 바닥이라는 말로 읽히지 않게 '7월 바닥 쪽'으로 쓴다."""
    return "7월 바닥 쪽" if w == "바닥 쪽" else (w or "")


def main() -> int:
    r = json.loads((OUT / "compare_tables.json").read_text("utf-8"))
    t117 = json.loads((ANT / "emotion" / "review_test117_compare.json").read_text("utf-8"))
    G, S, O = r["groups"], r["separators"], r["october"]
    sep = {s["key"]: s for s in S}
    cands = [s for s in S if s["status"] == "후보"]
    names = {s["key"]: s["name"] for s in S}
    ch4, ch3 = r["chance"]["4"], r["chance"]["3"]
    L: list[str] = []
    a = L.append

    both_keys = {s["key"] for s in S if s.get("both_days_outside_same_side")}
    from compare import EMO, EMO_KO
    pair_names = [EMO_KO[e] for e in EMO if f"share_{e}" in both_keys and f"mean_{e}" in both_keys]
    npair = len(pair_names)
    dc = r["depth_corr_dd_ref"]["2026-06-22~2026-07-30"]
    bmc = r["base20_ratio_summary"]["MC_A"]
    ex = r["chance_exchange_6days"]
    mx = ex["max_non_bottom"]
    pdays = r["placebo_days_same_side"]
    same = {x["date"]: x["n_same_side"] for x in pdays["by_day"]}
    ge7 = pdays["other_ge7"]
    top_o = max(ge7, key=lambda x: x["n_same_side"])
    soon = [x for x in ge7 if x["min_ret_15td"] is not None and x["min_ret_15td"] < 0]
    notsoon = [x for x in ge7 if x["min_ret_15td"] is not None and x["min_ret_15td"] >= 0]
    bc = r["blind_check"]

    def survived(s):
        ok = s.get("both_days_outside_same_side") and s.get("ext_outside_same_side") and s.get("outside_without_0623") \
            and (s.get("intraday") or {}).get("outside_same_side")
        if s["family"].startswith("라벨"):
            ok = ok and (s.get("boot_keep_outside") or 0) >= 0.8
        return bool(ok)
    surv = [s for s in cands if survived(s)]

    a("# 7월 바닥과 10월 저점의 군중 비교 (설명용)")
    a("")
    a("> 판정이 아니다. 동결 가설(`HYPOTHESES*.md`) 판정과 무관한 사례 비교이고, 매수·매도 추천이 아니다. "
      "비교할 바닥은 7월 하나(7/29·7/30)뿐이다. 10/15는 아직 진행 중인 국면이라 바닥인지 모른다.")
    a(f"> 만든 시각 {r['made']}. 계산 `code/compare.py`, 이 문서 `code/compare_report.py`. 가격은 KOSPI 종가만 썼다.")
    a("")
    a("## 결론")
    a("")
    a("**이 자료로는 10/15가 저점인지 말할 수 없다.** 7월 바닥에서만 달랐던 지표가 우연 수준만큼만 나왔고, 10/15는 그 지표들에서 엇갈린다.")
    a("")
    a(f"- 지표 {r['n_checked']}개를 봤고, 7월 바닥 값이 바닥 아닌 급락 4일의 범위 밖인 후보는 {r['n_candidates']}개다. "
      f"바닥과 급락일이 서로 바꿔도 같은 분포라면 우연만으로 약 {r['expected_outside_by_chance_mean_rule']:.1f}개가 범위 밖에 나온다. "
      f"두 날 모두 같은 쪽 밖인 지표는 {r['n_both_days_outside']}개다. 우연이면 약 {r['expected_both_days_by_chance']:.1f}개다. "
      f"이 {r['n_both_days_outside']}개 가운데 {npair * 2}개는 같은 감정의 몫·세기 쌍({', '.join(pair_names)})이라 따로 센 증거가 아니다.")
    a(f"- 같은 6일(바닥 2일 + 급락 4일)에서 아무 2일을 '바닥'으로 골라 같은 규칙을 돌리면 {ex['n_assign']}가지 가운데 "
      f"{ex['n_assign_out_ge_obs']}가지가 범위 밖 {ex['observed']['n_out']}개 이상, {ex['n_assign_both_ge_obs']}가지가 두 날 모두 밖 "
      f"{ex['observed']['n_both']}개 이상이다(실제 배정 포함). 바닥이 아닌 {md_date(mx['pair'][0])}·{md_date(mx['pair'][1])}를 고르면 "
      f"{mx['n_out']}개·{mx['n_both']}개로 실제 바닥보다 많다. 지표끼리 얽힌 것까지 넣어도 7월 바닥의 모습은 급락일끼리의 차이보다 크지 않다.")
    a(f"- 보조 점검을 모두 버틴 후보는 {len(surv)}개다: " + ", ".join(names[s['key']] for s in surv) + ". "
      "사전 두 개(반대매매·인버스)는 직전 20일 평균 대비 배수로 보면 바닥이 급락일과 다르지 않다. "
      f"7월 하락 구간(거래일 {dc['n_trading_days']}일)에서는 낙폭이 깊을수록 컸다(고점 대비 낙폭과 상관 반대매매 {dc['MC_A']:+.2f}, 인버스 {dc['LS_A']:+.2f}). "
      "라벨 두 개(기쁨·분노)는 검수 시험에서 실패한 감정 칸이다.")
    w15 = {o["key"]: o["cells"]["2026-10-15"]["where"] for o in O}
    w16 = {o["key"]: o["cells"]["2026-10-16"]["where"] for o in O}
    bs15 = [names[k] for k, v in w15.items() if v == "바닥 쪽"]
    ot15 = [f"{names[k]}({v})" for k, v in w15.items() if v != "바닥 쪽"]
    bs16 = [names[k] for k, v in w16.items() if v == "바닥 쪽"]
    a(f"- 10/15는 후보 {len(O)}개에서 갈린다(아래 표). 7월 바닥 쪽: " + ", ".join(bs15) + ". 아닌 쪽: " + ", ".join(ot15) + ". "
      "10/16(+2.0% 반등일) 7월 바닥 쪽: " + ", ".join(bs16) + ".")
    a(f"- **7월 바닥 쪽이 많다는 것은 7월에도 저점 신호가 아니었다.** 후보 {pdays['n_cands']}개 가운데 7월 바닥 쪽인 개수는 "
      f"10/15 {same['2026-10-15']}개, 10/16 {same['2026-10-16']}개다. 바닥 이틀을 뺀 라벨 {pdays['n_other_days']}일 가운데 7개 이상인 날은 "
      f"{len(ge7)}일(" + ", ".join(md_date(x["date"]) for x in ge7) + f")이고, {md_date(top_o['date'])}는 {top_o['n_same_side']}개였다. "
      + "·".join(md_date(x["date"]) for x in soon) + "는 다음 15거래일 안에 더 낮은 종가가 왔다(최저 종가 "
      + ", ".join(f"{x['min_ret_15td'] * 100:+.1f}%" for x in soon) + ")"
      + "".join(f". {md_date(x['date'])}은 {md_date(x['first_lower_close_date'])}까지 더 낮은 종가가 없었다" for x in notsoon)
      + ". 10/16은 뒤 가격이 아직 없다. 후보를 7월 바닥에서 골랐으니 바닥 이틀은 정의상 크게 나온다.")
    a("- 검수를 통과한 예측 칸(군중이 말한 앞날 방향)은 7월에도 바닥과 급락일을 가르지 못했다. "
      f"10월 군중 예측은 7월 어느 묶음보다 하락 쪽이다(10월 하락 중 P̄ {fmt('P_bar', G[5]['ind']['P_bar']['value'])}, "
      f"10/15 {fmt('P_bar', G[6]['ind']['P_bar']['value'])}, 7월 바닥 {fmt('P_bar', G[0]['ind']['P_bar']['value'])}). "
      "미리보기에서 군중 예측은 지난 1일 수익을 따라갔다(+0.46). 그래서 이 값도 바닥 신호로 읽지 않는다.")
    a("")

    # ---- 10월 표
    a("### 10월 날들이 후보 지표에서 어느 쪽인가 (지표마다 따로, 합친 점수 없음)")
    a("")
    a("'7월 바닥 쪽' = 급락 4일 범위를 벗어나 7월 바닥과 같은 방향. 그날이 바닥이라는 뜻이 아니다(위 글머리의 7/14). "
      "'반대쪽' = 범위를 반대 방향으로 벗어남. 값은 하루 창(그날 08:00~다음 날 07:59) 기준이다.")
    a("")
    a("| 지표 | 7월 바닥(2일 평균) | 급락 4일 범위 | 10월 하락 중(10/6~10/14 평균) | 10/15 | 10/16(금) | 10/17(토, 16시간) |")
    a("|---|---|---|---|---|---|---|")
    for o in O:
        k = o["key"]
        c = o["cells"]

        def cell(cc):
            return f"{fmt(k, cc['value'])} {disp(cc['where'])}" if cc["value"] is not None else "—"
        a(f"| {o['name']} | {fmt(k, o['bottom_mean'])} | {fmt(k, o['drop_min'])}~{fmt(k, o['drop_max'])} | "
          f"{cell(c['oct_fall_mean'])} | {cell(c['2026-10-15'])} | {cell(c['2026-10-16'])} | {cell(c['2026-10-17'])} |")
    a("")
    a("- 사전 지표는 그날 모든 유효 작성자를 센 값이다(10/15 작성자 "
      f"{G[6]['ind']['MC_A']['n_units']:,}명). 라벨 지표는 그날 80단위 표본이라 하루 값의 흔들림이 크다(아래 보조 표의 재표본 비율).")
    a("- 10/17은 토요일이고 글이 16시간치뿐이다. 7월 바닥과 급락 4일은 모두 평일이다.")
    cx = {c["date"]: c for c in r["context"]}
    a(f"- 10월 바탕 수준이 7월과 다르다. 직전 20일 평균 하루 글 수가 7/30 무렵 {cx['2026-07-30']['d_no_base20_mean']:,.0f}건, "
      f"10/15 무렵 {cx['2026-10-15']['d_no_base20_mean']:,.0f}건이다. 인버스·숏 언급은 9월 말 평상 {fmt('LS_A', G[4]['ind']['LS_A']['value'])}로 "
      f"7월 바닥 직전 주({fmt('LS_A', G[1]['ind']['LS_A']['value'])})의 4분의 1 안팎이다. 같은 값이라도 뜻이 다를 수 있다.")
    a("")
    a("**보조 판 (후보 정의는 그대로)**")
    a("")
    a("| 지표 | 장중만(08:00~15:30) 7월 바닥이 범위 밖? | 장중만 10/15 | 장중만 10/16 | 직전 20일 대비 배수판 7월 바닥이 범위 밖? | 배수 10/15 | 배수 10/16 | 재표본에서 7월 바닥 쪽으로 남는 비율 10/15 · 10/16 |")
    a("|---|---|---|---|---|---|---|---|")
    for o in O:
        k = o["key"]
        s = sep[k]
        it = s.get("intraday") or {}
        br = s.get("base20_ratio") or {}
        c15, c16 = o["cells"]["2026-10-15"], o["cells"]["2026-10-16"]

        def iv(c):
            return f"{fmt(k, c.get('intraday_value'))} {disp(c.get('intraday_where', '—'))}" if c.get("intraday_value") is not None else "—"

        def bv(c):
            return f"×{c['base20_ratio']:.1f} {disp(c.get('base20_where', ''))}" if c.get("base20_ratio") is not None else "—"
        bs = (f"{c15['boot_share_bottom_side'] * 100:.0f}% · {c16['boot_share_bottom_side'] * 100:.0f}%"
              if "boot_share_bottom_side" in c15 else "—(전수)")
        a(f"| {o['name']} | {yn(it.get('outside_same_side'))} | {iv(c15)} | {iv(c16)} | "
          f"{yn(br.get('outside_same_side')) if br else '—'} | {bv(c15) if br else '—'} | {bv(c16) if br else '—'} | {bs} |")
    a("")
    a("- 장중판: 하루 창에는 장 마감 뒤 밤이 들어간다. 7/30 창에는 야간 선물 급등을 전하는 글(23시대)과 '오늘 상한가 분위기' 글(7/31 06시대)이 들어 있다. "
      "7/31 KOSPI는 +17.9%였다. 마감 때 알 수 있던 것만 보려고 08:00~15:30 단위로 다시 쟀다.")
    o1316 = bmc["oct_13_16"].values()
    a(f"- 배수판: 값 ÷ 직전 20일 평균. 10월은 바탕이 거의 0이라 반대매매 언급 배수가 10/13~10/16에 ×{min(o1316):.1f}~×{max(o1316):.1f}로, "
      f"6~9월 어느 날(최대 {md_date(bmc['max_jun_sep_date'])} ×{bmc['max_jun_sep']:.1f})보다 크다. 7월 바닥은 이 판에서 급락일과 갈리지 않았다.")
    a("- '기준 없음' = 그 판에서는 7월 바닥도 급락 4일 범위 안이라 10월 날을 견줄 기준이 없다.")
    a("")

    # ---- 1. 자료
    a("## 1. 자료와 한계")
    a("")
    a("| 날짜 | KOSPI 종가 | 그날 등락 | 기준 고점 대비 | 요일 | 수집 | 하루 글 수 | 직전 20일 평균 글 수 |")
    a("|---|---|---|---|---|---|---|---|")
    for c in r["context"]:
        cl = f"{c['kospi_close']:,.0f}" if c["kospi_close"] else "휴장"
        rt = f"{c['kospi_ret'] * 100:+.1f}%" if c["kospi_ret"] is not None else "—"
        dd = f"{c['kospi_dd_ref'] * 100:.1f}% ({md_date(c['ref_peak'])} 고점)" if c["kospi_dd_ref"] is not None else "—"
        b20 = f"{c['d_no_base20_mean']:,.0f}" if c["d_no_base20_mean"] else "—"
        a(f"| {md_date(c['date'])} | {cl} | {rt} | {dd} | {WD[c['weekday']]} | {c['collection']}{'' if c['complete'] else '(16시간)'} | "
          f"{c['d_no']:,.0f} | {b20} |")
    a("")
    a("- **사례 수.** 바닥 1개(2일), 바닥 아닌 급락 4일(6/23·7/2·7/8·7/13, 고정 목록), 10월 저점 1일. 통계 검정이 아니라 사례 묘사다.")
    a("- **깊이가 다르다.** 7월 바닥은 고점 대비 −38.6%, 5주 넘게 이어진 하락의 끝이었다. 급락 4일은 −10~−25%, 10/15는 −21.6%다. "
      "하락이 깊고 길수록 커지는 지표(반대매매·인버스 언급)에서 10/15가 '급락일 쪽'으로 나오는 것은 깊이 차이일 수 있다.")
    a("- **7/29도 규칙상 '바닥 아닌 급락'이다.** −6.0%였고 다음 날 더 낮게 마감했다. 바닥 묶음은 지시대로 7/29·7/30 두 날로 두었다.")
    a("- **수집 방식.** 10/13부터 실시간, 그 전은 소급이다. 결번율·글당 댓글·새 작성자 몫은 방식 차이가 섞여 7월과 10월을 이 값으로 비교하기 어렵다(1단계 메타).")
    p, n = t117["prim"]
    dpp, dn = t117["dir"]
    kap = ", ".join(f"{k} κ {v[2]:.2f}" for k, v in t117["per_emotion"].items() if isinstance(v[2], float) and v[2] == v[2] and v[3] == "미달")
    a(f"- **라벨 신뢰도.** 같은 지침(v2)의 검수 시험 117건에서 주된 감정 일치는 {p}/{n} = {p / n * 100:.0f}%로 실패했다. "
      f"감정별로는 {kap}(모두 0.6 미만)이고, 나머지 감정은 표본이 10건이 안 돼 보류다. 그래서 감정·세기 지표는 설명용이다. "
      f"예측 방향 일치는 {dpp}/{dn} = {dpp / dn * 100:.0f}%로 통과했다.")
    li = r["label_info"]
    a(f"- **라벨 표본.** 51일 × 글 40·댓글 40(10/03 글은 36개), {li['merged']:,}단위, 미라벨 {li['unlabeled']}건. 작성자당 1회로 셌다(감정은 그날 그 작성자의 최대 세기, 예측은 (날짜, 작성자, 값) 쌍 1회). "
      "글과 댓글을 같은 수로 뽑은 설계라 게시판 전체의 글·댓글 비율(약 1:5)과 다르다. 날짜끼리 비교하는 데에는 같은 설계라 문제가 없다.")
    a("- **6/23 라벨**은 라벨 AI의 지식 마감 달(6월)이다. 후보마다 6/23을 빼고도 범위 밖인지 따로 봤다.")
    cs = bc["clues"]
    n_season = cs["여름(장마감·중복 빼고)"]["units"] + cs["가을·연휴"]["units"]
    a(f"- **가림.** 표본은 날짜 × 글/댓글 층에서 무작위로 뽑았고 가격을 읽지 않았다. 라벨러에게 준 조각은 id·본문 두 칸뿐이고, "
      f"조각마다 {bc['days_per_shard'][0]}~{bc['days_per_shard'][1]}일이 섞였다(조각 순서와 날짜 순위 상관 {bc['order_date_rank_corr']:+.2f}). "
      f"가림 뒤 남은 계절 단서(여름·휴가, 가을·연휴)는 {n_season}단위({n_season / bc['n_units'] * 100:.1f}%), 요일 단어는 "
      f"{cs['요일']['units']}단위({cs['요일']['units'] / bc['n_units'] * 100:.1f}%)다. 미리보기 표본과 겹친 {bc['preview_overlap_units']}단위에서 "
      f"예측 값이 {bc['preview_overlap_pred_same']}단위 같았다(따로 한 실행). 라벨러에게 준 지시문은 이 폴더에 파일로 남지 않아, "
      "연구 목적을 알렸는지는 확인할 수 없다.")
    a("")

    # ---- 2. 묶음 표
    a("## 2. 묶음별 지표")
    a("")
    a("값은 묶음 안 날짜 값의 평균(하루 1표)이다. 일수·단위 수는 표 아래에 있다. 주말·휴장일도 범위에 있으면 넣었다: "
      + "; ".join(f"{g['name'].split('(')[0]} {', '.join(md_date(x) for x in g['non_trading_days'])}" for g in G if g["non_trading_days"]) + ".")
    a("")
    a("| 지표 | " + " | ".join(g["name"] for g in G) + " |")
    a("|---|" + "---|" * len(G))
    fam_prev = None
    for k in G[0]["ind"]:
        fam = sep[k]["family"]
        if fam != fam_prev:
            a(f"| **{fam}** |" + " |" * len(G))
            fam_prev = fam
        mark = " ◆" if sep[k]["status"] == "후보" else ""
        a(f"| {names[k]}{mark} | " + " | ".join(fmt(k, g["ind"][k]["value"]) for g in G) + " |")
    a("")
    a("◆ = 3절의 후보(7월 바닥 값이 급락 4일 범위 밖).")
    a("")
    a("**일수 / 단위 수** (같은 분모를 쓰는 지표는 한 줄로 묶었다)")
    a("")
    pat: dict[tuple, list[str]] = {}
    for k in G[0]["ind"]:
        key = tuple((g["ind"][k]["n_days"], g["ind"][k]["n_units"]) for g in G)
        pat.setdefault(key, []).append(k)
    a("| 지표 | 단위 | " + " | ".join(g["name"].split("(")[0] for g in G) + " |")
    a("|---|---|" + "---|" * len(G))
    unit_of = {"d_no_z20": "글(번호)", "comments_per_post": "글", "missing_rate": "글(번호)", "new30_share": "작성자",
               "TILT": "미장·국장 단어 글", "P_bar": "예측 쌍", "P_strong1": "예측 쌍", "P_share": "쌍"}
    for key, ks in pat.items():
        k0 = ks[0]
        un = unit_of.get(k0, "작성자" if k0.endswith("_A") else "라벨 단위")
        nm = ", ".join(names[k] for k in ks) if len(ks) <= 3 else f"{names[ks[0]]} 외 {len(ks) - 1}개"
        a(f"| {nm} | {un} | " + " | ".join(f"{d}일/{u:,}" for d, u in key) + " |")
    a("")
    a("읽을 것(설명용):")
    gb, gpre, gpost, gdrop = G[0]["ind"], G[1]["ind"], G[2]["ind"], G[3]["ind"]
    a(f"- 7월 바닥 이틀은 앞뒤 주와 달랐다. 반대매매 언급 {fmt('MC_A', gb['MC_A']['value'])}(직전 {fmt('MC_A', gpre['MC_A']['value'])}, 직후 {fmt('MC_A', gpost['MC_A']['value'])}), "
      f"글 수 z {fmt('d_no_z20', gb['d_no_z20']['value'])}(직전 {fmt('d_no_z20', gpre['d_no_z20']['value'])}, 직후 {fmt('d_no_z20', gpost['d_no_z20']['value'])}). "
      f"다만 글 수 z는 급락 4일({fmt('d_no_z20', gdrop['d_no_z20']['value'])})과 비슷하다. 크게 빠진 날이면 늘 글이 몰렸다.")
    go = G[6]["ind"]
    a(f"- 10/15 라벨은 7월 바닥보다 무겁다. 불안 {fmt('share_FEAR', go['share_FEAR']['value'])}(7월 바닥 {fmt('share_FEAR', gb['share_FEAR']['value'])}), "
      f"우울 {fmt('share_DESP', go['share_DESP']['value'])}({fmt('share_DESP', gb['share_DESP']['value'])}), 후회 {fmt('share_REGRET', go['share_REGRET']['value'])}({fmt('share_REGRET', gb['share_REGRET']['value'])}). "
      "하루 80단위 표본이고 감정 칸은 설명용이다.")
    a("")

    # ---- 3. 후보
    a("## 3. 7월 바닥에서만 달랐던 지표 찾기 (설명용, 사례 1개라 판정 아님)")
    a("")
    a("- 규칙: 바닥 값 = 7/29·7/30 두 날 값의 평균. 급락 4일 값의 [최솟값, 최댓값] 밖이면 후보. 이 정의는 결과를 보기 전에 정했고 바꾸지 않았다.")
    a(f"- **다중 비교.** 본 지표는 {r['n_checked']}개다(활동 4, 사전 묶음 8, 라벨 감정 몫 11, 라벨 감정 세기 9, 라벨 예측 3). "
      f"값들이 서로 바꿔도 같은 분포라면 두 날 평균이 급락 4일 범위 밖에 나올 확률은 모의실험으로 {ch4['p_mean_outside_sim'] * 100:.0f}%"
      f"(급락일 값이 3일뿐이면 {ch3['p_mean_outside_sim'] * 100:.0f}%)다. 두 날 모두 같은 쪽 밖일 확률은 정확히 2/15 = {ch4['p_both_same_side_exact'] * 100:.0f}%"
      f"(3일이면 {ch3['p_both_same_side_exact'] * 100:.0f}%)다. 그래서 우연만으로 기대되는 개수는 {r['expected_outside_by_chance_mean_rule']:.1f}개, "
      f"{r['expected_both_days_by_chance']:.1f}개다. 실제는 {r['n_candidates']}개, {r['n_both_days_outside']}개였다. "
      "지표끼리 얽혀 있어(같은 감정의 몫과 세기 등) 우연의 폭은 이보다 넓다. 하나쯤이 아니라 열 개 안팎이 우연히 나올 수 있는 설계다. "
      f"얽힘을 그대로 둔 판: 6일 가운데 2일을 '바닥'으로 바꿔 고른 {ex['n_assign']}가지에서 범위 밖 개수는 가운데값 "
      f"{sorted(e['n_out'] for e in ex['assignments'])[len(ex['assignments']) // 2]}개, 최대 {mx['n_out']}개"
      f"({md_date(mx['pair'][0])}·{md_date(mx['pair'][1])})이고, 실제 배정({ex['observed']['n_out']}개)보다 많거나 같은 배정이 "
      f"{ex['n_assign_out_ge_obs']}가지다.")
    a("")
    a("| 후보 | 쪽 | 7/29 · 7/30 | 급락 4일 범위 | 두 날 모두 밖 | 확장 급락일에서도 밖 | 6/23 빼도 밖 | 장중만 봐도 밖 | 재표본 유지 | 바닥 평균보다 더 극단인 다른 날 |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for s in cands:
        k = s["key"]
        b = s["bottom_values"]
        more = s["n_days_more_extreme"]
        md_ = ", ".join(md_date(x) for x in s["more_extreme_days"][:4]) + (" 등" if more > 4 else "")
        boot = f"{s['boot_keep_outside'] * 100:.0f}%" if s.get("boot_keep_outside") is not None else "—(전수)"
        a(f"| {s['name']} | {s['side']} | {fmt(k, b['2026-07-29'])} · {fmt(k, b['2026-07-30'])} | {fmt(k, s['drop_min'])}~{fmt(k, s['drop_max'])} | "
          f"{yn(s['both_days_outside_same_side'])} | {yn(s['ext_outside_same_side'])} | {yn(s['outside_without_0623'])} | "
          f"{yn((s.get('intraday') or {}).get('outside_same_side'))} | {boot} | {more}/{s['n_days_all'] - 2}일" + (f" ({md_})" if more else "") + " |")
    a("")
    a(f"- 확장 급락일 = −5% 이하이고 뒤에 더 낮은 종가가 온 날을 규칙대로 모두 모은 것에서 7/29(바닥 묶음)와 10/13(10월 대상)을 뺀 "
      f"{len(r['ext_rule_days_used'])}일({', '.join(md_date(x) for x in r['ext_rule_days_used'])}). 라벨 지표는 이 가운데 라벨이 있는 날만 쓴다.")
    a("- 재표본 유지 = 날짜 안에서 작성자를 2,000번 다시 뽑았을 때 바닥 평균이 급락 4일 범위 밖(같은 쪽)으로 남은 비율. 표본 흔들림만 보는 값이고 다중 비교는 고치지 않는다.")
    a("- '더 극단인 다른 날'은 바닥 이틀을 뺀 모든 날(라벨은 라벨 49일) 가운데 바닥 평균보다 더 바닥 쪽인 날 수다. 이 수가 크면 바닥만의 모습이 아니라 평소에도 흔한 값이다.")
    a("")
    a("읽을 것:")
    a(f"- **반대매매·마진콜 언급, 인버스·숏 언급**은 모든 보조 점검을 버텼고, 6/01~10/17 전체에서 바닥 평균보다 높은 날이 "
      f"{sep['MC_A']['n_days_more_extreme']}일({', '.join(md_date(x) for x in sep['MC_A']['more_extreme_days'])}), "
      f"{sep['LS_A']['n_days_more_extreme']}일({', '.join(md_date(x) for x in sep['LS_A']['more_extreme_days'])})뿐이다. "
      "하지만 직전 20일 평균 대비 배수로 보면 7/2·7/8의 급락일 배수가 바닥보다 크다. 7월 하락 구간에서 낙폭과 함께 늘어난 값이라 '바닥이라서'보다 '깊이 빠져서'일 수 있다. "
      "7/30 댓글 많은 글에 '(속보) 한국 마진계좌 36만개 청산'이 있다.")
    a("- **세력·외인 분노 단어**는 장중만 보면 범위 안이고, 더 높은 날이 6일 있다.")
    a("- **들뜸**은 바닥 평균보다 높은 날이 라벨 49일 중 12일이다. 급락일에 들뜸이 사라진 것이지 바닥에 특별히 많았던 것이 아니다. 장중만 보면 범위 안이다.")
    import pandas as pd
    dl = pd.read_parquet(OUT / "daily_labels.parquet").set_index("date")
    def nj(d):
        x = dl.loc[pd.Timestamp(d)]
        return int(round(x["share_JOY"] * x["n_authors"]))
    jb = "·".join(str(nj(d)) for d in ("2026-07-29", "2026-07-30"))
    jd = "·".join(str(nj(d)) for d in ("2026-06-23", "2026-07-02", "2026-07-08", "2026-07-13"))
    ak = t117["per_emotion"]["ANGER"][2]
    a(f"- **기쁨 몫·분노 세기**는 보조 점검을 버텼지만 감정 칸은 검수 실패다(분노 κ {ak:.2f}, 기쁨은 표본 부족으로 보류). "
      f"기쁨이 있는 작성자는 바닥 이틀 {jb}명, 급락 4일 {jd}명이다.")
    a("- **중립이 적음·우울 세기**는 두 날이 서로 달랐다(우울 세기 7/29 "
      f"{fmt('mean_DESP', sep['mean_DESP']['bottom_values']['2026-07-29'])}, 7/30 {fmt('mean_DESP', sep['mean_DESP']['bottom_values']['2026-07-30'])}).")
    noc = [s for s in S if s["status"] == "범위 안"]
    a(f"- 범위 안 {len(noc)}개: " + ", ".join(s["name"] for s in noc) + ". 활동(글 수 z·댓글·결번·새 작성자)과 예측 칸은 모두 여기다.")
    a(f"- **다른 날도 7월 바닥처럼 보였다.** 후보 {pdays['n_cands']}개 가운데 7월 바닥 쪽 개수는 바닥 이틀을 뺀 라벨 {pdays['n_other_days']}일에서 "
      f"가운데값 {pdays['other_median']:.0f}개, 5개 이상 {pdays['other_ge5_n']}일, 7개 이상 {len(ge7)}일이다. "
      f"{md_date(top_o['date'])}({top_o['n_same_side']}개)는 고점 대비 {top_o['kospi_dd_ref'] * 100:.1f}%였고 다음 15거래일 안 최저 종가가 "
      f"{top_o['min_ret_15td'] * 100:+.1f}%였다. 7/14·7/30 하루 창은 모두 다음 날 큰 반등(7/15 +6.2%, 7/31 +17.9%) 앞의 밤을 품는다.")
    a("")

    # ---- 4. 대표 글
    a("## 4. 대표 글 (마감 전 댓글이 가장 많은 글 5개, 작성자 정보 없음)")
    a("")
    a("제목은 40자 안으로 줄였다. [정치]·[뉴스]는 1단계 판별기 표시다. 감정 조합은 그날 라벨 80단위의 조합(단위 그대로)이다. "
      "제목은 게시판 글을 그대로 옮긴 군중의 모습이고, '사야 하는 이유' 같은 제목도 이 문서의 의견이나 권유가 아니다.")
    a("")
    for p in r["rep_posts"]:
        cb = " · ".join(f"{c['combo']} {c['n']}건({c['share'] * 100:.0f}%)" for c in p["combos_top3"])
        te = p["top_emotion_combo"]
        a(f"**{md_date(p['date'])}** (그날 저장 글 {p['n_posts_day']:,}건) — 흔한 조합: {cb}. 감정이 있는 조합 가운데 1위: {te['combo']} {te['n']}건.")
        a("")
        a("| 순위 | 제목 | 마감 전 댓글 | 쓴 시각 |")
        a("|---|---|---|---|")
        for i, tt in enumerate(p["top5"], 1):
            tag = {"politics": " [정치]", "news": " [뉴스]"}.get(tt["excl"], "")
            a(f"| {i} | {tt['title'].replace('|', '｜')}{tag} | {tt['n_comments_preclose']} | {tt['hour_kst']}시 |")
        a("")
    a("- 7/29·7/30 상위 10개 가운데 5개가 정부·대통령 탓 공방이고, 10/15에도 1개 있다. 7/30에는 반대매매 청산 소식, 10/16에는 '정부 긴급 대응 발표'와 반등 종목 글이 올라왔다.")
    a("")

    # ---- 5. 파일
    a("## 5. 파일")
    a("")
    a("`C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/ant_index/bottom_case/` 안:")
    a("")
    a("- `results/daily_labels.parquet`: 라벨 51일 날짜 표(감정 몫·평균 세기·세기 ≥ 1 평균, 중립·섞임, P̄·예측 몫·강한 하락, 단위 그대로판 `_u`, 흔한 조합)")
    a("- `results/daily_topic.parquet`: SEMI·MEM 작성자 몫, TILT(139일)")
    a("- `results/compare_tables.json`: 묶음 표, 후보 점검 전부(범위 안 지표 포함), 10월 대조, 장중·배수판, 대표 글, 가격 맥락")
    a("- `results/chart.json`: 날짜별 지표 8개와 KOSPI 종가(라벨 지표는 51일만 값)")
    a("- `results/compare_meta.json`: 입력·출력·코드 SHA-256")
    a("- `results/audit.json`: 점검 결과(묶음 표 280칸·후보·10월 표를 따로 짠 계산으로 대조, 6일 교환 15가지, 가림 단서, 문장 검색). `code/audit.py`가 만든다(이 문서 다음에 실행)")
    a("- `code/compare.py`, `code/compare_report.py`, `code/audit.py`")
    (OUT / "compare.md").write_text("\n".join(L) + "\n", "utf-8")
    meta_p = OUT / "compare_meta.json"
    meta = json.loads(meta_p.read_text("utf-8"))
    meta["report_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta["outputs_sha256"]["compare.md"] = hashlib.sha256((OUT / "compare.md").read_bytes()).hexdigest()
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    print(f"compare.md {len(L)} lines, survived {len(surv)}: {[s['key'] for s in surv]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
