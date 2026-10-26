"""feat_text.parquet·feat_text_meta.json → results/feat_text.md (열 사전·결측률·진단). 숫자는 모두 파일에서 다시 센다.

실행: py feat_text_report.py   (feat_text.py 다음에)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"

# (열, 가설, 정의, 쓰는 v0.2 묶음) — 정의는 HYPOTHESES.md §2·§3·부록 1 그대로, 해석은 notes_feat_text.md
DICT = [
    ("S_P / S_C", "S 재료", "E_A − F_A를 글(첫 저장 줄 제목+본문, 뉴스형·정치 단위 제외)과 댓글(창 안 = 마감 전, 지운 댓글·정치 단위 제외) 따로", "E·F"),
    ("S", "H01·H14·H15·H49·H51·H53·부록 1", "½(E_A − F_A)글 + ½(E_A − F_A)댓글", "E·F"),
    ("S_z60 / Pess", "H15", "S의 z60 / Pess = −S_z60", "E·F"),
    ("S_tok_P·S_tok_C·S_tok / Pess_tok", "H15 보조", "어절 정규화판: (E 적중 − F 적중) ÷ 어절 수, 단위당 같은 단어 3회 상한 / −z60", "E·F"),
    ("E_A_P_z60·E_A_C_z60 / H19_E", "H19", "E_A 글·댓글 각각 z60의 평균(부정·질문형 적중은 매처가 뺀다)", "E"),
    ("FO_mkt_A_P_z60·FO_mkt_A_C_z60 / H21_FO", "H21", "FO 적중이 시장어(lexicon §2 MARKET)와 같은 단위에 나온 판의 작성자 비율, 글·댓글 z60 평균", "FO"),
    ("H21_FO_plain", "H21 보조", "시장어 조건 없는 FO_A의 글·댓글 z60 평균", "FO"),
    ("F/CAP/HES/RES/ANG_A_all_z60", "H24~H26 보조", "글+댓글 작성자 비율의 z60", "F·CAP·HES·RES·ANG"),
    ("ED_P_raw·ED_C_raw / _word / _mark", "H05 원값", "뉴스형·PX 단위(PX 적중 또는 가격 서술 문장)를 뺀 단위 중 E·F·FO·ANG 단어나 강한 표지(ㅋ4+·ㅠ2+·ㄷㄷ류·! 2개+)가 든 비율 / 단어만 / 표지만", "E·F·FO·ANG"),
    ("ED_P·ED_C / H05_ED", "H05", "시간대(장중 09:00~15:30·저녁 15:30~22:00·심야 22:00~09:00)별 비율의 z60을 단위 수로 가중 평균 / 글·댓글 평균", "E·F·FO·ANG"),
    ("n_news_price·event·other", "H16", "뉴스형 글을 첫 저장 줄의 제목 + 본문 첫 줄로 분류한 글 수(가격 보도 → 사건 뉴스 → 기타 순)", "—"),
    ("S_news_price·event·other", "H16", "그 분류 뉴스형 글 작성자의 E_A − F_A(뉴스 거르기만 끄고 같은 매처)", "E·F"),
    ("S_opinion / Pess_opinion", "H16", "의견형 = 뉴스 아닌 글의 S_P / −z60", "E·F"),
    ("Pess_news_event / Pess_news_price", "H16", "−z60(S_news_event) / −z60(S_news_price). M0 잔차화는 가격 단계 몫", "E·F"),
    ("n_bot_authors", "H06 등", "봇·연재: 마감 전 직전 30일 안 같은 제목 패턴(한글·영문만 남김)을 15일 이상 쓴 작성자 수", "—"),
    ("n_scored_posts / n_votes", "H06", "점수 낸 글 수(뉴스형 제외) / 봇·연재를 뺀 작성자 점수 ≠ 0 작성자 수(표)", "BULL·BEAR"),
    ("B", "H06·H09·H11", "(강세 작성자 − 약세 작성자) ÷ 합. 작성자 점수 = 창 안 그 작성자 글 점수 평균, 0인 작성자 제외", "BULL·BEAR"),
    ("D_SD_raw / D_SD_exp / D_SD", "H06·H09·H11", "작성자 점수 SD / 직전 60창(최소 40창) 작성자 점수 풀에서 같은 n을 1,000번 뽑은 SD 평균 / 비. 표 20개 미만이면 D_SD 결측", "BULL·BEAR"),
    ("D_AF", "H06 보조", "√(1 − B²)", "BULL·BEAR"),
    ("semi_*", "H10", "SEMI 태그 글만 쓴 같은 셈(미국 창이 주, 작성자 20명 미만이면 결측)", "BULL·BEAR·SEMI"),
    ("B_z60 / D_SD_z60", "H11", "z60", "BULL·BEAR"),
    ("dD3 / dD3_z60 / H11_X", "H11", "ΔD3 = D_SD(D) − 평균(D_SD(D−5..D−3)) / z60 / X = −z(ΔD3) × z(B)", "BULL·BEAR"),
    ("n_ranked_posts / n_ranked_gap", "H12·H55", "순위 글 수(마감 120분 전까지 쓴 글, 관심 기준 있음. 창 열림일 직전 20일이 다 자료 안인 창만) / 그중 뉴스형·PX 글을 뺀 수", "—"),
    ("Gap_raw / H12_Gap", "H12", "관심 상위 10% 글 점수 부호 평균 − 창 작성자 점수 부호 평균 / z60. 관심 = 등록 뒤 120분 안이면서 마감 전 댓글 수 ÷ 같은 등록 시간 칸의 직전 20일 중앙값", "BULL·BEAR"),
    ("K / A7_K", "H55", "관심 상위 10% 글이 차지한 마감 전 댓글 몫 / rz8", "—"),
    ("SH / A6_SH / n_empty_body", "H54", "본문(공백·URL 제거) 20자 이하이면서 제목 25자 이하인 글 비율(본문이 빈 글은 이미지만 있는 글일 수 있어 뺀다) / rz8 / 뺀 글 수", "—"),
    ("RES5·ANG5 / _z60 / H26_PH", "H26", "직전 5창 합 작성자 수 ÷ 5창 합 활동 작성자 / z60 / PH = RES5_z60 − ANG5_z60", "RES·ANG"),
    ("LL_US_A_all / LL_US_noNVDL_A_all", "H31", "LL 미국 하위 묶음 작성자 비율(글+댓글) / NVDL(사후 단어)을 뺀 판", "LL_US"),
    ("LL_US_d20 / LL_US_noNVDL_d20", "H31", "20창 변화 x(D) − x(D−20창). SOX 잔차화는 가격 단계 몫", "LL_US"),
    ("LLLS_ratio", "H31 보조", "(L − I)/(L + I), L = LL_US 작성자 수, I = LS 작성자 수", "LL_US·LS"),
    ("MC_A_all / MC_n_all / MC_A_all_z60 / MC_A_all_pct250", "H33", "반대매매 작성자 비율 / 작성자 수 / z60 / pct250", "MC"),
    ("H33_sig", "H33", "(z60 ≥ 2 또는 pct250 ≥ 90) 이면서 작성자 3명 이상. 둘 다 결측이면 결측", "MC"),
    ("F_A_d9 / n_auth_d9", "H24", "사건 창 [P 09:00, D 08:00)(= P 09:00~D 07:59, 분 포함)의 F_A(글+댓글) / 활동 작성자. 모든 창에 계산", "F"),
    ("F_A_base20 / H24_R", "H24", "직전 20창(W_KR) F_A_all 평균(20창이 다 있어야) / R = F_A_d9 − F_A_base20", "F"),
    ("CAP_n_d9·HES_n_d9 / H25_CR_d9", "H25", "사건 창 CAP·HES 고유 작성자 수 / CR = (CAP+1)/(HES+1)", "CAP·HES"),
    ("px_kospi_rz_P / px_hynix_rz_P", "H24 사건 표지", "직전 거래일 P의 r ÷ σ60(P 직전 60개 일수익 SD). 지난 종가만", "가격(지난)"),
    ("ev_kospi·ev_hynix·ev_any / ev_first·ev_first_kospi", "H24", "r/σ60 ≤ −2 / 같은 날은 하나, 5거래일 안 이어진 사건은 묶은 첫 사건(합집합판·KOSPI만 판)", "가격(지난)"),
    ("S_pct250 / stage10 / stage20", "부록 1 H57a·b", "pct250(S) / ⌈10 × pct⌉ / ⌈20 × pct⌉", "E·F"),
    ("SG / SG_d20 / H59_sent_bear·bull", "부록 1 H57c·H59", "20창 평균 단계 / 20창 전 대비 변화 / 엇갈림의 기분 쪽 조건(SG_d20 ≤ −2 / ≥ +2). 가격 조건은 빼 둠", "E·F"),
    ("week_id / week_S_mean / week_stage10", "부록 1 주 단계", "주 = D 뒤 첫 월요일 마감 / 그 주 창 S 평균(주의 마지막 창 행에만, 첫·끝 주와 무효 창이 든 주는 결측) / 직전 52주 순위 10단계", "E·F"),
    ("PX_unit_share", "§2.3 메아리 계기", "유효 단위 중 PX 적중 또는 가격 서술 문장이 든 단위 비율", "PX"),
]


KR_ONLY = ("px_", "ev_", "F_A_d9", "n_auth_d9", "CAP_n_d9", "HES_n_d9", "H25_CR_d9", "F_A_base20", "H24_R")


MISS_COLS = {
    "S_tok_P·S_tok_C·S_tok / Pess_tok": ["S_tok", "Pess_tok"],
    "E_A_P_z60·E_A_C_z60 / H19_E": ["H19_E"],
    "FO_mkt_A_P_z60·FO_mkt_A_C_z60 / H21_FO": ["H21_FO"],
    "F/CAP/HES/RES/ANG_A_all_z60": ["F_A_all_z60"],
    "ED_P_raw·ED_C_raw / _word / _mark": ["ED_P_raw", "ED_C_raw"],
    "ED_P·ED_C / H05_ED": ["H05_ED"],
    "n_news_price·event·other": ["n_news_event"],
    "S_news_price·event·other": ["S_news_event", "S_news_price"],
    "semi_*": ["semi_B", "semi_D_SD"],
    "RES5·ANG5 / _z60 / H26_PH": ["RES5", "H26_PH"],
    "ev_kospi·ev_hynix·ev_any / ev_first·ev_first_kospi": ["ev_first"],
    "SG / SG_d20 / H59_sent_bear·bull": ["SG", "H59_sent_bear"],
    "week_id / week_S_mean / week_stage10": ["week_S_mean", "week_stage10"],
    "D_SD_raw / D_SD_exp / D_SD": ["D_SD_raw", "D_SD"],
    "dD3 / dD3_z60 / H11_X": ["dD3", "H11_X"],
    "MC_A_all / MC_n_all / MC_A_all_z60 / MC_A_all_pct250": ["MC_A_all", "MC_A_all_z60", "MC_A_all_pct250"],
}


def miss(df: pd.DataFrame, col: str, market: str) -> str:
    if col not in df.columns:
        return "—"
    if market == "us" and col.startswith(KR_ONLY):
        return "해당 없음"
    s = df.loc[df["market"] == market, col]
    return f"{100 * s.isna().mean():.1f}%"


def main() -> int:
    f = pd.read_parquet(RESULTS / "feat_text.parquet")
    meta = json.loads((RESULTS / "feat_text_meta.json").read_text("utf-8"))
    st = meta["lexicon_status"]
    kr, us = f[f["market"] == "kr"], f[f["market"] == "us"]
    meta_cols = ["D", "P", "weekday", "open_kst", "close_kst", "length_h", "kind", "main_kind", "over96",
                 "xholiday_mismatch", "dst_week", "early_close_D", "complete", "outage_flag", "outage_unknown",
                 "spam_wave", "missing_rate", "trunc_share", "c_conf_share", "c_latest", "n_posts", "n_comments",
                 "n_authors", "d_no", "valid"]
    feat_cols = [c for c in f.columns if c not in meta_cols + ["win_id", "market"]]

    L = []
    L.append("# feat_text — 텍스트 특징(감정·어휘) 열 사전과 결측\n")
    L.append("> `code/feat_text.py`가 만들고 `code/feat_text_report.py`가 이 문서를 쓴다. 정의는 `HYPOTHESES.md` §2·§3, "
             "`HYPOTHESES_addendum1.md` 그대로이고, 모호한 곳의 해석은 `interim/notes_feat_text.md`에 있다. "
             "앞날 가격은 계산하지도 읽지도 않았다.\n")
    L.append("## 입력과 지문\n")
    L.append("| 항목 | 값 |\n|---|---|")
    L.append(f"| 행 | 국내 본판 W_KR {len(kr)}창({kr['win_id'].iloc[0]}~{kr['win_id'].iloc[-1]}), "
             f"미국 본판 W_U {len(us)}창({us['win_id'].iloc[0]}~{us['win_id'].iloc[-1]}). 로더의 완전한 창만 |")
    L.append(f"| 열 | {len(f.columns)}개(메타 {len(meta_cols) + 2}, 특징 {len(feat_cols)}) |")
    lx = meta["lexicon"]
    L.append(f"| 사전 | lexicon_v02.json `{lx['lexicon_v02_json_sha256']}`, lexicon_v02_full.json "
             f"`{lx['lexicon_v02_full_sha256']}`, 기반 후보 `{lx['candidate_sha256'][:8]}…`. 매처 = "
             f"`lexicon_v02_final.load_v02()`(뺀 단어는 자리를 차지한 채 버림). 단어 집합 불일치 "
             f"{len(lx['word_set_mismatch'])}개 묶음 |")
    L.append(f"| 로더 산출 | units.parquet `{meta['units_sha256'][:16]}…`, windows_kr `{meta['windows_kr_sha256'][:16]}…`, "
             f"windows_us `{meta['windows_us_sha256'][:16]}…` |")
    L.append(f"| 매처에 넣은 단위 | {meta['units_matched']:,}개(국내·미국 본판 창 안). 거른 단위: "
             + ", ".join(f"{k or '없음(셈 대상)'} {v:,}" for k, v in meta["excl"].items()) + " |")
    L.append(f"| 가격(사건 표지만) | KOSPI ^KS11·SK하이닉스 000660.KS 종가만, {meta['prices']['kospi']['first']}~"
             f"{meta['prices']['kospi']['last']}({meta['prices']['kospi']['n']}거래일). 마지막 값은 마지막 창(10/15)의 "
             "P일 종가다. r(P)/σ60만 만들었고 사건 뒤 수익·낙폭·갭은 만들지 않았다 |")
    L.append("| 무효 창 | 장애 창(로더 outage = True)은 메타를 뺀 모든 특징을 결측으로 두었다: "
             + (", ".join(f.loc[~f['valid'].astype(bool), 'win_id']) or "없음")
             + ". 로더 기준선이 짧아 장애 여부를 모르는 창은 값을 두고 `outage_unknown`으로 표시 |")
    L.append("")

    L.append("## 정규화와 최소 창 수(채우지 않음)\n")
    L.append("| 정규화 | 규칙 | 지금 첫 값(국내 / 미국) |\n|---|---|---|")

    def first(col):
        out = []
        for d in (kr, us):
            s = d.set_index("win_id")[col].dropna()
            out.append(f"{s.index[0]}({len(s)}창)" if len(s) else "없음(0창)")
        return " / ".join(out)
    L.append(f"| z60 | 직전 60창(행 위치) 가운데 값이 있는 창이 40개 이상, 자기 제외, ddof 1 | S_z60: {first('S_z60')} |")
    L.append(f"| rz8 | 같은 요일 직전 8창(연휴 뒤 창·무효 창 제외) 중앙값·MAD×1.4826, ±6 자름, 8창이 다 차야 | A6_SH: {first('A6_SH')} |")
    L.append(f"| pct250 | 직전 250창, 최소 120창, 자기 제외 | S_pct250: {first('S_pct250')} |")
    L.append(f"| 주 52주 | 직전 52주 다 차야 | week_stage10: {first('week_stage10')} |")
    L.append("")
    L.append("- 자료가 7/13부터라 pct250(최소 120창)과 그것을 쓰는 단계(stage10·20·SG·엇갈림)·MC pct250·주 단계는 전부 결측이다. "
             "수집기가 거꾸로 받기를 이어 가면 같은 코드가 그대로 채운다(국내 120창 = 수집 시작 뒤 약 6달치).")
    L.append("- z60을 쓰는 특징은 41번째 창부터 값이 있다. 그 위에 z60을 한 번 더 쓰는 dD3_z60·H11_X, 기준 풀이 40창 필요한 "
             "D_SD는 더 늦다.\n")

    L.append("## 열 사전과 결측률(창 기준)\n")
    L.append("측정 상태는 `lexicon_v02_full.json` status다. '측정 실패' 묶음도 계산은 했고 표시만 한다.\n")
    L.append("| 열 | 가설 | 정의 | v0.2 묶음(상태) | 결측 국내 | 결측 미국 |\n|---|---|---|---|---|---|")
    for name, hyp, d, b in DICT:
        real = MISS_COLS.get(name) or [c.strip() for part in name.split("/") for c in part.split("·")
                                         if c.strip() in f.columns]
        bs = []
        for x in b.replace("·", ",").split(","):
            x = x.strip()
            if x in st:
                bs.append(f"{x} {'통과' if st[x] == '통과' else '실패'}")
            elif x:
                bs.append(x)
        mk = "; ".join(f"{c} {miss(f, c, 'kr')}" for c in real) if real else "—"
        mu = "; ".join(f"{c} {miss(f, c, 'us')}" for c in real) if real else "—"
        L.append(f"| `{name}` | {hyp} | {d} | {', '.join(bs)} | {mk} | {mu} |")
    L.append("")
    raw_cols = [c for c in f.columns if c.endswith(("_A_P", "_A_C", "_A_all"))]
    L.append(f"- 작성자 비율 원값 `{{X}}_A_P·_C·_all`({len(raw_cols)}열, X = E·F·PX·CAP·HES·FO·RES·ANG·LL_US·LL_KR·LS·MC·SEMI·"
             f"FO_mkt·LL_US_noNVDL)의 결측: 국내 {100 * kr[raw_cols].isna().mean().max():.1f}%, "
             f"미국 {100 * us[raw_cols].isna().mean().max():.1f}%(무효 창 2개). 분모 `n_auth_P·_C·_all`, 분자 `{{X}}_n_all`.")
    wk_k = kr["week_S_mean"].notna().sum()
    wk_u = us["week_S_mean"].notna().sum()
    L.append(f"- `week_S_mean`은 주의 마지막 창 행에만 둔다. 주 기준으로는 국내 {kr['week_id'].nunique()}주 중 {wk_k}주, "
             f"미국 {us['week_id'].nunique()}주 중 {wk_u}주에 값이 있다(첫 주·끝 주는 덜 차서 결측).\n")

    L.append("## 전체 특징 결측률\n")
    L.append("| 특징 | 국내 | 미국 |\n|---|---|---|")
    for c in feat_cols:
        if c == "week_id":
            continue
        L.append(f"| `{c}` | {miss(f, c, 'kr')} | {miss(f, c, 'us')} |")
    L.append("")

    L.append("## 진단(가격 없이)\n")
    v = kr["n_votes"]
    L.append(f"- 표(봇·연재를 뺀 점수 ≠ 0 작성자) 수, 국내 창: 중앙값 {v.median():.0f}, 최소 {v.min():.0f}, 최대 {v.max():.0f}. "
             f"20개 이상인 창 {int((v >= 20).sum())}/{len(kr)}. 미국 창 20개 이상 {int((us['n_votes'] >= 20).sum())}/{len(us)}. "
             f"그래서 D_SD는 기준 풀(40창)과 표 20개를 함께 만족한 국내 {int(kr['D_SD'].notna().sum())}창·"
             f"미국 {int(us['D_SD'].notna().sum())}창뿐이다.")
    sv = us["semi_n_votes"]
    L.append(f"- H10 반도체 태그판 표(미국 창): 중앙값 {sv.median():.0f}, 20개 이상 {int((sv >= 20).sum())}/{len(us)}창. "
             f"semi_D_SD 값이 있는 창 {int(us['semi_D_SD'].notna().sum())}개.")
    x = kr[kr["n_votes"] >= 20]
    if len(x) >= 3:
        L.append(f"- D_SD와 B의 기계적 상관(H11 가격 전 보고): D_SD는 2창뿐이라 못 잼. 원값 D_SD_raw로 표 20개 이상 국내 "
                 f"{len(x)}창에서 Pearson {x['D_SD_raw'].corr(x['B']):.2f}, Spearman "
                 f"{x['D_SD_raw'].corr(x['B'], method='spearman'):.2f}, |B|와 Pearson "
                 f"{x['D_SD_raw'].corr(x['B'].abs()):.2f}.")
    L.append(f"- 봇·연재 작성자(창마다): 국내 중앙값 {kr['n_bot_authors'].median():.0f}명(최대 {kr['n_bot_authors'].max():.0f}), "
             f"점수 셈에서 뺀 글 중앙값 {kr['n_bot_posts_excl'].median():.0f}개. 자료가 7/13부터라 7/28 무렵 전 창에서는 15일 이력이 안 돼 덜 잡힌다.")
    nc = meta["news_cat"]
    L.append(f"- 뉴스형 글 분류(H16, 본판 창 안): 가격 보도 {nc.get('price', 0):,}, 사건 뉴스 {nc.get('event', 0):,}, "
             f"기타 {nc.get('other', 0):,}. 사건 뉴스가 0개인 창 국내 {100 * (kr['n_news_event'] == 0).mean():.1f}%, "
             f"미국 {100 * (us['n_news_event'] == 0).mean():.1f}%, 5개 미만 국내 {100 * (kr['n_news_event'] < 5).mean():.1f}%"
             " — H16 가격 전 점검의 '사건 뉴스형 결측률 30%' 문턱(1개 이상이면 값으로 볼 때)을 국내는 넘는다.")
    L.append(f"- 관심 순위(H12·H55): 순위 글 수 국내 중앙값 {kr['n_ranked_posts'].median():.0f}, 관심 기준(직전 20일 전부 자료 안)이 "
             f"생긴 첫 창 {kr.set_index('win_id')['K'].dropna().index[0]}.")
    L.append(f"- 짧은 글(H54): 본문이 비어 뺀 글 국내 창 중앙값 {kr['n_empty_body'].median():.0f}개, SH 중앙값 {kr['SH'].median():.3f}.")
    L.append(f"- PX 몫(메아리 계기): 국내 창 중앙값 {kr['PX_unit_share'].median():.4f}, 최대 {kr['PX_unit_share'].max():.4f}.")
    sig_k = kr["H33_sig"].astype("boolean")
    sig_u = us["H33_sig"].astype("boolean")
    L.append(f"- H33 신호 창: 국내 {int(sig_k.fillna(False).sum())}개(판정 가능한 {int(sig_k.notna().sum())}창 중), "
             f"미국 {int(sig_u.fillna(False).sum())}개({int(sig_u.notna().sum())}창 중). 실시간 수집분만 판정하는 전방 트랙이다.")
    ev = meta["events"]
    L.append(f"- H24 사건(r/σ60 ≤ −2, 지난 종가만): {len(ev)}개 창 — "
             + "; ".join(f"{e['win_id']}(P {e['P']}, KOSPI {e['kospi_rz']:.2f}σ, 하닉 {e['hynix_rz']:.2f}σ, "
                         f"R {'결측' if e['H24_R'] is None or (isinstance(e['H24_R'], float) and np.isnan(e['H24_R'])) else round(e['H24_R'], 4)}, "
                         f"CR {e['H25_CR_d9']:.2f})" for e in ev)
             + ". 직전 20창이 다 차지 않아 R이 결측이다. R̂·u는 2025 사건 적합이 필요해 못 만든다.\n")

    L.append("## 계산 못 한 것\n")
    L.append("| 대상 | 이유 |\n|---|---|")
    L.append("| H24 R̂·u | R̂ 계수는 2025 사건으로만 적합해 고정한다(§3 H24). 포럼 자료가 2026-07-13부터라 2025 사건이 없다 |")
    L.append("| stage10·stage20·SG·SG_d20·H59 기분 조건, 주 단계, MC pct250 | pct250 최소 120창·직전 52주가 모자라 전부 결측(값을 채우지 않음) |")
    L.append("| D_SD(대부분 창)·dD3·H11_X·semi_D_SD | 표 20개 미만 창이 절반을 넘고, 기준 풀이 40창부터 생기며, ΔD3에 z60을 한 번 더 써서 지금은 값이 없다 |")
    L.append("| H31 8월 규제 시행일 ±10거래일 제외 표지 | 시행일은 가격 전 점검(§5.1 #11) 외부 자료다. 이 단계는 WebSearch를 쓰지 않는다 |")
    L.append("| H14 z_e, H16·H50 M0 잔차화, H31 SOX 잔차화, H35·H36류 | M0(가격) 회귀가 필요하다. 가격 단계 몫 |")
    L.append("| H49 S^E, H51 S^NM, H53 ΔS | 이번 대상 목록에 없다(같은 매처·S 정의로 창만 바꾸면 된다) |")
    L.append("| 강건판 창(국내 고정 24시간, 미국 09:00 ET·고정 24시간), 본문 수정 표지 제외판 | 이번에는 본판 창만 만들었다 |")
    (RESULTS / "feat_text.md").write_text("\n".join(L) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
