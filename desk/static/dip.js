"use strict";
/* 급락 후 담기 — renders /api/dip (정본 docs/desk_system.md §10). Reuses h()/svg()/$/mount/niceTicks/
   sectionNav/toast from app.js. 관찰용 화면이다: 매수 추천·목표 비중·주문 수량이 없다.
   값은 모두 퍼센트(전일 정규장 종가 대비)로 온다. 상승 = red, 하락 = blue, 부호와 ▲▼를 함께 쓴다.
   3분봉 차트는 손으로 그린 SVG 캔들이고, 차트의 모든 기준선·매수 지점은 아래 재생 표에도 있다. */

const DP = { data: null, loading: null, filter: "all", sel: null, detail: null, dLoading: null,
  news: new Map(), ready: false, timer: null };

const dpPct = (v, d = 1) => (v == null ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}%`);
const dpProb = (v) => (v == null ? "—(표본 적음)" : `${v}%`);
const dpSpan = (xs) => {
  const v = xs.filter((x) => x != null);
  if (!v.length) return "—";
  const lo = Math.min(...v), hi = Math.max(...v);
  return lo === hi ? `${lo}%` : `${lo}~${hi}%`;
};
const dpDir = (v) => (v == null || v === 0 ? "" : v > 0 ? "up" : "down");
const dpUsd = (v) => (v == null ? "—" : v >= 1e12 ? `${(v / 1e12).toFixed(1)}조 달러`
  : `${Math.round(v / 1e8).toLocaleString("ko-KR")}억 달러`);
// 뉴욕 "HH:MM" → 한국 "HH:MM" (서머타임 +13, 표준시 +14)
function dpKst(et, shift) {
  if (!et) return "—";
  const [hh, mm] = et.split(":").map(Number);
  const k = (hh + shift) % 24;
  return `${String(k).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
}
const dpTable = (head, rows) => table(head.map((x) => (Array.isArray(x) ? x : [x])), rows);
function dpSegment(label, items, cur, set) {  // 거인의 어깨와 같은 모양(gi-seg)
  return h("div", { class: "gi-ctl" }, h("span", { class: "gi-ctl-label" }, label),
    h("div", { class: "gi-seg", role: "radiogroup", "aria-label": label }, items.map(([k, text]) =>
      h("button", { role: "radio", "aria-checked": String(cur === k), onclick: () => set(k) }, text))));
}
const dpTag = (text, cls = "", tip = null) => h("span", { class: `tag ${cls}`, title: tip }, text);
function dpFlag(v, label, tipYes, tipNo) {
  if (v == null) return h("span", { class: "dp-flag wait", title: `${label}: 아직 그 시각 전입니다` }, `${label} 대기`);
  return h("span", { class: `dp-flag ${v ? "on" : ""}`, title: v ? tipYes : tipNo }, `${label} ${v ? "✓" : "—"}`);
}

/* ------------------------------------------------------------------ load */
async function dpFetch(url) {
  const r = await fetch(url, { cache: "no-store" });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

async function dpLoad(force) {
  if (DP.loading) return DP.loading;
  const root = $("#dip-view");
  root.classList.add("dp-busy");
  if (!DP.data) {
    mount(root, h("section", { class: "card dp-wait" }, h("div", { class: "eyebrow" }, "급락 후 담기"),
      h("p", {}, "시총 10억 달러 이상 미국 종목 약 2,500개의 시세를 훑고, 크게 빠진 종목의 1분봉을 받는 중입니다(약 10초). 2분 동안은 같은 결과를 보여 줍니다.")));
  }
  DP.loading = dpFetch("/api/dip" + (force ? "?refresh=1" : ""))
    .then((d) => {
      DP.data = d;
      dpRender();
      if (force) toast("급락 종목을 다시 찾았습니다");
    })
    .catch((e) => {
      const msg = /not found/i.test(e.message)
        ? "급락 후 담기를 열려면 데스크를 재시작하세요. 실행 창을 닫고 ‘야간데스크.cmd’를 다시 실행하면 됩니다."
        : "급락 종목을 찾지 못했습니다. " + e.message;
      if (DP.data) toast(msg);
      else mount(root, h("section", { class: "card" }, h("p", { class: "err" }, msg)));
    })
    .finally(() => { DP.loading = null; root.classList.remove("dp-busy"); });
  return DP.loading;
}

async function dpSelect(t, day, scroll = true) {
  DP.sel = { t, day: day || null };
  DP.detail = null;
  dpRenderDetail();
  if (scroll) jumpTo("dp-detail");
  const key = `${t}|${day || ""}`;
  DP.dLoading = key;
  try {
    const d = await dpFetch(`/api/dip/detail?t=${encodeURIComponent(t)}${day ? `&day=${day}` : ""}`);
    if (DP.dLoading !== key) return;  // 그사이 다른 종목을 골랐다
    DP.detail = d;
  } catch (e) {
    if (DP.dLoading !== key) return;
    DP.detail = { error: e.message };
  }
  dpRenderDetail();
  dpRenderList();
  if (DP.detail && !DP.detail.error && !DP.news.has(t)) dpLoadNews(t);
}

async function dpLoadNews(t) {
  DP.news.set(t, { loading: true });
  dpRenderNews();
  try {
    DP.news.set(t, await dpFetch(`/api/dip/news?t=${encodeURIComponent(t)}`));
  } catch (e) {
    DP.news.set(t, { error: e.message });
  }
  dpRenderNews();
}

/* ------------------------------------------------------------- structure */
function dpRender() {
  const root = $("#dip-view");
  if (!DP.ready || !$("#dp-list", root)) {
    mount(root,
      h("section", { class: "page-heading dp-head", id: "dp-head" }),
      sectionNav("급락 후 담기 바로가기", [["dp-kpis", "지금 시장"], ["dp-list", "급락 종목"],
        ["dp-detail", "3분봉 다시 보기"], ["dp-evidence", "과거 성적"], ["dp-cases", "사례"]]),
      h("section", { class: "alerts", id: "dp-alerts", "aria-live": "polite" }),
      h("section", { class: "dp-kpis", id: "dp-kpis", "aria-label": "지금 시장" }),
      h("section", { class: "card dp-list", id: "dp-list" }),
      h("section", { class: "card dp-detail", id: "dp-detail" }),
      h("section", { class: "card dp-evidence", id: "dp-evidence" }),
      h("section", { class: "card dp-cases", id: "dp-cases" }),
      h("details", { class: "notes-panel" }, h("summary", {}, "계산 기준·자료 출처·한계"),
        h("footer", { class: "notes", id: "dp-notes" })));
    DP.ready = true;
  }
  dpRenderHead(); dpRenderKpis(); dpRenderList(); dpRenderDetail(); dpRenderEvidence(); dpRenderCases(); dpRenderNotes();
}

function dpRenderHead() {
  const d = DP.data;
  const form = h("form", { class: "dp-replay", onsubmit: (e) => {
    e.preventDefault();
    const t = e.currentTarget.t.value.trim().toUpperCase();
    if (!t) return;
    dpSelect(t, e.currentTarget.day.value || null);
  } },
  h("label", {}, h("span", { class: "gi-ctl-label" }, "종목"), h("input", { name: "t", placeholder: "STX", autocomplete: "off", required: true, maxlength: 10 })),
  h("label", {}, h("span", { class: "gi-ctl-label" }, "날짜(최근 30일, 비우면 최근 거래일)"), h("input", { name: "day", type: "date" })),
  h("button", { class: "btn", type: "submit" }, "3분봉 보기"));
  mount($("#dp-head"),
    h("div", {},
      h("div", { class: "eyebrow" }, `급락 종목 관찰 · 시세 ${d.now_et.slice(5)} 뉴욕 (한국 ${dpKst(d.now_et.slice(11), d.kst_shift)}) · ${d.dst ? "서머타임" : "표준시"} · 갱신 ${d.computed_at.slice(11, 16)}`),
      h("h1", {}, "급락 후 담기"),
      h("p", {}, "오늘 크게 빠진 미국 종목(시총 10억 달러 이상)을 찾고, 프리장·정규장 흐름과 과거 비슷한 경우의 결과를 함께 봅니다. 매수 추천이나 주문 수량은 없습니다.")),
    h("div", { class: "page-meta" },
      h("button", { class: "btn", title: "시세를 새로 받아 급락 종목을 다시 찾습니다(약 10초)", onclick: (e) => {
        e.currentTarget.disabled = true; dpLoad(true).finally(() => { e.currentTarget.disabled = false; });
      } }, "↻ 다시 찾기"),
      form));
}

function dpRenderKpis() {
  const d = DP.data, rows = d.rows, ru = d.rules;
  const deep = rows.filter((r) => r.chg <= -ru.deep).length;
  const f08 = rows.filter((r) => r.flag08).length, f0929 = rows.filter((r) => r.flag0929).length;
  const fHold = rows.filter((r) => r.flag_hold).length;
  const shift = d.kst_shift, open = dpKst("09:30", shift);
  const tile = (label, value, sub, cls = "") => h("div", { class: "gi-kpi" },
    h("div", { class: "gi-kpi-label" }, label), h("div", { class: `gi-kpi-value ${cls}` }, value), h("div", { class: "gi-kpi-sub" }, sub));
  mount($("#dp-kpis"),
    tile("지금 미국장", d.session || "—", `뉴욕 ${d.now_et.slice(11)} · 한국 ${dpKst(d.now_et.slice(11), shift)} · 정규장 ${open} 시작(한국)`),
    tile("시장(SPY)", dpPct(d.spy?.chg, 2), d.spy && d.spy.chg <= ru.market ? `시장 동반 급락(${ru.market}% 이하)` : `전일 종가 대비 · ${ru.market}% 이하면 시장 동반 급락`, dpDir(d.spy?.chg)),
    tile("급락 종목", `${rows.length}개`, `-${ru.drop}% 이하 · -${ru.deep}% 이하 ${deep}개 · ${d.scanned.toLocaleString("ko-KR")}종목 확인`),
    tile("과거 플러스였던 조건", `${f08 + f0929 + fHold}개`, `08:00 반등 중 ${f08} · 09:29 반등 10~20% ${f0929} · 바닥 15분 ${fHold} (관찰 후보)`),
    tile("내가 볼 수 있는 정규장", `${open}~24:00`, d.dst ? `서머타임: 뉴욕 09:30~${ru.window_end}, 정규장 첫 1시간 30분` : "표준시: 정규장 첫 30분만 깨어 있음"));
  mount($("#dp-alerts"), d.warnings?.length ? h("div", { class: "banner" }, "자료 확인 필요: " + d.warnings.join(" · ")) : null);
}

/* ------------------------------------------------------------------ list */
function dpRows() {
  const rows = DP.data.rows;
  if (DP.filter === "noearn") return rows.filter((r) => !r.earnings);
  if (DP.filter === "deep") return rows.filter((r) => r.pm_low != null && r.pm_low <= -DP.data.rules.deep);
  if (DP.filter === "flag") return rows.filter((r) => r.flag08 || r.flag0929 || r.flag_hold);
  return rows;
}

function dpRenderList() {
  if (!DP.data) return;
  const ru = DP.data.rules, rows = dpRows();
  const seg = dpSegment("보기", [
    ["all", `전체 ${DP.data.rows.length}`], ["noearn", `실적 제외 ${DP.data.rows.filter((r) => !r.earnings).length}`],
    ["deep", `프리장 -${ru.deep}% 이하 ${DP.data.rows.filter((r) => r.pm_low != null && r.pm_low <= -ru.deep).length}`],
    ["flag", `조건 충족 ${DP.data.rows.filter((r) => r.flag08 || r.flag0929 || r.flag_hold).length}`],
  ], DP.filter, (k) => { DP.filter = k; dpRenderList(); });
  const head = ["종목", ["시총", 1], ["지금", 1], ["프리장 저가", 1], ["저점 대비 반등", 1], ["시가 · 장중 저가", 1], "뉴스 시점 · 구분", "과거 플러스였던 조건", ""];
  const body = rows.map((r) => {
    const selected = DP.sel && DP.sel.t === r.symbol && !DP.sel.day;
    return h("tr", { class: selected ? "dp-sel" : null, onclick: () => dpSelect(r.symbol, null) },
      h("td", {}, h("b", { class: "mono" }, r.symbol), h("div", { class: "dp-name" }, r.name), r.sector ? h("div", { class: "dp-sub" }, r.sector) : null),
      h("td", { class: "r num" }, dpUsd(r.cap)),
      h("td", { class: "r num" }, h("b", { class: dpDir(r.chg) }, dpPct(r.chg)), h("div", { class: "dp-sub" }, r.session,
        r.post_chg != null ? ` · 장후 ${dpPct(r.post_chg)}` : "")),
      h("td", { class: "r num" }, r.pm_low == null ? h("span", { class: "muted" }, "—") : [dpPct(r.pm_low), h("div", { class: "dp-sub" }, r.pm_low_at || "")]),
      h("td", { class: "r num" }, r.reb == null ? h("span", { class: "muted" }, "—") : h("span", { class: dpDir(r.reb) }, dpPct(r.reb))),
      h("td", { class: "r num" }, r.open == null ? h("span", { class: "muted" }, "개장 전") : [dpPct(r.open),
        h("div", { class: "dp-sub" }, `저가 ${dpPct(r.reg_low)} ${r.reg_low_at || ""}`)]),
      h("td", {}, h("div", { class: "dp-tags" },
        r.ah_ret == null ? null : r.news_ah ? dpTag(`전날 애프터 ${dpPct(r.ah_ret)}`, "sell", "전날 장후에 이미 빠졌습니다(SMCI형). 과거에는 정규장이 프리장 저가를 더 자주 깼습니다.") : dpTag("당일 뉴스", "", "전날 장후에는 크게 빠지지 않았습니다(BULL형)."),
        r.earnings ? dpTag("실적", "semi", "Yahoo 실적 일정이 최근 36시간 안입니다. 실적 급락은 따로 집계했습니다.") : null,
        r.market ? dpTag("시장 동반", "semi", "SPY도 크게 빠졌습니다. 시장 전체 급락일은 개별 악재와 다르게 움직였습니다.") : null)),
      h("td", {}, h("div", { class: "dp-tags" },
        dpFlag(r.flag08, "08:00", "08:00(뉴욕)에 -15% 이하였고 프리장 저점에서 5% 이상 반등 중이었습니다. 과거 +5% 닿음 74~76%, 평균 +1%(t<1).", "08:00 조건이 맞지 않았습니다."),
        dpFlag(r.flag0929, "09:29", "당일 뉴스·실적 외 급락이 09:29에 프리장 저점에서 10~20% 반등했습니다. 2026년 시가 매수 → 첫 1시간 평균 +6%(19건, 대부분 바이오).", "09:29 조건이 맞지 않았습니다."),
        dpFlag(r.flag_hold, "바닥15", `${r.hold?.signal_at || ""} 그날 저가가 15분 동안 안 깨졌고 아직 저가 +2% 안, 전일 대비 −7% 이하였습니다(11:00 전). 과거 +2% 익절 승률 63~66%, 평균 +0.1~0.4%(뒤 기간 t≈0).`, "11:00 전까지 바닥 15분 조건이 맞지 않았습니다.")),
        r.hold && r.flag_hold == null && r.hold.since != null ? h("div", { class: "dp-sub" },
          `저가 뒤 ${r.hold.since}분 · 저가 ${dpPct(r.hold.near)} · 그날 바닥일 확률 ${dpProb(r.hold.survival)}`) : null),
      h("td", {}, h("button", { class: "linkbtn", onclick: (e) => { e.stopPropagation(); dpSelect(r.symbol, null); } }, "3분봉")));
  });
  mount($("#dp-list"),
    h("header", {}, h("h2", {}, "급락 종목"),
      h("span", { class: "sub" }, `기준 종가 대비 -${ru.drop}% 이하 · 하락 큰 순 ${ru.max_detail}개까지 1분봉으로 프리장 흐름 계산 · 행을 누르면 3분봉`)),
    h("div", { class: "gi-filters" }, seg),
    rows.length ? h("div", { class: "scroll dp-scroll" }, dpTable(head, body))
      : h("p", { class: "muted" }, DP.data.rows.length ? "이 보기에 해당하는 종목이 없습니다." : "지금은 크게 빠진 종목이 없습니다."));
}

/* ---------------------------------------------------------------- detail */
function dpRenderDetail() {
  const host = $("#dp-detail");
  if (!host) return;
  if (!DP.sel) {
    mount(host, h("header", {}, h("h2", {}, "3분봉 다시 보기")),
      h("p", { class: "muted" }, "급락 종목을 누르거나, 위에서 종목과 날짜(최근 30일)를 넣으면 그날 3분봉과 시각별 조건, 정규장 매수 규칙의 결과를 보여 줍니다. 예: STX 2026-10-02."));
    return;
  }
  const d = DP.detail;
  if (!d) {
    mount(host, h("header", {}, h("h2", {}, `${DP.sel.t} 3분봉`)), h("p", { class: "muted" }, "1분봉을 받아 3분봉으로 묶는 중입니다…"));
    return;
  }
  if (d.error) {
    mount(host, h("header", {}, h("h2", {}, `${DP.sel.t} 3분봉`)), h("p", { class: "err" }, d.error));
    return;
  }
  const f = d.features || {}, shift = d.kst_shift;
  const facts = [
    ["전일 종가", d.c0 == null ? "—" : `$${d.c0.toFixed(2)}`],
    ["프리장 저가", f.pm_low == null ? "—(−7% 이하 거래 없음)" : `${dpPct(f.pm_low)} · ${f.pm_low_at} 뉴욕`],
    ["09:29", f.pm_last == null ? "—" : `${dpPct(f.pm_last)} · 저점 대비 ${dpPct(f.reb)}`],
    ["시가", f.open == null ? "개장 전" : dpPct(f.open)],
    ["장중 저가", f.reg_low == null ? "—" : `${dpPct(f.reg_low)} · ${f.reg_low_at} 뉴욕`],
    [d.live ? "지금" : "종가", f.reg_last == null ? "—" : dpPct(f.reg_last)],
    ["다음 날 시가", f.next_open == null ? "—" : dpPct(f.next_open)],
    ...(f.hold && f.hold.since != null ? [[d.live ? "지금 저가 뒤" : "마감 때 저가 뒤", `${f.hold.since}분 · 저가 ${dpPct(f.hold.near)} · 그날 바닥일 확률 ${dpProb(f.hold.survival)}`],
      ["바닥 15분 신호", f.hold.signal_at ? `${f.hold.signal_at} 뉴욕 · ${dpKst(f.hold.signal_at, d.kst_shift)} 한국` : f.flag_hold === null ? "지켜보는 중(11:00까지)" : "11:00 전까지 없음"]] : []),
  ];
  const band = f.reb == null ? null : f.reb < 5 ? "5% 미만" : f.reb < 10 ? "5~10%" : f.reb < 20 ? "10~20%" : "20% 이상";
  mount(host,
    h("header", {}, h("h2", {}, `${d.t} ${d.name && d.name !== d.t ? d.name : ""}`),
      h("span", { class: "sub" }, `${d.day} · ${d.minutes}분봉 · 뉴욕 시각(한국 +${shift}시간)${d.live ? ` · 지금 뉴욕 ${d.now_et}` : ""}`),
      h("div", { class: "right" }, h("button", { class: "linkbtn", onclick: () => dpSelect(d.t, DP.sel.day, false) }, "새로 고침"))),
    h("div", { class: "dp-facts" }, facts.map(([k, v]) => h("div", {}, h("span", {}, k), h("b", { class: "num" }, v)))),
    h("div", { class: "legend chart-legend" },
      h("span", {}, h("i", { class: "dp-key up" }), "오른 봉"), h("span", {}, h("i", { class: "dp-key down" }), "내린 봉"),
      h("span", {}, h("i", { class: "dp-key c0" }), "전일 종가"), h("span", {}, h("i", { class: "dp-key pml" }), "프리장 저가"),
      h("span", {}, h("i", { class: "dp-key vwap" }), "VWAP(정규장)"), h("span", {}, h("i", { class: "dp-key ext" }), "장전·장후"),
      h("span", {}, "▲ 정규장 규칙 매수 지점")),
    h("div", { class: "chart dp-chart", id: "dp-chart" }),
    h("div", { class: "dp-two" },
      h("div", {}, h("h3", {}, "판단 점검"),
        h("ul", { class: "dp-checks" },
          h("li", {}, h("b", {}, "뉴스 시점 "), f.ah_ret == null ? "—" : f.news_ah ? `전날 애프터에 이미 ${dpPct(f.ah_ret)} — SMCI형. 과거 정규장이 프리장 저가를 깬 비율 ${DP.data?.research.hold.ah_broke ?? "53~65%"}` : "당일 뉴스(전날 애프터는 평온) — BULL형"),
          h("li", {}, h("b", {}, "실적 "), d.live ? (d.earnings ? "Yahoo 실적 일정이 최근 36시간 안 — 실적 급락으로 따로 봅니다" : "최근 실적 발표 아님") : "지난 날짜는 표시하지 않습니다"),
          h("li", {}, h("b", {}, "개장 전 반등 "), band ? `${band} 구간 → 아래 ‘과거 성적’의 같은 줄을 보세요` : "프리장 -7% 이하 거래가 없어 계산하지 않습니다"),
          h("li", {}, h("b", {}, "08:00 조건 "), f.flag08 == null ? "아직 08:00 전" : f.flag08 ? "충족(−15% 이하·저점 대비 5% 이상 반등)" : "불충족"),
          h("li", {}, h("b", {}, "09:29 조건 "), f.flag0929 == null ? "아직 09:29 전" : f.flag0929 ? "충족(당일 뉴스·실적 외·반등 10~20%)" : "불충족")),
        h("h3", {}, "뉴스로 확인할 네 가지"),
        h("ol", { class: "dp-checks" },
          h("li", {}, "법적 강제력이 있나(기소·처벌 vs 보고서·보도)"),
          h("li", {}, "손실 크기가 숫자로 나왔나(벌금·매출 비중)"),
          h("li", {}, "돈 버는 사업을 직접 건드리나(핵심 공급·주력 제품)"),
          h("li", {}, "처음 듣는 이야기인가(이미 알려진 내용인가)"))),
      h("div", { id: "dp-news" })),
    h("h3", {}, "시각별 조건 재생 (프리장, +5% 지정가 · 못 닿으면 다음 날 시가)"),
    d.hour_replay.length
      ? h("div", { class: "dp-tw" }, dpTable(["매수 시각(뉴욕 · 한국)", ["그때 등락", 1], ["저점 대비", 1], "+5% 닿은 시각", ["못 닿으면 다음 날 시가", 1]],
        d.hour_replay.map((r) => h("tr", {}, h("td", { class: "mono" }, `${r.at} · ${dpKst(r.at, shift)}`),
          h("td", { class: "r num" }, dpPct(r.close_pct)), h("td", { class: "r num" }, dpPct(r.reb)),
          h("td", { class: "mono" }, r.tp5_at ? `${r.tp5_at} · ${dpKst(r.tp5_at, shift)}` : h("span", { class: "muted" }, "못 닿음")),
          h("td", { class: "r num" }, r.tp5_at ? h("span", { class: "muted" }, "—") : h("span", { class: dpDir(r.next_open) }, dpPct(r.next_open)))))))
      : h("p", { class: "muted" }, "프리장에서 −15% 이하(직전 봉 −10% 이하) 조건이 맞은 시각이 없습니다."),
    h("h3", {}, "정규장 규칙 재생 (09:30~11:00 뉴욕에 3분봉 종가로 매수)"),
    d.rules.length
      ? h("div", { class: "dp-tw" }, dpTable(["규칙", "매수 시각(뉴욕 · 한국)", ["시가 대비", 1], "+2% 닿음", "+3% 닿음", "+5% 닿음", "11시 전 손절 신호", ["종가", 1], ["다음 날 시가", 1]],
        d.rules.map((r) => h("tr", { class: r.rule === "hold15" ? "dp-hl" : null }, h("td", {}, r.name),
          r.at ? [h("td", { class: "mono" }, `${r.at} · ${dpKst(r.at, shift)}`), h("td", { class: "r num" }, dpPct(r.entry_vs_open)),
            h("td", { class: "mono" }, r.tp2_at || h("span", { class: "muted" }, "—")),
            h("td", { class: "mono" }, r.tp3_at || h("span", { class: "muted" }, "—")),
            h("td", { class: "mono" }, r.tp5_at || h("span", { class: "muted" }, "—")),
            h("td", { class: "mono" }, r.stop_at || h("span", { class: "muted" }, "없음")),
            h("td", { class: "r num" }, h("span", { class: dpDir(r.close) }, dpPct(r.close))),
            h("td", { class: "r num" }, h("span", { class: dpDir(r.next_open) }, dpPct(r.next_open)))]
            : h("td", { colspan: 8, class: "muted" }, "11:00 전까지 신호가 없었습니다")))))
      : h("p", { class: "muted" }, "정규장 자료가 아직 없습니다."),
    h("p", { class: "dp-foot" }, "규칙: 시가 = 첫 거래 / 10:00 = 10:00에 끝나는 봉 / 저점+2% 반등 = 시가 아래로 내려간 뒤 그때까지 저가의 +2% 이상에서 마감 / VWAP 회복 = VWAP 아래 마감 다음 위로 마감 / 15분 고가 돌파 = 09:45 이후 첫 15분 고가 위로 마감 / 바닥 15분 유지 = 그날 저가가 15분 동안 안 깨졌고 종가가 저가 +2% 안, 전일 대비 −7% 이하(손절은 매수 전 저가 −1%를 봉 저가가 닿은 시각). 익절은 다음 봉부터 그날 마감까지 고가로 판정하고, 손절 신호는 11:00까지 봉 종가가 매수 전 저가(시가 매수는 −5%) 아래인 시각입니다. 표의 시각은 모두 그 일이 일어난 봉이 끝나는 시각입니다. 장전·장후 봉의 긴 꼬리는 한 번 찍고 사라진 오류 체결일 수 있습니다(프리장 저가는 이런 체결을 걸러서 잽니다)."));
  dpDrawChart();
  dpRenderNews();
}

function dpRenderNews() {
  const host = $("#dp-news");
  if (!host || !DP.detail || DP.detail.error) return;
  const t = DP.detail.t, n = DP.news.get(t);
  const head = h("h3", {}, "최근 뉴스 ", h("span", { class: "dp-sub" }, "나스닥 종목 뉴스 · 새 탭에서 열림"));
  if (!n || n.loading) return mount(host, head, h("p", { class: "muted" }, n ? "뉴스를 받는 중…" : ""));
  if (n.error) return mount(host, head, h("p", { class: "err" }, n.error));
  mount(host, head, n.items.length
    ? h("ul", { class: "dp-news" }, n.items.map((x) => h("li", {},
      h("a", { href: x.url, target: "_blank", rel: "noopener noreferrer" }, x.title),
      h("div", { class: "dp-sub" }, [x.publisher, x.when].filter(Boolean).join(" · ")))))
    : h("p", { class: "muted" }, "최근 뉴스가 없습니다."));
}

/* 3분봉 캔들: 전일 종가 대비 % 축, 장전·장후 음영, 전일 종가·프리장 저가·VWAP 선, 정규장 규칙 매수 ▲. */
function dpDrawChart() {
  const host = $("#dp-chart"), d = DP.detail;
  if (!host || !d || d.error) return;
  host.replaceChildren();
  const bars = d.candles, n = bars.length, c0 = d.c0, f = d.features || {};
  if (!n || !c0) { mount(host, h("p", { class: "muted" }, "봉 자료가 없습니다.")); return; }
  const W = Math.max(320, host.clientWidth), H = W < 640 ? 280 : 380;
  const m = { l: 54, r: 14, t: 12, b: 28 };
  const pc = (v) => (v / c0 - 1) * 100;
  let lo = 0, hi = 0;
  for (const b of bars) { lo = Math.min(lo, pc(b.l)); hi = Math.max(hi, pc(b.h)); }
  if (f.pm_low != null) lo = Math.min(lo, f.pm_low);
  const ticks = niceTicks(lo - (hi - lo) * 0.04, hi + (hi - lo) * 0.04, 6);
  const t0 = ticks[0], t1 = ticks[ticks.length - 1];
  const ys = (v) => m.t + (H - m.t - m.b) * (1 - (v - t0) / (t1 - t0));
  const step = (W - m.l - m.r) / n, bw = Math.max(1, Math.min(9, step * 0.68));
  const xs = (i) => m.l + step * (i + 0.5);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img",
    "aria-label": `${d.t} ${d.day} ${d.minutes}분봉. 전일 종가 대비 최저 ${lo.toFixed(1)}%, 최고 ${hi.toFixed(1)}%` }, host);
  // 장전·장후 음영(연속 구간마다 한 칸)
  let run = null;
  bars.forEach((b, i) => {
    const ext = b.ses !== "reg";
    if (ext && run == null) run = i;
    if ((!ext || i === n - 1) && run != null) {
      const endI = ext ? i : i - 1;
      svg("rect", { class: "dp-ext", x: m.l + step * run, y: m.t, width: step * (endI - run + 1), height: H - m.t - m.b }, root);
      run = null;
    }
  });
  const grid = svg("g", { class: "grid" }, root), axis = svg("g", { class: "axis" }, root);
  const text = (p, a, v) => { const t = svg("text", a, p); t.textContent = v; return t; };
  ticks.forEach((v) => {
    svg("line", { x1: m.l, x2: W - m.r, y1: ys(v), y2: ys(v) }, grid);
    text(axis, { x: m.l - 8, y: ys(v) + 4, "text-anchor": "end" }, `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v)}%`);
  });
  const every = W < 640 ? 2 : 1;
  let lastHour = -1;
  bars.forEach((b, i) => {
    const hh = +b.t.slice(0, 2);
    if (b.t.endsWith(":00") && hh !== lastHour && hh % every === 0) {
      text(axis, { x: xs(i), y: H - 8, "text-anchor": "middle" }, `${hh}시`);
      lastHour = hh;
    }
  });
  const hline = (v, cls) => svg("line", { class: cls, x1: m.l, x2: W - m.r, y1: ys(v), y2: ys(v) }, root);
  hline(0, "dp-c0");
  if (f.pm_low != null) hline(f.pm_low, "dp-pml");
  let vp = "", pen = false;
  bars.forEach((b, i) => {
    if (b.vwap == null) { pen = false; return; }
    vp += `${pen ? "L" : "M"}${xs(i).toFixed(1)},${ys(pc(b.vwap)).toFixed(1)}`;
    pen = true;
  });
  if (vp) svg("path", { class: "dp-vwap", d: vp }, root);
  bars.forEach((b, i) => {
    const up = b.c >= b.o, cls = up ? "dp-up" : "dp-down";
    svg("line", { class: cls, x1: xs(i), x2: xs(i), y1: ys(pc(b.h)), y2: ys(pc(b.l)) }, root);
    const y1 = ys(pc(Math.max(b.o, b.c))), y2 = ys(pc(Math.min(b.o, b.c)));
    svg("rect", { class: cls, x: xs(i) - bw / 2, y: y1, width: bw, height: Math.max(1, y2 - y1) }, root);
  });
  // 정규장 규칙 매수 ▲: 매수 시각은 봉이 끝나는 시각이다(시가는 첫 봉)
  const idx = new Map(bars.map((b, i) => [b.t, i]));
  const back = (at) => {
    const [hh, mm] = at.split(":").map(Number), tot = hh * 60 + mm - d.minutes;
    return `${String(Math.floor(tot / 60)).padStart(2, "0")}:${String(tot % 60).padStart(2, "0")}`;
  };
  for (const r of d.rules || []) {
    if (!r.at) continue;
    const i = idx.get(r.rule === "open" ? "09:30" : back(r.at));
    if (i == null) continue;
    const x = xs(i), y = ys(pc(bars[i].l)) + 12, s = 5;
    const tri = svg("polygon", { class: "dp-mark", points: `${x},${y - s} ${x - s},${y + s * 0.8} ${x + s},${y + s * 0.8}` }, root);
    const tt = svg("title", {}, tri); tt.textContent = `${r.name} 매수 ${r.at}`;
  }
  // 마우스·키보드: 봉 정보
  const cross = svg("line", { class: "cross", y1: m.t, y2: H - m.b, style: "opacity:0" }, root);
  const tip = h("div", { class: "tooltip", role: "presentation" });
  host.appendChild(tip);
  host.tabIndex = 0;
  let cur = n - 1;
  const show = (i) => {
    cur = Math.max(0, Math.min(n - 1, i));
    const b = bars[cur], x = xs(cur);
    cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.style.opacity = 1;
    mount(tip, h("div", { class: "d" }, `${b.t} 뉴욕 · ${dpKst(b.t, d.kst_shift)} 한국 · ${b.ses === "pm" ? "장전" : b.ses === "post" ? "장후" : "정규장"}`),
      [["시가", b.o], ["고가", b.h], ["저가", b.l], ["종가", b.c]].map(([k, v]) =>
        h("div", { class: "row" }, h("span", {}, k), h("b", {}, `$${v.toFixed(2)} (${dpPct(pc(v))})`))),
      b.vwap != null ? h("div", { class: "row" }, h("span", {}, "VWAP"), h("b", {}, `$${b.vwap.toFixed(2)}`)) : null);
    tip.style.opacity = 1;
    const tw = tip.offsetWidth;
    tip.style.left = Math.min(W - tw - 4, Math.max(0, x > W * 0.6 ? x - tw - 14 : x + 14)) + "px";
    tip.style.top = m.t + 6 + "px";
  };
  const hide = () => { cross.style.opacity = 0; tip.style.opacity = 0; };
  const overlay = svg("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, root);
  overlay.addEventListener("pointermove", (e) => {
    const r = root.getBoundingClientRect();
    show(Math.floor((((e.clientX - r.left) * W) / r.width - m.l) / step));
  });
  overlay.addEventListener("pointerleave", hide);
  host.onkeydown = (e) => {
    if (e.key === "ArrowLeft") { show(cur - 1); e.preventDefault(); }
    if (e.key === "ArrowRight") { show(cur + 1); e.preventDefault(); }
    if (e.key === "Escape") hide();
  };
  host.onfocus = () => show(cur);
  host.onblur = hide;
}

/* -------------------------------------------------------------- evidence */
function dpRenderEvidence() {
  const R = DP.data.research, shift = DP.data.kst_shift;
  const cell = (v, d = 1) => h("td", { class: `r num ${dpDir(v)}` }, dpPct(v, d));
  const timeRows = R.time.map(([at, h25, a25, n25, h26, a26, n26, miss]) => h("tr", { class: at === "08:00" || at === "09:29" || at === "시가" ? "dp-hl" : null },
    h("td", { class: "mono" }, at === "시가" ? `시가 · ${dpKst("09:30", shift)}` : `${at} · ${dpKst(at, shift)}`),
    h("td", { class: "r num" }, `${h25}%`), cell(a25), h("td", { class: "r num muted" }, n25),
    h("td", { class: "r num" }, `${h26}%`), cell(a26), h("td", { class: "r num muted" }, n26), cell(miss)));
  const t8 = Object.entries(R.time_08).map(([k, [h25, a25, n25, h26, a26, n26]]) => h("tr", {},
    h("td", {}, `08:00 · ${k}`), h("td", { class: "r num" }, `${h25}%`), cell(a25), h("td", { class: "r num muted" }, n25),
    h("td", { class: "r num" }, `${h26}%`), cell(a26), h("td", { class: "r num muted" }, n26), h("td", {})));
  const periods = ["2026상", "2026하", "2025", "2024"];
  const reb = R.rebound.map(([band, by]) => h("tr", { class: band === "10~20%" ? "dp-hl" : null }, h("td", {}, band),
    periods.map((p) => by[p] ? h("td", { class: "r num" }, h("span", { class: dpDir(by[p][0]) }, dpPct(by[p][0])), " / ",
      h("span", { class: dpDir(by[p][1]) }, dpPct(by[p][1])), h("span", { class: "muted" }, ` (${by[p][2]})`)) : h("td", { class: "r muted" }, "2건 이하"))));
  const lim = R.limit.map(([lv, by]) => h("tr", {}, h("td", {}, `${lv.replace("-", "−")} 지정가`),
    periods.map((p) => h("td", { class: "r num" }, h("span", { class: dpDir(by[p][0]) }, dpPct(by[p][0])), h("span", { class: "muted" }, ` (${by[p][1]})`)))));
  const intra = R.intraday.map(([rule, tp, he, ae, ne, hl, al, nl]) => h("tr", {}, h("td", {}, `${rule} · +${tp}%`),
    h("td", { class: "r num" }, `${he}%`), cell(ae), h("td", { class: "r num muted" }, ne),
    h("td", { class: "r num" }, `${hl}%`), cell(al), h("td", { class: "r num muted" }, nl)));
  const lowE = R.intraday_low["앞"], lowL = R.intraday_low["뒤"];
  mount($("#dp-evidence"),
    h("header", {}, h("h2", {}, "과거 성적"), h("span", { class: "sub" }, `연구: ${R.study.period} · ${R.study.universe}`)),
    h("ul", { class: "dp-lede" },
      h("li", {}, h("b", {}, "저점을 맞혀 사는 규칙은 손해였습니다. "), `정해 둔 하락률(−15~−35%)에 지정가로 사면 2024~2026 모두 평균 마이너스였고, 정규장이 프리장 저가를 다시 깬 비율이 ${R.hold.broke}, 5일 안에 더 낮은 저가가 나온 비율이 ${R.hold.new_low_5d}입니다.`),
      h("li", {}, h("b", {}, "+5% 익절 확률이 확실히 높은 시각은 없습니다. "), "어느 시각이든 47~62%이고, 늦게 살수록(개장 직전·시가) 덜 잃었습니다."),
      h("li", {}, h("b", {}, "두 해 모두 플러스였던 건 ‘반등을 확인하고 사기’뿐입니다. "), "08:00에 저점 대비 5% 이상 반등 중(+5% 닿음 74~76%, 평균 +1%), 09:29에 10~20% 반등한 당일 뉴스(2026년 첫 1시간 평균 +6%). 표본이 작고 여러 조합 중 고른 것이라 관찰 후보입니다."),
      h("li", {}, h("b", {}, "마지막 하락 지점은 시간과 반등 폭이 함께 알려 줍니다. "), "저가가 15분 안 깨지면 그날 바닥일 확률 31~35%, 1시간 48~50%, 2시간 59~63%입니다. 같은 시간이어도 저가 +1% 안에 붙어 있으면 낮고(2시간 뒤 27~39%), +2~4% 떠 있으면 높습니다(1시간 뒤 74~82%). 다만 떠 있는 만큼 남은 반등이 작아 +2% 익절·저가 −1% 손절로는 최근 기간 손실이었습니다. 기다리는 비용은 작습니다: 저가가 1~2시간 버틴 시점에도 가격은 저가 대비 중앙 +1.3~1.5%였습니다. 깨져도 대개 1~1.5% 더 빠지는 정도입니다."),
      h("li", {}, h("b", {}, "깨어 있는 시간(11:00 전)의 ‘바닥 15분 유지 + 2% 익절’만 두 기간 모두 플러스였습니다. "), "승률 63~66%, 평균 +0.1~0.4%. 최근 기간은 우연과 구별되지 않을 만큼 얇고(t≈0), 못 닿으면 다음 날 크게 잃는 날이 있습니다(최악 −11~−21%). 11:00까지 저가 −1% 손절을 함께 쓰면 최악이 −7%로 줄고 평균은 +0.1% 안팎입니다."),
      h("li", {}, h("b", {}, "갭 하락일 장중 3분봉 규칙은 본전입니다. "), `그날 저점이 첫 15분에 나온 비율 ${lowL[0]}~${lowE[0]}%, 13시 이후 ${lowE[3]}~${lowL[3]}%. 시가 매수 +3% 익절은 닿음 50~65%에 평균 +0.1~0.7%였습니다.`)),
    h("div", { class: "dp-grid" },
      h("div", {}, h("h3", {}, "프리장 매수 시각별 (실적 외 개별 악재 · 그 시각 −15% 이하에서 사서 당일 +5% 지정가)"),
        h("div", { class: "dp-tw" }, dpTable(["매수 시각(뉴욕 · 한국)", ["2025 닿음", 1], ["평균", 1], ["건", 1], ["2026 닿음", 1], ["평균", 1], ["건", 1], ["못 닿은 건", 1]], [...timeRows, ...t8]))),
      h("div", {}, h("h3", {}, "09:29 저점 대비 반등 폭별 시가 매수 (당일 뉴스·실적 외 · 첫 1시간 / 다음 날 시가)"),
        h("div", { class: "dp-tw" }, dpTable(["반등 폭", ["2026 상반기", 1], ["2026 하반기", 1], ["2025", 1], ["2024", 1]], reb))),
      h("div", {}, h("h3", {}, "정해 둔 하락률 지정가 → 시가 매도 (실적 외 개별 악재)"),
        h("div", { class: "dp-tw" }, dpTable(["매수", ["2026 상반기", 1], ["2026 하반기", 1], ["2025", 1], ["2024", 1]], lim))),
      h("div", {}, h("h3", {}, "그날 바닥일 확률 (그 순간 −7% 이하인 종목 · 저가 뒤 지난 시간 × 지금 저가 대비 · 세 기간 범위)"),
        h("div", { class: "dp-tw" }, dpTable(["저가 뒤 지난 시간", ...R.survival_near.map((lab) => [lab, 1]), ["시간만", 1], ["깨진 날 더 빠진 폭(중앙)", 1], ["하위 10%", 1]],
          R.survival.map(([lab, a, b5, b3, med, p10], i) => h("tr", {}, h("td", {}, lab),
            R.survival_grid[i].map((c) => c ? h("td", { class: "r num", title: `앞 5분봉 ${c[0]}% · 뒤 5분봉 ${c[1]}% · 뒤 3분봉 ${c[2]}%(${c[3]}건)` }, dpSpan(c.slice(0, 3)))
              : h("td", { class: "r muted" }, "표본 적음")),
            h("td", { class: "r num muted" }, dpSpan([a, b5, b3])), h("td", { class: "r num down" }, med), h("td", { class: "r num down" }, p10)))))),
      h("div", {}, h("h3", {}, "바닥 15분 유지 매수 (11:00 전 신호 · +2% 지정가 · 못 닿으면 다음 날 시가)"),
        h("div", { class: "dp-tw" }, dpTable(["손절 · 기간", ["평균", 1], ["승률", 1], ["익절", 1], ["손절", 1], ["최악", 1], ["건", 1]],
          R.bottom.flatMap(([stop, cells]) => cells.map(([win, avg, wr, tp, st, worst, n]) => h("tr", {},
            h("td", {}, `${stop} · ${win}`), cell(avg, 2), h("td", { class: "r num" }, `${wr}%`), h("td", { class: "r num" }, `${tp}%`),
            h("td", { class: "r num" }, `${st}%`), cell(worst), h("td", { class: "r num muted" }, n))))))),
      h("div", {}, h("h3", {}, "갭 하락(시가 −7% 이하)일 정규장 규칙 (앞 8/10~9/5 5분봉 · 뒤 9/8~10/6 3분봉)"),
        h("div", { class: "dp-tw" }, dpTable(["규칙 · 익절", ["앞 닿음", 1], ["평균", 1], ["건", 1], ["뒤 닿음", 1], ["평균", 1], ["건", 1]], intra)))));
}

function dpRenderCases() {
  const R = DP.data.research;
  mount($("#dp-cases"),
    h("header", {}, h("h2", {}, "사례"), h("span", { class: "sub" }, "전일 종가 대비 · 최근 30일 안의 날은 3분봉으로 다시 볼 수 있습니다")),
    h("div", { class: "dp-cases-grid" }, R.cases.map((c) => {
      const recent = (Date.now() - new Date(c.day + "T16:00:00-04:00")) / 864e5 <= 29;
      return h("article", { class: "dp-case" },
        h("header", {}, h("b", { class: "mono" }, c.t), h("span", { class: "dp-sub" }, c.day), recent
          ? h("button", { class: "linkbtn", onclick: () => dpSelect(c.t, c.day) }, "3분봉 다시 보기") : null),
        h("p", { class: "dp-case-news" }, c.news, h("span", { class: "dp-sub" }, ` · ${c.when}`)),
        h("div", { class: "dp-facts small" },
          [["프리장 저가", c.pm_low], ["09:29", c.at_0929], ["저점 대비", c.reb], ["시가", c.open], ["종가", c.close]].map(([k, v]) =>
            h("div", {}, h("span", {}, k), h("b", { class: `num ${dpDir(v)}` }, v == null ? "진행 중" : dpPct(v))))),
        h("p", { class: "dp-sub" }, c.after),
        h("p", {}, c.lesson));
    })));
}

function dpRenderNotes() {
  const d = DP.data, ru = d.rules, R = d.research;
  mount($("#dp-notes"),
    h("p", {}, `급락 종목은 나스닥 공개 종목표에서 시총 10억 달러 이상인 미국 상장 보통주(워런트·우선주 제외)를 골라 Yahoo 시세로 훑습니다. ‘지금’은 장전이면 전일 종가, 장중·장후면 전전일 종가 대비 등락입니다. 장후에는 정규장 등락에 장후 등락을 더합니다. 미국 시각 20:00~04:00(한국 낮)에는 Yahoo에 새 시세가 없어 지난 세션을 보여 줍니다.`),
    h("p", {}, `프리장 저가는 5분봉 종가가 −${7}% 이하인 봉의 저가로만 잽니다. Yahoo 프리장 봉에는 한 번 찍고 사라지는 오류 체결이 많기 때문입니다. 시각별 조건은 1시간봉 종가가 −${ru.deep}% 이하이고 직전 봉도 −10% 이하이며 두 봉 모두 가격 폭이 있을 때만 맞습니다(연구와 같은 정의).`),
    h("p", {}, `‘과거 플러스였던 조건’ 세 가지는 연구에서 두 기간 모두 플러스였던 칸입니다. 08:00 조건은 68개 후보, 09:29 조건은 48개 후보, 바닥 15분은 60개 조합 중에서 골랐습니다. 앞의 둘은 표본이 작고(17~19건, 13·6건), 바닥 15분은 표본은 크지만(222~250건) 최근 기간의 플러스 폭이 우연과 구별되지 않을 만큼 얇습니다. 매수 신호가 아니라 관찰 표시입니다.`),
    h("p", {}, `바닥 15분은 정규장 3분봉 마감 때 판단합니다: 그날 저가가 15분 동안 새로 나오지 않았고, 종가가 그 저가의 +2% 안이며, 전일 종가 대비 −7% 이하인 첫 봉(09:45~11:00 뉴욕). 그날 바닥일 확률은 같은 연구에서 15분마다 그 순간 −7% 이하인 종목을 저가 뒤 지난 시간과 지금 저가 대비 위치로 나눠, 칸마다 종목·날 한 번씩 잰 뒤 3분봉 값입니다(2026-08-10~10-06, 뒤 3분봉 15건 미만 칸은 ‘표본 적음’).`),
    h("p", {}, `정규장 규칙 재생은 3분봉 종가로 판단하고 체결한 것으로 계산합니다. 실제 체결은 호가 차이만큼 나쁠 수 있고, 급락주는 장전 호가가 0.5~2% 벌어지기도 합니다. 수수료는 0입니다(§1).`),
    h("p", {}, `사용자 주문 가능 시간은 한국 08:00~24:00입니다. 서머타임이면 정규장 첫 1시간 30분(뉴욕 09:30~11:00), 표준시(11월 첫 일요일부터)면 첫 30분만 볼 수 있습니다. 자는 동안에도 현재가보다 높은 지정가 매도·낮은 지정가 매수와 손절 주문을 걸어 둘 수 있습니다. 이 탭의 규칙 재생은 연구와 같게 손절을 11:00까지만 판단합니다.`),
    h("p", {}, `자료: 종목표·뉴스는 나스닥(api.nasdaq.com), 시세·1분봉은 Yahoo Finance(yfinance). 1분봉은 최근 30일만 있어 그보다 오래된 날은 다시 볼 수 없습니다. 연구 보고서: ${R.study.report}.`));
}

/* ------------------------------------------------------------------ boot */
(function dpBoot() {
  window.addEventListener("desk:tab", (e) => {
    if (e.detail !== "dip") return;
    if (!DP.data && !DP.loading) dpLoad(false);
    else if (DP.detail) dpDrawChart();  // 숨겨진 동안 폭이 0이었다
  });
  let rt;
  window.addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => { if (document.body.dataset.tab === "dip" && DP.detail) dpDrawChart(); }, 150);
  });
  // 장전·장중에는 2분마다 목록을 다시 받는다(서버도 2분 캐시). 다른 탭이나 숨긴 창에서는 쉬지 않는다.
  DP.timer = setInterval(() => {
    if (document.hidden || document.body.dataset.tab !== "dip" || !DP.data) return;
    if (DP.data.session === "장전" || DP.data.session === "장중") dpLoad(false);
  }, 120000);
})();
