"use strict";
/* 추세 등급 — renders /api/trend (정본 docs/desk_system.md §11). Reuses h()/svg()/$/mount/md/toast/
   sectionNav/lineChart/table from app.js and the gi-* controls from giant.css. 관찰용 화면이다: 매수 추천·
   목표 비중·주문 수량이 없다. 등급은 순서가 있는 값이라 브라스 한 색조의 밝기 단계로 칠하고(검증 통과,
   trend.css), 글자(S~D)를 항상 함께 쓴다. 상승 = red, 하락 = blue, 부호와 ▲▼를 함께 쓴다. */

const TG = { data: null, loading: null, sel: null, chart: null, cLoading: null, basis: "day",
  grade: "all", market: "all", fund: "all", query: "", sort: null, ready: false };
try { if (localStorage.getItem("tg-basis") === "week") TG.basis = "week"; } catch (_) { /* storage unavailable */ }
const TG_GRADES = ["S", "A", "B", "C", "D"];
const TG_RANK = { S: 0, A: 1, B: 2, C: 3, D: 4 };
const TG_MARKETS = [["all", "전체"], ["us", "미국"], ["intl", "미국 밖"]];
const TG_FUNDS = [["all", "두 펀드"], ["TIME", "TIME"], ["KoAct", "KoAct"], ["both", "둘 다 보유"]];

function tgSegment(label, items, cur, set) {  // 거인의 어깨와 같은 모양(gi-seg)
  return h("div", { class: "gi-ctl" }, h("span", { class: "gi-ctl-label" }, label),
    h("div", { class: "gi-seg", role: "radiogroup", "aria-label": label }, items.map(([k, text]) =>
      h("button", { role: "radio", "aria-checked": String(cur === k), onclick: () => set(k) }, text))));
}
const tgPct = (v, d = 1) => (v == null ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}%`);
const tgDir = (v) => (v == null || v === 0 ? "" : v > 0 ? "up" : "down");
const tgNorm = (t) => ({ GOOGL: "GOOG" }[t] || t);
const tgName = (s) => s.ko || s.name;

function tgGrade(g, extra = "") {
  return g ? h("span", { class: `tg-g g-${g} ${extra}` }, g) : h("span", { class: "tg-g none", title: "등급을 매기지 않았습니다" }, "—");
}

// 등급 기준: 일별(매일 종가) 또는 주간(직전 주 마지막 거래일 종가로 정해 한 주 동안 고정). 판단 단위는 주가 근거가
// 가장 좋았다(정본 §11 '등급의 시간 단위'). 보드·표·분포·차트 띠가 고른 기준을 따른다.
const tgWeek = () => TG.basis === "week";
const tgG = (s) => (tgWeek() ? s.wgrade : s.grade) || null;
const tgP = (s) => (tgWeek() ? s.wprev : s.prev) || null;
const tgHist = (s) => (tgWeek() ? s.whist : s.hist) || [];
const tgStreak = (s) => (tgWeek() ? `${s.wstreak}주째` : `${s.streak}거래일째`);
function tgDist() {
  const out = { S: 0, A: 0, B: 0, C: 0, D: 0, none: 0 };
  for (const s of TG.data.stocks) out[tgG(s) || "none"] += 1;
  return out;
}

// 등급 변화: 일별은 어제, 주간은 지난주 대비. 오르면 ▲(빨강), 내리면 ▼(파랑).
function tgMove(s) {
  const g = tgG(s), p = tgP(s);
  if (!g || !p || g === p) return null;
  const up = TG_RANK[g] < TG_RANK[p];
  return h("span", { class: `tg-move ${up ? "up" : "down"}`, title: tgWeek() ? `지난주 ${p} → 이번 주 ${g}` : `어제 ${p} → 오늘 ${g}` },
    `${up ? "▲" : "▼"} ${p}`);
}

// 당일 점수 등급: 10일 평균하지 않은 오늘 점수로 매긴 등급. 평균 등급과 다를 때만 붙인다(급락 때 먼저 바뀐다).
// 주간 기준에서는 더 빠른 일별 등급을, 일별 기준에서는 당일 점수 등급을 다를 때만 붙인다.
function tgRaw(s) {
  const g = tgG(s), fast = tgWeek() ? s.grade : s.raw;
  if (!fast || !g || fast === g) return null;
  const up = TG_RANK[fast] < TG_RANK[g];
  const tip = tgWeek() ? `오늘 일별 등급은 ${fast}입니다(이번 주 주간 등급 ${g})` : `오늘 하루 점수만 보면 ${fast}등급입니다(10일 평균 등급 ${g})`;
  return h("span", { class: `tg-raw ${up ? "up" : "down"}`, title: tip }, `${tgWeek() ? "일별" : "당일"} ${fast}`);
}

// 실적 발표가 가까우면(7일 안) 또는 막 지났으면(5일 안) 꼬리표. S 등급 변화가 가장 많이 몰리는 사건이다.
function tgEarnTag(s) {
  const e = s.earn;
  if (!e) return null;
  if (e.next_in != null && e.next_in <= 7)
    return h("span", { class: "tag tg-earn", title: `다음 실적 발표 ${e.next}` }, e.next_in === 0 ? "실적 오늘" : `실적 D-${e.next_in}`);
  if (e.last_ago != null && e.last_ago <= 5)
    return h("span", { class: "tag tg-earn", title: `실적 발표 ${e.last}` }, `실적 ${e.last_ago}일 전`);
  return null;
}
const tgEarnText = (e) => (!e ? "—" : e.next ? `${md(e.next)} (D-${e.next_in})` : e.last ? `지난 ${md(e.last)}` : "—");

// 내 계좌·주문 탭의 상태(app.js S)에서 7종목과 내 보유를 읽는다. 데스크 계산 전이면 빈 집합.
function tgMine() {
  if (typeof S === "undefined" || !S) return { seven: new Set(), mine: new Set() };
  return { seven: new Set((S.targets || []).map((r) => tgNorm(r.ticker))),
           mine: new Set(((S.account || {}).positions || []).map((p) => tgNorm(p.ticker))) };
}
function tgTags(s, mine) {
  const t = tgNorm(s.symbol);
  return [mine.seven.has(t) ? h("span", { class: "tag in", title: "이번 주 7종목 전략 보유" }, "7종목") : null,
          mine.mine.has(t) ? h("span", { class: "tag tg-mine", title: "내 계좌 보유" }, "내 보유") : null];
}

// 등급 이력 띠: 최근 거래일마다 한 칸. 칸 색은 등급, 마우스를 올리면 날짜·등급.
function tgStrip(hist, days) {
  return h("span", { class: "tg-strip", role: "img",
    "aria-label": `최근 ${hist.length}거래일 등급: ${hist.map((g) => g || "—").join(" ")}` },
  hist.map((g, i) => h("i", { class: g ? `g-${g}` : "none", title: `${md(days[i])} ${g || "—"}` })));
}

/* ------------------------------------------------------------------ load */
async function tgFetch(url) {
  const r = await fetch(url, { cache: "no-store" });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

async function tgLoad(force) {
  if (TG.loading) return TG.loading;
  const root = $("#trend-view");
  root.classList.add("tg-busy");
  if (!TG.data) {
    mount(root, h("section", { class: "card tg-wait" }, h("div", { class: "eyebrow" }, "추세 등급"),
      h("p", {}, "두 펀드 보유 종목의 2년 일봉을 받아 이평선 추세 등급을 계산하고 있습니다(약 10~20초). 이후 60분 동안은 저장된 결과를 보여 줍니다.")));
  }
  TG.loading = tgFetch("/api/trend" + (force ? "?refresh=1" : ""))
    .then((d) => {
      TG.data = d;
      if (!TG.sel || !d.stocks.some((s) => s.symbol === TG.sel)) TG.sel = (d.stocks.find((s) => tgG(s) === "S") || d.stocks[0])?.symbol;
      tgRender();
      tgSelect(TG.sel, false);
      if (force) toast("추세 등급을 다시 계산했습니다");
    })
    .catch((e) => {
      const msg = /not found/i.test(e.message)
        ? "추세 등급을 열려면 데스크를 재시작하세요. 실행 창을 닫고 ‘야간데스크.cmd’를 다시 실행하면 됩니다."
        : "추세 등급을 계산하지 못했습니다. " + e.message;
      if (TG.data) toast(msg);
      else mount(root, h("section", { class: "card" }, h("p", { class: "err" }, msg)));
    })
    .finally(() => {
      TG.loading = null;
      root.classList.remove("tg-busy");
      const again = $("#tg-head .btn");
      if (again) again.disabled = false;
    });
  return TG.loading;
}

async function tgSelect(sym, scroll = true) {
  if (!sym) return;
  TG.sel = sym;
  const row = TG.data.stocks.find((x) => x.symbol === sym);
  TG.chart = row && row.why === "시세 없음" ? { error: "이 종목은 시세를 받지 못해 차트가 없습니다." } : null;
  tgRenderBoard(); tgRenderTable(); tgRenderDetail();
  if (scroll) jumpTo("tg-detail");
  if (TG.chart) return;
  const key = sym;
  TG.cLoading = key;
  try {
    const c = await tgFetch(`/api/trend/chart?t=${encodeURIComponent(sym)}`);
    if (TG.cLoading !== key) return;
    TG.chart = c;
  } catch (e) {
    if (TG.cLoading !== key) return;
    TG.chart = { error: e.message };
  }
  tgRenderDetail();
}

/* ------------------------------------------------------------- structure */
function tgRender() {
  const root = $("#trend-view");
  if (!TG.ready || !$("#tg-board", root)) {
    mount(root,
      h("section", { class: "page-heading tg-head", id: "tg-head" }),
      sectionNav("추세 등급 바로가기", [["tg-kpis", "요약"], ["tg-board", "등급별 종목"], ["tg-detail", "종목 차트"],
        ["tg-table", "전체 표"], ["tg-evidence", "등급 검증"]]),
      h("section", { class: "alerts", id: "tg-alerts", "aria-live": "polite" }),
      h("section", { class: "gi-kpis tg-kpis", id: "tg-kpis", "aria-label": "추세 등급 요약" }),
      h("section", { class: "card tg-board", id: "tg-board" }),
      h("section", { class: "card tg-detail", id: "tg-detail" }),
      h("section", { class: "card tg-table", id: "tg-table" }),
      h("section", { class: "card tg-evidence", id: "tg-evidence" }),
      h("details", { class: "notes-panel" }, h("summary", {}, "계산 기준·등급 읽는 법·자료 출처"),
        h("footer", { class: "notes", id: "tg-notes" })));
    TG.ready = true;
  }
  tgRenderHead(); tgRenderAlerts(); tgRenderKpis(); tgRenderBoard(); tgRenderDetail(); tgRenderTable();
  tgRenderEvidence(); tgRenderNotes();
}

function tgRenderHead() {
  const d = TG.data;
  const funds = Object.entries(d.funds).map(([k, v]) => `${k} ${md(v)}`).join(" · ");
  mount($("#tg-head"),
    h("div", {},
      h("div", { class: "eyebrow" }, `시세 기준 ${md(d.as_of)} 종가 · 펀드 보유 내역 ${funds} · 계산 ${d.computed_at.slice(5, 16).replace("T", " ")} (한국시간)`),
      h("h1", {}, "추세 등급 ", h("span", { class: "tg-h1-sub" }, `두 펀드 보유 ${d.stocks.length}종목의 이평선 추세`)),
      h("p", {}, "지금 가격이 이평선들 위에서 얼마나 강한 모양인지 S~D로 나눕니다. 거인의 어깨 회복 등급(급락에서 되찾은 정도)과 달리 오늘의 추세 상태만 봅니다. 매수 추천이나 주문 수량은 없습니다.")),
    h("div", { class: "page-meta tg-head-tools" },
      tgSegment("등급 기준", [["day", "일별"], ["week", d.week_from ? `주간 (${md(d.week_from)} 종가)` : "주간"]], TG.basis, (k) => {
        TG.basis = k;
        try { localStorage.setItem("tg-basis", k); } catch (_) { /* storage unavailable */ }
        tgRender(); tgDrawChart();
      }),
      h("button", { class: "btn", title: "최신 일봉을 받아 다시 계산합니다(약 10~20초)", onclick: (e) => {
        e.currentTarget.disabled = true; tgLoad(true);
      } }, "↻ 등급 갱신")));
}

function tgRenderAlerts() {
  const d = TG.data;
  const items = (d.warnings || []).map((w) => h("div", { class: "banner warn" }, h("span", { class: "ico", "aria-hidden": "true" }, "!"), h("span", {}, w)));
  mount($("#tg-alerts"), items);
}

function tgRenderKpis() {
  const d = TG.data, b = d.breadth;
  const tile = (label, value, sub, extra) => h("div", { class: "gi-kpi" },
    h("div", { class: "gi-kpi-label" }, label), h("div", { class: "gi-kpi-value" }, value), h("div", { class: "gi-kpi-sub" }, sub), extra || null);
  const dist = tgDist();
  const graded = TG_GRADES.reduce((n, g) => n + (dist[g] || 0), 0);
  const bar = h("div", { class: "tg-distbar", role: "img",
    "aria-label": TG_GRADES.map((g) => `${g} ${dist[g] || 0}종목`).join(", ") },
  TG_GRADES.map((g) => h("i", { class: `g-${g}`, style: `flex:${dist[g] || 0}`, title: `${g} ${dist[g] || 0}종목` })));
  const ups = d.stocks.filter((s) => tgG(s) && tgP(s) && TG_RANK[tgG(s)] < TG_RANK[tgP(s)]).length;
  const downs = d.stocks.filter((s) => tgG(s) && tgP(s) && TG_RANK[tgG(s)] > TG_RANK[tgP(s)]).length;
  const soon = d.stocks.filter((s) => s.earn && s.earn.next_in != null && s.earn.next_in <= 7);
  const q = d.bench;
  mount($("#tg-kpis"),
    tile(tgWeek() ? "주간 등급 분포" : "등급 분포", TG_GRADES.map((g) => `${g} ${dist[g] || 0}`).join(" · "),
      `${graded}종목 등급${dist.none ? ` · 미산정 ${dist.none}(상장 ${d.rules.min_bars}거래일 미만·시세 없음)` : ""}`, bar),
    tile("50일선 위", b.above50 == null ? "—" : `${b.above50}%`, `200일선 위 ${b.above200 == null ? "—" : b.above200 + "%"} · 등급을 매긴 ${b.n}종목 중`),
    tile(tgWeek() ? "이번 주 등급 변화" : "오늘 등급 변화", h("span", {}, h("span", { class: "up" }, `▲ ${ups}`), " · ", h("span", { class: "down" }, `▼ ${downs}`)),
      tgWeek() ? `지난주(${md(d.week_prev)} 종가) 대비 올라간·내려간 종목` : `직전 거래일(${md(d.prev_day)}) 대비 올라간·내려간 종목`),
    tile("7일 안 실적 발표", `${soon.length}종목`, soon.length ? soon.slice(0, 6).map((s) => s.symbol).join(" · ") + (soon.length > 6 ? " …" : "")
      : "S 등급이 가장 자주 바뀌는 때입니다"),
    tile(`${q.symbol} 위치`, tgPct(q.d50), `50일선 대비 · 200일선 대비 ${tgPct(q.d200)} · 52주 고점 대비 ${tgPct(q.h52)}`),
    tile("S 등급 기준", `상위 ${Math.round((1 - d.rules.ladder[0][1]) * 100)}%`,
      `점수 백분위 하한: ${d.rules.ladder.map(([g, f]) => `${g} ${Math.round(f * 100)}`).join(" · ")} · 그 아래 D`));
}

/* ----------------------------------------------------------------- board */
function tgRenderBoard() {
  const d = TG.data, mine = tgMine();
  const kept = $$("#tg-board .tg-col-list").map((el) => el.scrollTop);  // 칸마다 목록 스크롤 위치를 지킨다
  const cols = TG_GRADES.map((g) => {
    const list = d.stocks.filter((s) => tgG(s) === g).sort((a, b) => (b.pct ?? -1) - (a.pct ?? -1));
    return h("div", { class: "tg-col" },
      h("div", { class: "tg-col-head" }, tgGrade(g), h("b", {}, `${list.length}종목`), h("span", { class: "muted" }, d.rules.labels[g])),
      h("div", { class: "tg-col-list" }, list.length ? list.map((s) => h("button", {
        class: "tg-chip" + (s.symbol === TG.sel ? " sel" : ""), "aria-pressed": String(s.symbol === TG.sel), onclick: () => tgSelect(s.symbol),
        title: `${s.name} · ${s.sub || s.sector} · 점수 ${s.pct == null ? "—" : s.pct.toFixed(0)} · ${tgStreak(s)} ${g}`,
      }, h("b", { class: "mono" }, s.symbol), h("span", { class: "tg-chip-name" }, tgName(s)), tgMove(s), tgRaw(s), tgEarnTag(s), tgTags(s, mine)))
        : h("p", { class: "muted tg-none" }, "없음")));
  });
  const none = d.stocks.filter((s) => !tgG(s));
  mount($("#tg-board"),
    h("header", {}, h("h2", {}, "등급별 종목"),
      h("span", { class: "sub" }, tgWeek() ? `주간 등급(${md(d.week_from)} 종가로 정해 한 주 동안 고정) · ▲▼는 지난주 등급 · 종목을 누르면 차트를 엽니다`
        : `점수 높은 순 · ▲▼는 어제 등급 · 종목을 누르면 차트를 엽니다`)),
    h("div", { class: "tg-cols" }, cols),
    none.length ? h("p", { class: "gi-foot" }, "등급 미산정: ", none.map((s, i) => [i ? ", " : "",
      h("button", { class: "linkbtn", onclick: () => tgSelect(s.symbol) }, `${s.symbol}(${s.why || "자료 없음"})`)])) : null);
  $$("#tg-board .tg-col-list").forEach((el, i) => { if (kept[i]) el.scrollTop = kept[i]; });
}

/* ---------------------------------------------------------------- detail */
function tgRenderDetail() {
  const d = TG.data, host = $("#tg-detail");
  const s = d.stocks.find((x) => x.symbol === TG.sel);
  if (!s) { mount(host, h("p", { class: "muted" }, "종목을 고르세요.")); return; }
  const mine = tgMine();
  const fact = (label, value, cls = "") => h("div", {}, h("span", {}, label), h("b", { class: cls }, value));
  mount(host,
    h("header", {},
      h("h2", {}, tgGrade(tgG(s), "big"), ` ${s.symbol} `, h("span", { class: "tg-h2-name" }, tgName(s))),
      h("span", { class: "sub" }, `${s.market} · ${s.sub || s.sector} · ${Object.entries(s.funds).map(([k, v]) => `${k} ${v.toFixed(2)}%`).join(" · ")}`,
        s.proxy_used ? ` · 상장 ${s.proxy_used.own_bars}거래일이라 등급·이평선 대비·차트는 ${s.proxy_used.symbol} 가격 이력(종가·수익률은 ${s.symbol} 자체)` : ""),
      h("div", { class: "right" }, tgTags(s, mine))),
    h("div", { class: "dp-facts tg-facts" },
      fact("점수(백분위)", s.pct != null ? s.pct.toFixed(0) : "—"),
      fact("일별 등급", s.grade || "—"),
      fact(d.week_from ? `주간 등급(${md(d.week_from)} 종가)` : "주간 등급", s.wgrade || "—"),
      fact("등급 유지", tgG(s) ? tgStreak(s) : "—"),
      fact("당일 점수 등급", s.raw || "—", s.raw && s.grade && s.raw !== s.grade ? (TG_RANK[s.raw] < TG_RANK[s.grade] ? "up" : "down") : ""),
      fact("다음 실적", s.earn?.next ? `${md(s.earn.next)} (D-${s.earn.next_in})` : "—"),
      fact("지난 실적", s.earn?.last ? `${md(s.earn.last)} (${s.earn.last_ago}일 전)` : "—"),
      fact("5거래일 전", s.g5 || "—"),
      fact("종가", s.close != null ? s.close.toLocaleString("en-US", { maximumFractionDigits: 2 }) : "—"),
      fact("1일", tgPct(s.chg1d), tgDir(s.chg1d)),
      fact("1개월", tgPct(s.r1m), tgDir(s.r1m)),
      fact("3개월", tgPct(s.r3m), tgDir(s.r3m)),
      fact("20일선 대비", tgPct(s.d20), tgDir(s.d20)),
      fact("50일선 대비", tgPct(s.d50), tgDir(s.d50)),
      fact("200일선 대비", tgPct(s.d200), tgDir(s.d200)),
      fact("정배열", s.order == null ? "—" : s.order ? "종가>20>50>200" : "아님"),
      fact("52주 고점 대비", tgPct(s.h52), tgDir(s.h52))),
    tgG(s) === "S" && d.check?.risk ? h("p", { class: "tg-risk" }, d.check.risk) : null,
    h("div", { class: "tg-parts" }, h("span", { class: "gi-ctl-label" }, "점수 재료(그날 보유 종목 안 백분위)"),
      Object.entries(s.parts || {}).map(([k, v]) => h("span", { class: "tg-part", title: d.rules.part_notes[k] || "" },
        `${k} `, h("b", {}, v == null ? "—" : v.toFixed(0))))),
    h("div", { class: "tg-hist" }, h("span", { class: "gi-ctl-label" }, `최근 ${d.days.length}거래일 ${tgWeek() ? "주간 " : ""}등급`), tgStrip(tgHist(s), d.days)),
    h("div", { class: "legend tg-legend", id: "tg-legend" }),
    h("div", { class: "chart tg-chart", id: "tg-chart" }));
  tgDrawChart();
}

function tgDrawChart() {
  const host = $("#tg-chart");
  if (!host || document.body.dataset.tab !== "trend") return;
  const c = TG.chart;
  if (!c) { mount(host, h("p", { class: "muted" }, "차트를 불러오는 중입니다…")); return; }
  if (c.error) { mount(host, h("p", { class: "err" }, "차트를 불러오지 못했습니다. " + c.error)); return; }
  const series = [
    { label: "종가", values: c.close, color: "var(--ink-2)", width: 1.8 },
    { label: "20일선", values: c.ma20, color: "var(--tg-ma20)", width: 1.6 },
    { label: "50일선", values: c.ma50, color: "var(--tg-ma50)", width: 1.6 },
    { label: "200일선", values: c.ma200, color: "var(--tg-ma200)", width: 1.6 },
  ];
  // 좁은 차트(560px 미만)는 lineChart가 최신값과 함께 범례를 직접 그린다.
  mount($("#tg-legend"), host.clientWidth < 560 ? null : series.map((s) => h("span", {}, swatch(s.color, "height:3px"), s.label)));
  const fmt = (v) => v.toLocaleString("en-US", { maximumFractionDigits: v < 20 ? 2 : 0 });
  lineChart(host, { dates: c.dates, series, fmtY: fmt, fmtTip: (v) => v.toLocaleString("en-US", { maximumFractionDigits: 2 }),
    height: 300, aria: `${c.symbol} 종가와 20·50·200일 이동평균` });
  // 등급 띠를 차트 그림 영역(lineChart: 왼쪽 58px, 좁은 차트 48px, 오른쪽 14px)과 같은 x에 맞춘다.
  // 칸 i의 가운데가 날짜 i의 x에 오도록 띠를 반 칸씩 양쪽으로 넓힌다.
  const W = Math.max(280, host.clientWidth), n = c.dates.length;
  const ml = W < 560 ? 48 : 58, span = W - ml - 14, cell = n > 1 ? span / (n - 1) : span;
  host.appendChild(h("div", { class: "tg-chart-strip" },
    h("span", { class: "gi-ctl-label", style: `margin-left:${ml}px` }, tgWeek() ? "같은 날짜에 쓰던 주간 등급" : "같은 날짜의 등급"),
    h("span", { class: "tg-strip wide", role: "img", "aria-label": "차트 기간의 날짜별 등급",
      style: `margin-left:${ml - cell / 2}px;width:${span + cell}px` },
    ((tgWeek() && c.wgrades) || c.grades).map((g, i) => h("i", { class: g ? `g-${g}` : "none", title: `${md(c.dates[i])} ${g || "—"}` })))));
}

/* ----------------------------------------------------------------- table */
const TG_COLS = [
  ["symbol", "종목", (s) => s.symbol],
  ["grade", "등급", (s) => (tgG(s) ? TG_RANK[tgG(s)] * 1000 - (s.pct ?? 0) : 1e9)],
  ["streak", "유지", (s) => (tgWeek() ? s.wstreak : s.streak)],
  ["earn", "실적", (s) => (s.earn && s.earn.next_in != null ? s.earn.next_in : null)],
  ["chg1d", "1일", (s) => s.chg1d],
  ["r1m", "1개월", (s) => s.r1m],
  ["r3m", "3개월", (s) => s.r3m],
  ["d20", "20일선", (s) => s.d20],
  ["d50", "50일선", (s) => s.d50],
  ["d200", "200일선", (s) => s.d200],
  ["h52", "52주 고점", (s) => s.h52],
  ["w", "펀드 비중", (s) => Math.max(...Object.values(s.funds))],
];

function tgFiltered() {
  const d = TG.data, q = TG.query.trim().toLowerCase();
  return d.stocks.filter((s) => {
    if (TG.grade !== "all" && (TG.grade === "none" ? tgG(s) : tgG(s) !== TG.grade)) return false;
    if (TG.market === "us" && s.market !== "미국") return false;
    if (TG.market === "intl" && s.market === "미국") return false;
    if (TG.fund === "both" && Object.keys(s.funds).length < 2) return false;
    if ((TG.fund === "TIME" || TG.fund === "KoAct") && !(TG.fund in s.funds)) return false;
    if (q && ![s.symbol, s.name, s.ko || "", s.sub || "", s.sector || ""].some((x) => x.toLowerCase().includes(q))) return false;
    return true;
  });
}

function tgSorted(rows) {
  const [key, dir] = TG.sort || ["grade", 1];
  const col = TG_COLS.find(([k]) => k === key);
  return [...rows].sort((a, b) => {
    const va = col[2](a), vb = col[2](b);
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    return (typeof va === "string" ? va.localeCompare(vb) : va - vb) * dir;
  });
}

function tgRenderTable() {
  const d = TG.data, mine = tgMine();
  const old = $("#tg-table .gi-scroll");
  const scroll = old ? [old.scrollTop, old.scrollLeft] : [0, 0];
  const rows = tgSorted(tgFiltered());
  const [skey, sdir] = TG.sort || ["grade", 1];
  const head = TG_COLS.map(([k, label]) => h("th", { class: k === "symbol" ? "" : "r", "aria-sort": k === skey ? (sdir > 0 ? "ascending" : "descending") : null },
    h("button", { class: "tg-sort", onclick: () => {
      TG.sort = [k, k === skey ? -sdir : (["symbol", "grade", "earn"].includes(k) ? 1 : -1)];
      tgRenderTable();
    } }, label, k === skey ? (sdir > 0 ? " ↑" : " ↓") : "")));
  head.splice(4, 0, h("th", {}, `최근 ${d.days.length}거래일`));
  const body = rows.map((s) => h("tr", { class: s.symbol === TG.sel ? "tg-sel" : null, tabindex: "0",
    "aria-selected": String(s.symbol === TG.sel), onclick: () => tgSelect(s.symbol),
    onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); tgSelect(s.symbol, false); } } },
    h("td", {}, h("div", { class: "tg-cell-name" }, h("b", { class: "mono" }, s.symbol), h("span", {}, tgName(s)), tgTags(s, mine),
      s.proxy_used ? h("span", { class: "tag", title: `상장 ${s.proxy_used.own_bars}거래일이라 등급·이평선 대비·차트는 ${s.proxy_used.symbol} 가격 이력으로 냅니다` }, "원주 이력") : null)),
    h("td", { class: "r num" }, tgGrade(tgG(s)), " ", h("span", { class: "muted" }, s.pct != null ? s.pct.toFixed(0) : ""), " ", tgMove(s), " ", tgRaw(s)),
    h("td", { class: "r num" }, tgG(s) ? (tgWeek() ? `${s.wstreak}주` : `${s.streak}일`) : "—"),
    h("td", { class: "r num", title: s.earn?.last ? `지난 실적 ${s.earn.last}` : null }, tgEarnText(s.earn)),
    h("td", {}, tgStrip(tgHist(s), d.days)),
    ...["chg1d", "r1m", "r3m", "d20", "d50", "d200", "h52"].map((k) => h("td", { class: `r num ${tgDir(s[k])}` }, tgPct(s[k]))),
    h("td", { class: "r num", title: Object.entries(s.funds).map(([k, v]) => `${k} ${v.toFixed(2)}%`).join(" · ") },
      Object.entries(s.funds).map(([k, v]) => `${k === "TIME" ? "T" : "K"} ${v.toFixed(1)}`).join(" · "))));
  mount($("#tg-table"),
    h("header", {}, h("h2", {}, "전체 종목"),
      h("span", { class: "sub" }, `${rows.length}/${d.stocks.length}종목 · 이평선 대비는 종가 기준 % · 52주 고점은 장중 고가 기준`)),
    h("div", { class: "gi-filters" },
      tgSegment(tgWeek() ? "주간 등급" : "등급", [["all", `전체 ${d.stocks.length}`], ...TG_GRADES.map((g) => [g, `${g} ${tgDist()[g] || 0}`]),
        ...(tgDist().none ? [["none", `미산정 ${tgDist().none}`]] : [])], TG.grade, (k) => { TG.grade = k; tgRenderTable(); }),
      tgSegment("시장", TG_MARKETS, TG.market, (k) => { TG.market = k; tgRenderTable(); }),
      tgSegment("펀드", TG_FUNDS, TG.fund, (k) => { TG.fund = k; tgRenderTable(); }),
      giSearch("tg-q", "종목 찾기", "티커·회사명·분야", TG.query, (v) => { TG.query = v; tgRenderTable(); })),
    rows.length ? h("div", { class: "gi-scroll tg-scroll" }, h("table", { class: "gi-table tg-tbl" },
      h("thead", {}, h("tr", {}, head)), h("tbody", {}, body)))
      : h("p", { class: "muted" }, "조건에 맞는 종목이 없습니다."));
  const nu = $("#tg-table .gi-scroll");
  if (nu) { nu.scrollTop = scroll[0]; nu.scrollLeft = scroll[1]; }
}

/* -------------------------------------------------------------- evidence */
function tgRealized(r) {
  if (!r) return null;
  const cell = (g) => {
    const x = r.by_grade[g];
    return h("div", {}, h("span", {}, `${g} (${x.n})`), h("b", { class: tgDir(x.ret) }, tgPct(x.ret)));
  };
  return h("div", { class: "tg-realized" },
    h("div", { class: "gi-ctl-label" }, `최근 ${r.days}거래일 성적 — ${md(r.from)} 종가 등급별로 그 뒤 오늘 종가까지 평균 수익(사후 확인)`),
    h("div", { class: "dp-facts tg-facts" }, TG_GRADES.map(cell),
      h("div", {}, h("span", {}, "등급 대상 전체"), h("b", { class: tgDir(r.all) }, tgPct(r.all))),
      h("div", {}, h("span", {}, "QQQ"), h("b", { class: tgDir(r.bench) }, tgPct(r.bench)))),
    h("p", { class: "gi-foot" }, "지난 성적일 뿐 다음 기간을 맞히는 값이 아닙니다. 2026년 시험에서는 최근 성적이 좋았던 직후(6월) 오히려 크게 뒤집혔습니다."));
}

function tgRenderEvidence() {
  const c = TG.data.check;
  if (!c) { mount($("#tg-evidence"), null); return; }
  const signed = (v) => h("td", { class: `r num ${tgDir(v)}` }, v == null ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(1)}`);
  mount($("#tg-evidence"),
    h("header", {}, h("h2", {}, "등급 검증"), h("span", { class: "sub" }, c.subtitle)),
    h("p", { class: "tg-verdict" }, c.verdict),
    tgRealized(TG.data.realized),
    c.tables.map((t) => h("div", { class: "tg-check" }, h("div", { class: "gi-ctl-label" }, t.title),
      h("div", { class: "scroll" }, table([["구간"], ...t.columns.map((x) => [x, 1])],
        t.rows.map(([label, vals]) => h("tr", {}, h("td", {}, label), vals.map(signed))))))),
    h("ul", { class: "tg-points" }, c.points.map((p) => h("li", {}, p))),
    h("p", { class: "gi-foot" }, `근거 보고서: ${c.report}`));
}

function tgRenderNotes() {
  const r = TG.data.rules;
  mount($("#tg-notes"), r.notes.map((p) => h("p", {}, p)));
}

/* ------------------------------------------------------------------ boot */
(function tgBoot() {
  // 7종목·내 보유 꼬리표는 내 계좌·주문 탭의 상태(S)를 읽는다. 그 상태가 늦게 오거나 바뀌면 다시 그린다.
  document.addEventListener("desk:state", () => {
    if (TG.data && !$("#trend-view").hidden) { tgRenderBoard(); tgRenderTable(); tgRenderDetail(); }
  });
  window.addEventListener("desk:tab", (e) => {
    if (e.detail !== "trend") return;
    if (!TG.data && !TG.loading) tgLoad(false);
    else if (TG.data) { tgRenderBoard(); tgRenderTable(); tgRenderDetail(); }  // 7종목·내 보유 표시와 숨겨진 동안 폭 0이던 차트
  });
  let rt;
  window.addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => { if (document.body.dataset.tab === "trend" && TG.chart) tgDrawChart(); }, 150);
  });
})();
