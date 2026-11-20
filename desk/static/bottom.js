"use strict";
/* 바닥 감지기(docs/desk_system.md §13.6): SOXX 하락 저점 대비 반등 폭. 관찰용이며 주문·비중을 바꾸지 않는다. */
const BD = { data: null, request: 0 };
const BD_KIND = { calm: "하락 국면 아님", falling: "하락 중 · 반등 확인 전", confirmed: "저점 대비 7% 반등 확인" };
const BD_BANDS = ["0~3%", "3~5%", "5~7%", "7% 이상", "9% 이상"];
const bdPct = (v, d = 1) => (v == null ? "—" : pct(v, d));
const bdRate = (v) => (v == null ? "—" : (v * 100).toFixed(v >= 0.95 && v < 1 ? 1 : 0) + "%");

async function bdLoad(refresh = false) {
  const seq = ++BD.request, root = $("#bd-view");
  if (!BD.data) mount(root, h("p", { class: "nw-meta", role: "status" }, "SOXX 바닥 예상 구간을 계산하고 있습니다…"));
  const button = $("#bd-refresh");
  if (button) { button.disabled = true; button.textContent = "종가 확인 중…"; }
  try {
    const data = await api("/api/bottom" + (refresh ? "?refresh=1" : ""));
    if (seq !== BD.request) return;
    BD.data = data;
    if (!root.hidden) bdRender();
  } catch (error) {
    if (seq !== BD.request) return;
    const box = h("div", { class: "nw-error", role: "alert" }, BD.data ? "이전 조회 결과입니다. 가격 갱신 실패 · " : "예측 자료를 불러오지 못했습니다. ", error.message, " ",
      h("button", { class: "btn", onclick: () => bdLoad(true) }, "다시 계산"));
    if (BD.data) { bdRender(); root.prepend(box); } else mount(root, box);
  }
}

function bdOutlook(d) {
  const o = d.outlook, v = o?.validation, paper = o?.paper;
  const dated = o?.as_of || d.status.date;
  const stale = dated < d.latest_completed;
  const valid = o?.available && [o.lower, o.center, o.upper, o.close].every(Number.isFinite);
  const score = (hits, n) => `${hits}/${n}회`;
  const errorPct = value => Number.isFinite(value) ? value.toFixed(2) + "%" : "—";
  const metric = (name, value) => h("div", {}, h("dt", {}, name), h("dd", {}, value));
  return h("section", {class: "card bd-outlook", "aria-labelledby": "bd-outlook-title"},
    h("div", {class: "bd-top"},
      h("span", {class: "eyebrow"}, `${d.symbol} · 미국 반도체 ETF · ${dated} 미국장 종가 기준`),
      h("button", {id: "bd-refresh", class: "btn", onclick: () => bdLoad(true)}, "가격 갱신")),
    h("h1", {id: "bd-outlook-title"}, o?.horizon ? `다음 ${o.horizon}거래일 바닥 예상 구간` : "바닥 예상 구간"),
    stale && h("p", {class: "bd-warn", role: "status"}, `이전 종가로 계산한 전망입니다. 최근 마감일(${d.latest_completed}) 자료를 기다립니다.`),
    valid ? [
      h("p", {class: "bd-range"}, usd(o.lower), h("span", {}, " – "), usd(o.upper)),
      h("p", {class: "bd-target"}, `미국 정규장 종가 중 최저값의 예상 범위입니다. 종가가 ${usd(o.lower)} 아래로 내려가면 하단을 이탈합니다.`),
      o.caveat && h("p", {class: "bd-warn", role: "note"}, (o.in_dd_now ? "지금은 하락 국면입니다. " : "") + o.caveat),
      h("dl", {class: "bd-metrics"},
        metric("예상 저점 중앙값", usd(o.center)),
        metric("기준 종가", usd(o.close)),
        metric("종가 대비 중앙값", bdPct(o.center / o.close - 1)),
        metric("과거 재생 · 구간 안", v?.n ? score(v.hits, v.n) : "채점 대기"),
        metric("과거 재생 · 하단 이탈", v?.n ? score(v.below, v.n) : "채점 대기"),
        metric("실제 기록 · 구간 안", paper ? (paper.n ? score(paper.hits, paper.n) : "완료 0회") : "확인 불가")),
      h("details", {class: "bd-validation"}, h("summary", {}, "예상 구간 산정 방식과 오차"),
        h("p", {}, `예측일 다음 세션부터 ${o.horizon}거래일의 최저 종가를 봅니다. 산정 방식은 ‘${o.method}’이며, 완료된 학습 표본 ${o.n_train}건의 10·50·90백분위를 사용합니다.`),
        v?.n ? h("p", {}, `과거 재생 ${v.start} ~ ${v.end} · 겹치지 않는 ${v.n}회 중 구간 안 ${v.hits}회, 하단 이탈 ${v.below}회, 상단 초과 ${v.above}회. 기준 종가 대비 중앙값의 평균 절대오차는 ${errorPct(v.mae_pct)}입니다. 기존 관측 저점을 그대로 쓴 오차는 ${errorPct(v.observed_low_mae_pct)}(${v.observed_low_n ?? 0}회)입니다.`)
          : h("p", {}, "채점할 수 있는 과거 구간이 아직 없습니다."),
        paper && h("p", {}, `장 시작 전에 고정한 실제 예측 중 ${paper.n}회 채점, ${paper.pending}회 관찰 중` +
          (paper.unscored ? `, 가격 누락으로 ${paper.unscored}회 미채점.` : ".")))]
      : h("p", {class: "bd-missing"}, o?.reason || "예상 구간을 계산할 자료가 없습니다. 가격을 갱신해 주세요."));
}

function bdAudit(a) {
  const heading = h("h2", {}, "어떤 신호 뒤에 저점이 유지됐나");
  if (!a?.rows?.length) return h("section", {id: "bd-audit", class: "card"}, heading,
    h("p", {class: "nw-meta"}, "비교할 신호 기록을 기다립니다."));
  const counts = (hits, total) => total ? `${hits}/${total} (${bdRate(hits / total)})` : "완료 0건";
  return h("section", {id: "bd-audit", class: "card bd-audit"},
    heading, h("p", {}, a.method),
    h("p", {class: "nw-meta"}, `${a.period} · 가격 기준일 ${a.as_of}`),
    h("div", {class: "scroll"}, h("table", {},
      h("thead", {}, h("tr", {}, ["신호와 판정", "20일 완료분 저점 유지", "같은 기간 하락일", "20일 미도래"].map(label => h("th", {scope: "col"}, label)))),
      h("tbody", {}, a.rows.map(r => h("tr", {},
        h("td", {}, h("b", {}, r.label), h("small", {}, r.role)),
        h("td", {}, counts(r.hits, r.n), r.early_breaks?.length > 0 &&
          h("small", {class: "bd-early-break"}, `미도래 중 이미 이탈 ${r.early_breaks.length}건`)),
        h("td", {}, r.baseline_n ? counts(r.baseline_hits, r.baseline_n) : "—"),
        h("td", {}, `${r.pending}건`)))))),
    a.ledger_review && h("p", {}, a.ledger_review),
    h("details", {}, h("summary", {}, "신호별 표본과 해석"), h("ul", {}, a.rows.map(r => h("li", {}, h("b", {}, r.label + " · "), r.detail)))),
    h("p", {class: "nw-meta"}, a.limitations));
}

function bdDetail(st, line) {
  if (st.kind === "calm")
    return `60일 고점 ${usd(st.high)}(${md(st.peak)}) 대비 ${bdPct(st.dd)}. 종가가 ${usd(st.dd_trigger)} 아래로 내려가면 하락 국면으로 셉니다.`;
  if (st.kind === "falling")
    return `저점 ${usd(st.low)}(${md(st.low_date)})에서 ${bdPct(st.up)}. 7% 선 ${usd(line.price)}까지 ${bdPct(line.gap)} 남았습니다.`;
  return `현재 종가는 ${md(st.low_date)} 저점 ${usd(st.low)}에서 ${bdPct(st.up)} 반등해 7% 선 ${usd(line.price)}에 도달했습니다.`;
}

function bdSignalLine(sig, hold) {
  if (!sig) return null;
  const text = {
    watching: `확인 ${md(sig.date)} · 관찰 ${sig.day}/${hold}거래일 · 저점 ${usd(sig.low)} 유지 중`,
    broke: `확인 ${md(sig.date)} 뒤 ${md(sig.on)}에 저점 ${usd(sig.low)} 이탈(${sig.day}거래일째) — 한 달 유지 실패`,
    held: `확인 ${md(sig.date)} 뒤 ${hold}거래일 동안 저점 ${usd(sig.low)} 유지`,
  }[sig.status];
  return h("p", { class: "bd-signal bd-" + sig.status }, text);
}

function bdGauge(st) {
  // 하락 국면이면 저점 대비 반등(0~12%, 7·8·10% 눈금), 아니면 60일 고점 대비 낙폭(0~10%)
  const rebound = st.kind !== "calm";
  const max = rebound ? Math.max(0.12, (st.up || 0) + 0.02) : 0.10;
  const value = rebound ? Math.max(0, st.up || 0) : Math.min(0.10, Math.max(0, -(st.dd || 0)));
  const ticks = rebound ? st.lines.map((x) => x.k) : [0.05, 0.10];
  return h("div", { class: "bd-gauge-wrap", "aria-hidden": "true" },
    h("div", { class: "bd-gauge" },
      h("div", { class: "bd-fill", style: `width:${Math.min(100, (value / max) * 100)}%` }),
      ticks.map((k) => h("div", { class: "bd-tick" + (rebound && k === 0.07 ? " main" : ""), style: `left:${(k / max) * 100}%` }))),
    h("div", { class: "bd-ticklabels" },
      h("span", { style: "left:0" }, rebound ? "저점" : "고점"),
      ticks.map((k) => h("span", { style: `left:${(k / max) * 100}%` }, (rebound ? "+" : "−") + Math.round(k * 100) + "%"))));
}

function bdTiles(st, sig, hold) {
  const line = st.lines[0];
  const tiles = [
    ["SOXX 종가", usd(st.close), md(st.date)],
    ["60일 고점", usd(st.high), `${md(st.peak)} · ${bdPct(st.dd)}`],
    ["이번 하락 저점", st.kind === "calm" ? "—" : usd(st.low), st.kind === "calm" ? "하락 국면 아님" : md(st.low_date)],
    ["저점 대비", st.kind === "calm" ? "—" : bdPct(st.up), st.kind === "calm" ? "" : `7% 선 ${usd(line.price)}`],
    ["관찰", sig && sig.status === "watching" ? `${sig.day}/${hold}` : "—", sig ? `확인 ${md(sig.date)}` : "진행 중인 확인 없음"],
  ];
  return h("div", { class: "bd-tiles" }, tiles.map(([label, value, foot]) =>
    h("div", { class: "bd-tile" }, h("div", { class: "eyebrow" }, label), h("b", {}, value), h("small", {}, foot))));
}

function bdTable(d, st) {
  const cols = d.periods.filter((p) => d.tables[p.key]);
  const row = (label, cells, on) => h("tr", { class: on ? "on" : null }, h("td", {}, label),
    cells.map((c) => h("td", {}, bdRate(c && c.rate), h("small", {}, c ? `${c.n}일` : ""))));
  return h("section", { class: "card bd-table" },
    h("h3", {}, "반등 폭별 ‘한 달 안 저점 유지’ 비율"),
    h("p", { class: "nw-meta" }, "하락 국면의 거래일마다 집계해 한 번의 반등이 여러 날에 포함될 수 있습니다. 현재 반등 폭에 해당하는 행을 표시했습니다."),
    h("div", { class: "scroll" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "저점 대비"), cols.map((p) => h("th", {}, p.label)))),
      h("tbody", {},
        BD_BANDS.map((label, i) => row(label, cols.map((p) => d.tables[p.key].bands[i]), st.band === i)),
        row("하락 국면 아무 날", cols.map((p) => d.tables[p.key].base), false)))));
}

function bdHistory(d) {
  const hold = d.rules.hold;
  const done = d.signals.filter((s) => s.status !== "watching");
  const held = done.filter((s) => s.status === "held").length;
  const result = (s) => ({ held: "유지", broke: `이탈 ${md(s.on)}`, watching: `관찰 ${s.day}/${hold}` }[s.status]);
  return h("section", { class: "card bd-table" },
    h("h3", {}, `지난 확인 기록 · ${held}/${done.length} 유지`),
    h("p", { class: "nw-meta" }, "하락 국면에서 반등 폭이 처음 7%를 넘은 날입니다. 최저·20일 뒤는 그날 종가 대비(정규장 종가끼리, 체결 성과 아님)."),
    h("div", { class: "scroll" }, h("table", {},
      h("thead", {}, h("tr", {}, ["확인일", "종가", "저점", "반등 폭", "20거래일 결과", "20일 안 최저", "20일 뒤"].map((s) => h("th", {}, s)))),
      h("tbody", {}, d.signals.slice().reverse().map((s) => h("tr", {},
        h("td", {}, s.date), h("td", {}, usd(s.close)), h("td", {}, `${usd(s.low)} · ${md(s.low_date)}`),
        h("td", {}, bdPct(s.up)), h("td", { class: "bd-" + s.status }, result(s)),
        h("td", { class: cls(s.worst) }, bdPct(s.worst)), h("td", { class: cls(s.ret20) }, bdPct(s.ret20))))))));
}

function bdRender() {
  const d = BD.data, st = d.status, lv = d.live, hold = d.rules.hold, root = $("#bd-view");
  const sig = st.signal && st.signal.current ? st.signal : null;
  const broke = sig?.status === "broke";
  const line = st.lines.find((x) => x.k === d.rules.confirm) || st.lines[0];
  const ending = st.kind === "falling" && st.dd_left != null && st.dd_left <= 5
    ? (st.dd_left === 0 ? "오늘이 하락 국면 마지막 날입니다" : `새로 −10% 날이 없으면 하락 국면은 ${st.dd_left}거래일 뒤 끝납니다`) +
      " — 그 전에 7%를 넘지 못하면 이번 하락은 확인 없이 지나갑니다." : null;
  const chartHost = h("div", { class: "chart", id: "bd-chart" });
  mount(root,
    bdOutlook(d),
    sectionNav("뉴스 근거 바로가기", [["bd-confirmation", "반등 확인"], ["bd-audit", "신호별 검증"], ["nw-body", "기사 기록"]]),
    d.notices.map((t) => h("p", { class: "nw-note" }, t)),
    d.warnings.map((t) => h("p", { class: "nw-error", role: "alert" }, t)),
    h("section", { id: "bd-confirmation", class: "card bd-card bd-" + (broke ? "failed-confirmation" : st.kind) },
      h("div", { class: "eyebrow" }, `이미 관측한 저점 · ${d.symbol} · ${st.date} 종가 기준`),
      h("h2", { class: "bd-title" }, broke ? "최근 확인 저점 이탈" : BD_KIND[st.kind]),
      bdSignalLine(sig, hold),
      h("p", {}, bdDetail(st, line)),
      bdGauge(st),
      ending && h("p", { class: "bd-warn" }, ending),
      lv && h("p", { class: "bd-live" }, h("b", {}, `잠정 · ${lv.session} ${lv.time_kst} KST `),
        `${usd(lv.price)} → 저점 대비 ${bdPct(lv.up)} · ${BD_KIND[lv.kind]}`, h("small", {}, " (지금 값을 오늘 종가로 가정, 확정은 정규장 종가)")),
      bdTiles(st, sig, hold)),
    bdAudit(d.signal_audit),
    h("section", { class: "card bd-chart-card" },
      h("h3", {}, "최근 6개월 · 종가와 이번 하락의 저점·7% 선"),
      h("div", { class: "legend" },
        h("span", {}, swatch("var(--ink-2)"), "종가"), h("span", {}, swatch("var(--down)"), "하락 저점"),
        h("span", {}, swatch("var(--accent)"), "7% 선"), h("span", {}, swatch("var(--context)"), "60일 고점"),
        h("span", {}, "▲ 확인(채움 = 유지, 빨강 = 이탈, 빈칸 = 관찰 중)")),
      chartHost),
    h("details", {class: "bd-notes"}, h("summary", {}, "과거 반등 폭별 유지 비율과 확인 기록"),
      h("div", { class: "bd-grid" }, bdTable(d, st), bdHistory(d))),
    h("details", { class: "bd-notes" }, h("summary", {}, "계산 방식·주의"),
      h("ul", {}, d.rules.notes.map((t) => h("li", {}, t))),
      h("p", { class: "nw-meta" }, `가격: ${d.source} · 계산 ${d.fetched_at}` +
        (d.long_range ? ` · ^SOX ${d.long_range[0]}~${d.long_range[1]}` : "") + ` · 근거: ${d.rules.source}`)));
  const c = d.chart, idx = new Set(c.dates);
  const markers = d.signals.filter((s) => idx.has(s.date)).map((s) => ({
    date: s.date, value: s.close, shape: "up",
    fill: s.status === "held" ? "var(--accent)" : s.status === "broke" ? "var(--critical)" : "var(--surface)",
    stroke: s.status === "broke" ? "var(--critical)" : "var(--accent)",
    label: `확인 ${s.date} · ${{ held: "20거래일 유지", broke: "저점 이탈 " + s.on, watching: "관찰 중" }[s.status]}`,
  }));
  lineChart(chartHost, {
    dates: c.dates, height: 280, aria: "SOXX 종가와 하락 저점, 7% 선",
    series: [
      { label: "60일 고점", values: c.high, color: "var(--context)", width: 1.2, context: true },
      { label: "종가", values: c.close, color: "var(--ink-2)", width: 1.8 },
      { label: "하락 저점", values: c.low, color: "var(--down)", width: 1.6 },
      { label: "7% 선", values: c.line, color: "var(--accent)", width: 1.6 },
    ],
    markers, fmtY: (v) => "$" + Math.round(v), fmtTip: (v) => usd(v),
  });
}

window.addEventListener("desk:tab", (e) => {
  if (e.detail === "news" && !$("#bd-view").hidden) {
    if (BD.data) bdRender();  // 숨겨진 동안 그린 차트 폭을 다시 잰다
    bdLoad();
  }
});
