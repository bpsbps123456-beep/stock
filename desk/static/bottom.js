"use strict";
/* 바닥 감지기(docs/desk_system.md §13.6): SOXX 하락 저점 대비 반등 폭. 관찰용이며 주문·비중을 바꾸지 않는다. */
const BD = { data: null, request: 0 };
const BD_KIND = { calm: "하락 국면 아님", falling: "하락 중 · 바닥 확인 전", confirmed: "바닥 확인 · 저점 대비 +7% 이상" };
const BD_BANDS = ["0~3%", "3~5%", "5~7%", "7% 이상", "9% 이상"];
const bdPct = (v, d = 1) => (v == null ? "—" : pct(v, d));
const bdRate = (v) => (v == null ? "—" : (v * 100).toFixed(v >= 0.95 && v < 1 ? 1 : 0) + "%");

async function bdLoad(refresh = false) {
  const seq = ++BD.request, root = $("#bd-view");
  if (!BD.data) mount(root, h("p", { class: "nw-meta" }, "바닥 감지기 계산 중… (처음 계산은 10초 안팎)"));
  try {
    const data = await api("/api/bottom" + (refresh ? "?refresh=1" : ""));
    if (seq !== BD.request) return;
    BD.data = data;
    bdRender();
  } catch (error) {
    if (seq !== BD.request) return;
    const box = h("div", { class: "nw-error", role: "alert" }, "바닥 감지기: ", error.message, " ",
      h("button", { class: "btn", onclick: () => bdLoad(true) }, "다시 계산"));
    if (BD.data) { bdRender(); root.prepend(box); } else mount(root, box);
  }
}

function bdDetail(st, line) {
  if (st.kind === "calm")
    return `60일 고점 ${usd(st.high)}(${md(st.peak)}) 대비 ${bdPct(st.dd)}. 종가가 ${usd(st.dd_trigger)} 아래로 내려가면 하락 국면으로 셉니다.`;
  if (st.kind === "falling")
    return `저점 ${usd(st.low)}(${md(st.low_date)})에서 ${bdPct(st.up)}. 7% 선 ${usd(line.price)}까지 ${bdPct(line.gap)} 남았습니다.`;
  return `저점 ${usd(st.low)}(${md(st.low_date)})에서 ${bdPct(st.up)} 올라 7% 선 ${usd(line.price)}을 넘었습니다. 뜻: 이 저점이 앞으로 20거래일 안에 종가로 다시 깨질 가능성이 낮았다는 것까지입니다.`;
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
    h("p", { class: "nw-meta" }, "하락 국면 날 전부를 셉니다. 지금 칸을 칠했습니다. 2024~ 수치는 지금 장세의 값이고, 긴 이력·약세장 칸이 장세가 바뀔 때의 위험입니다."),
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
  const line = st.lines.find((x) => x.k === d.rules.confirm) || st.lines[0];
  const rates = st.band == null ? null : d.periods.filter((p) => d.tables[p.key])
    .map((p) => `${p.label} ${bdRate(d.tables[p.key].bands[st.band].rate)}`).join(" · ");
  const ending = st.kind === "falling" && st.dd_left != null && st.dd_left <= 5
    ? (st.dd_left === 0 ? "오늘이 하락 국면 마지막 날입니다" : `새로 −10% 날이 없으면 하락 국면은 ${st.dd_left}거래일 뒤 끝납니다`) +
      " — 그 전에 7%를 넘지 못하면 이번 하락은 확인 없이 지나갑니다." : null;
  const chartHost = h("div", { class: "chart", id: "bd-chart" });
  mount(root,
    d.notices.map((t) => h("p", { class: "nw-note" }, t)),
    d.warnings.map((t) => h("p", { class: "nw-error", role: "alert" }, t)),
    h("section", { class: "card bd-card bd-" + st.kind },
      h("div", { class: "bd-top" },
        h("div", { class: "eyebrow" }, `바닥 감지기 · ${d.symbol} · ${st.date} 종가 기준`),
        h("button", { class: "btn", onclick: () => bdLoad(true) }, "가격 갱신")),
      h("h2", { class: "bd-title" }, BD_KIND[st.kind]),
      h("p", {}, bdDetail(st, line)),
      bdGauge(st),
      rates && h("p", { class: "nw-meta" }, `이 반등 구간(${BD_BANDS[st.band]})의 과거 한 달 저점 유지: ${rates}`),
      ending && h("p", { class: "bd-warn" }, ending),
      bdSignalLine(sig, hold),
      lv && h("p", { class: "bd-live" }, h("b", {}, `잠정 · ${lv.session} ${lv.time_kst} KST `),
        `${usd(lv.price)} → 저점 대비 ${bdPct(lv.up)} · ${BD_KIND[lv.kind]}`, h("small", {}, " (지금 값을 오늘 종가로 가정, 확정은 정규장 종가)")),
      bdTiles(st, sig, hold),
      h("p", { class: "nw-meta" }, "매수 신호가 아닙니다. 확인 뒤에도 진입가에서 5% 넘게 다시 빠진 경우가 절반 이상이었고, 가장 낮은 바닥 보증도 아닙니다(최종 저점 5/7).")),
    h("section", { class: "card bd-chart-card" },
      h("h3", {}, "최근 6개월 · 종가와 이번 하락의 저점·7% 선"),
      h("div", { class: "legend" },
        h("span", {}, swatch("var(--ink-2)"), "종가"), h("span", {}, swatch("var(--down)"), "하락 저점"),
        h("span", {}, swatch("var(--accent)"), "7% 선"), h("span", {}, swatch("var(--context)"), "60일 고점"),
        h("span", {}, "▲ 확인(채움 = 유지, 빨강 = 이탈, 빈칸 = 관찰 중)")),
      chartHost),
    h("div", { class: "bd-grid" }, bdTable(d, st), bdHistory(d)),
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
  if (e.detail === "news") {
    if (BD.data) bdRender();  // 숨겨진 동안 그린 차트 폭을 다시 잰다
    bdLoad();
  }
});
