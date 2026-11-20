"use strict";
/* 시즌 계기판 — renders /api/season inside the 추세 등급 tab (정본 docs/desk_system.md §11.2).
   관찰용: 7종목 규칙·주문 초안을 바꾸지 않고, 매수 추천·비중·수량이 없다. 다이얼 선택은 서버가 세션마다
   처음 계산한 순간 종이 기록(season_paper.json)에 잠근다. Reuses h()/$/mount/md/pct/cls/table/lineChart from app.js. */

const SS = { data: null, loading: null };
const SS_PICK = { MOM: "현행 7종목", REV: "최근 5일 최하위 7종목", CASH: "현금(달러)" };

function ssNum(v, d = 2) { return v == null ? "—" : (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d); }

async function ssLoad(force) {
  if (SS.loading) return SS.loading;
  SS.loading = fetch("/api/season" + (force ? "?refresh=1" : ""), { cache: "no-store" })
    .then(async (r) => { const j = await r.json().catch(() => ({})); if (!r.ok) throw new Error(j.error || r.statusText); return j; })
    .then((d) => { SS.data = d; ssRender(); if (force) toast("시즌 계기판을 다시 계산했습니다"); })
    .catch((e) => {
      SS.data = { error: /not found/i.test(e.message)
        ? "시즌 계기판을 열려면 데스크를 재시작하세요. 실행 창을 닫고 ‘야간데스크.cmd’를 다시 실행하면 됩니다."
        : "시즌 계기판을 계산하지 못했습니다. " + e.message };
      ssRender();
    })
    .finally(() => { SS.loading = null; });
  return SS.loading;
}

function ssRender() {
  const host = $("#tg-season");
  if (!host) return;
  const d = SS.data;
  if (!d) { mount(host, h("div", { class: "eyebrow" }, "시즌 계기판"), h("p", { class: "muted" }, "7종목 계산을 읽어 시즌을 판정하는 중입니다…")); return; }
  if (d.error) { mount(host, h("div", { class: "eyebrow" }, "시즌 계기판"), h("p", { class: "err" }, d.error)); return; }
  const tile = (label, value, sub, klass) => h("div", { class: "gi-kpi" }, h("div", { class: "gi-kpi-label" }, label),
    h("div", { class: `gi-kpi-value ${klass || ""}` }, value), h("div", { class: "gi-kpi-sub" }, sub));
  const p1 = d.picks.dial1, p2 = d.picks.dial2;
  const lastSwitch = (p) => (p.switches.length ? `최근 전환 ${md(p.switches[p.switches.length - 1].date)} → ${SS_PICK[p.switches[p.switches.length - 1].to]}` : "2026 전환 없음");
  const kpis = h("div", { class: "gi-kpis tg-kpis ss-kpis" },
    tile("지금 시즌", d.season.label, d.season.why, `ss-season ss-${d.season.label === "모멘텀 시즌" ? "mom" : d.season.label === "급락 경계" ? "down" : "rev"}`),
    tile("모멘텀 성적 h20·K20", ssNum(d.ic.h20_k20), "많이 오른 종목이 그 뒤 20일에도 더 올랐나(최근 20개 확정값 평균, −1~+1)", cls(d.ic.h20_k20)),
    tile("모멘텀 성적 h10·K10", ssNum(d.ic.h10_k10), "같은 값을 10일 뒤 수익·최근 10개로 잰 빠른 판", cls(d.ic.h10_k10)),
    tile("SOXX 20일선", d.soxx.up == null ? "—" : d.soxx.up ? "위" : "아래",
      `${md(d.as_of)} 종가 ${d.soxx.close.toFixed(2)} · 20일선 ${d.soxx.sma20.toFixed(2)}`, d.soxx.up ? "up" : "down"),
    tile(`${p1.name}의 ${md(d.session)} 선택`, SS_PICK[p1.pick], `${p1.rule}. ${lastSwitch(p1)}`),
    tile(`${p2.name}의 ${md(d.session)} 선택`, SS_PICK[p2.pick], `${p2.rule}. ${lastSwitch(p2)}`));
  const basket = (title, obj) => h("div", { class: "ss-basket" }, h("div", { class: "gi-ctl-label" }, title),
    Object.keys(obj).length ? h("div", { class: "ss-tags" }, Object.entries(obj).sort((a, b) => b[1] - a[1])
      .map(([t, w]) => h("span", { class: "tag", title: `비중 ${pctAbs(w)}` }, `${t} ${pctAbs(w, 0)}`))) : h("p", { class: "muted" }, "없음"));
  const hist = table([["2026 사후 재생"], ["1/2 이후", 1], ["7/1 이후", 1], ["최대 낙폭", 1]],
    d.history.map((r) => h("tr", {}, h("td", {}, r.name), h("td", { class: `r ${cls(r.ret)}` }, pct(r.ret)),
      h("td", { class: `r ${cls(r.since_jul)}` }, pct(r.since_jul)), h("td", { class: "r down" }, pct(r.mdd)))));
  const pp = d.paper || {};
  const paper = !pp.sessions
    ? h("p", { class: "muted" }, `첫 기록: ${md(pp.start || d.session)} 세션 선택을 잠갔습니다. 그 세션 장 마감 뒤부터 성적을 채점합니다.`)
    : table([["종이 기록 " + md(pp.start) + " 이후"], ["수익", 1]],
      [["성격 다이얼", pp.dial1], ["두 다이얼", pp.dial2], ["현행 7종목", pp.MOM], ["최근 5일 최하위 7종목", pp.REV], ["QQQ", pp.QQQ]]
        .map(([n, v]) => h("tr", {}, h("td", {}, n), h("td", { class: `r ${cls(v)}` }, pct(v)))));
  mount(host,
    h("div", { class: "ss-head" },
      h("div", {}, h("div", { class: "eyebrow" }, `시즌 계기판 · 관찰·종이 기록 · ${md(d.as_of)} 종가 기준 · 계산 ${d.computed_at.slice(5, 16).replace("T", " ")}`),
        h("h2", {}, "지금은 ", h("span", { class: "ss-label" }, d.season.label)),
        h("p", { class: "muted" }, "모멘텀이 통하는 시즌인지, 두 다이얼이 다음 개장에 무엇을 들었을지 보여 줍니다. 7종목 규칙·주문 초안은 바꾸지 않습니다.")),
      h("button", { class: "btn", title: "다시 계산합니다", onclick: (e) => { e.currentTarget.disabled = true; ssLoad(true); } }, "↻ 시즌 갱신")),
    kpis,
    h("div", { class: "ss-grid" },
      h("div", {}, h("div", { class: "gi-ctl-label" }, "확정된 모멘텀 성적(최근 90거래일) · 0 아래면 많이 오른 종목이 그 뒤에 덜 오른 구간"),
        h("div", { class: "legend" }, h("span", {}, swatch("var(--tg-ma20)", "height:3px"), "h20·K20(성격 다이얼)"),
          h("span", {}, swatch("var(--tg-ma50)", "height:3px"), "h10·K10(두 다이얼)")),
        h("div", { class: "chart", id: "ss-chart" })),
      h("div", {}, basket(`${md(d.session)} 현행 7종목`, d.baskets.MOM), basket(`${md(d.session)} 최근 5일 최하위 7종목`, d.baskets.REV))),
    h("div", { class: "ss-grid" }, hist, h("div", {}, paper,
      pp.missing && pp.missing.length ? h("p", { class: "muted" }, `기록이 빠진 세션 ${pp.missing.length}개(직전 선택 유지로 계산): ${pp.missing.slice(-5).map(md).join(", ")}`) : null,
      d.after_open ? h("p", { class: "muted" }, `${md(d.session)} 선택은 개장 뒤에 처음 잠갔습니다(쓰는 정보는 ${md(d.as_of)} 종가까지).`) : null)),
    h("details", { class: "notes-panel" }, h("summary", {}, "계기판 읽는 법·한계"), h("footer", { class: "notes" }, d.notes.map((n) => h("p", {}, n)))));
  ssDrawChart();
}

function ssDrawChart() {
  const host = $("#ss-chart");
  if (!host || !SS.data || SS.data.error || document.body.dataset.tab !== "trend") return;
  const c = SS.data.chart;
  lineChart(host, { dates: c.days, height: 220, aria: "확정된 모멘텀 성적 h20·K20과 h10·K10",
    series: [{ label: "h20·K20", values: c.ic20, color: "var(--tg-ma20)", width: 1.8 },
      { label: "h10·K10", values: c.ic10, color: "var(--tg-ma50)", width: 1.6 },
      { label: "0", values: c.days.map(() => 0), color: "var(--ink-3)", width: 1, context: true }],
    fmtY: (v) => v.toFixed(1), fmtTip: (v) => ssNum(v) });
}

(function ssBoot() {
  window.addEventListener("desk:tab", (e) => {
    if (e.detail !== "trend") return;
    if (!SS.data && !SS.loading) ssLoad(false);
    else ssDrawChart();
  });
  let rt;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(ssDrawChart, 150); });
})();
