"use strict";
/* 야간 데스크 — renders /api/state. No dependencies. All DOM is built with h()/svg(), so
   text (journal notes, fund names, tickers) is always set as text, never parsed as HTML.
   Charts are hand-built SVG: hover crosshair, keyboard stepping, end labels, table twin. */

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
let S = null;
let priceTicker = null;
let planAmount = null;
let offline = false;
const tableMode = { model: false, account: false };
const tabScroll = new Map();
// 비중 상한은 서버의 config.RULE을 그대로 보여 준다(2026-10-10 반도체 60%→50%).
const capPct = (key) => (S && S.rule && S.rule[key] != null ? `${Math.round(S.rule[key] * 100)}%` : "—");

/* ------------------------------------------------------------------ dom */
function h(tag, props, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "style") n.style.cssText = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "data") Object.assign(n.dataset, v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  append(n, kids);
  return n;
}
function append(n, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false || kid === "") continue;
    n.append(kid instanceof Node ? kid : String(kid));
  }
}
const mount = (root, ...kids) => { root.replaceChildren(); append(root, kids); };
const SVGNS = "http://www.w3.org/2000/svg";
function svg(tag, attrs, parent) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
const swatch = (color, style = "") => h("i", { class: "swatch", style: `background:${color};${style}` });

function sectionNav(label, items) {
  return h("nav", { class: "section-nav", "aria-label": label }, items.map(([id, text]) =>
    h("button", { type: "button", "data-jump": id }, text)));
}

function jumpTo(id) {
  document.getElementById(id)?.scrollIntoView({
    behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start",
  });
}

/* ------------------------------------------------------------------ format */
const nf = (v, d) => Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const usd = (v, d = 2) => (v == null ? "—" : (v < 0 ? "−$" : "$") + nf(Math.abs(v), d));
const krw = (v) => (v == null ? "—" : Math.round(v).toLocaleString("ko-KR") + "원");
const krwShort = (v) => {
  if (v == null) return "—";
  const a = Math.abs(v), s = v < 0 ? "−" : "";
  if (a >= 1e8) return s + (a / 1e8).toFixed(a >= 1e9 ? 1 : 2) + "억";
  if (a >= 1e4) return s + Math.round(a / 1e4).toLocaleString("ko-KR") + "만";
  return s + Math.round(a).toLocaleString("ko-KR");
};
const pct = (v, d = 1) => (v == null ? "—" : (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v * 100).toFixed(d) + "%");
const pctAbs = (v, d = 1) => (v == null ? "—" : (v * 100).toFixed(d) + "%");
const qtyFmt = (v) => (v == null ? "—" : v >= 100 ? nf(v, 1) : String(+Number(v).toFixed(4)));
const md = (s) => { if (!s) return "—"; const [, m, d] = s.split("-"); return `${+m}/${+d}`; };
const cls = (v) => (v == null || v === 0 ? "" : v > 0 ? "up" : "down");
const arw = (v) => (v == null || v === 0 ? "" : v > 0 ? "▲ " : "▼ ");
const daysSince = (d) => Math.max(0, Math.floor((Date.now() - new Date(d + "T00:00:00+09:00")) / 864e5));
const hhmm = (s) => (s || "").split(" ").pop();

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 3400);
}

/* --------------------------------------------------------------------- api */
async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  } catch (_) {  // no answer at all: the server process is not running
    setOffline(true);
    throw new Error("데스크에 연결할 수 없습니다. ‘야간데스크.cmd’를 다시 실행하세요.");
  }
  setOffline(false);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
function setOffline(on) {
  if (on === offline) return;
  offline = on;
  $("#offline").hidden = !on;
  if (on) {
    const retry = setInterval(async () => {
      try { await fetch("/api/ping"); } catch (_) { return; }
      clearInterval(retry);
      setOffline(false);
      load();
    }, 5000);
  }
}
async function load(kind) {
  $("#app").classList.add("loading");
  // The page opens before the server has the prices; on the first view say what the wait is.
  const slow = S ? null : setTimeout(() => mount($("#warnings"), h("div", { class: "banner" },
    h("span", { class: "ico", "aria-hidden": "true" }, "…"),
    h("span", null, "가격과 7종목 신호를 계산하는 중입니다. 하루 첫 실행은 가격을 새로 받느라 더 걸릴 수 있습니다 — 끝나면 저절로 채워집니다."))), 1200);
  try {
    S = kind === "refresh" ? await api("/api/refresh", { method: "POST", body: "{}" }) : await api("/api/state");
    render();
  } catch (e) {
    if (!offline) toast("화면 자료를 불러오지 못했습니다. " + e.message);
  } finally {
    clearTimeout(slow);
    if (!S) mount($("#warnings"));
    $("#app").classList.remove("loading");
  }
}
async function post(path, body) {
  $("#app").classList.add("loading");
  try { return await api(path, { method: "POST", body: JSON.stringify(body || {}) }); }
  finally { $("#app").classList.remove("loading"); }
}

/* ------------------------------------------------------------------ render */
function render() {
  renderChips(); renderWarnings(); renderTicket(); renderTiles();
  renderAlloc(); renderRank(); renderModel(); renderPrice(); renderPositions();
  renderAccount(); renderJournal(); renderRisk(); renderNotes();
  document.dispatchEvent(new CustomEvent("desk:state"));  // other tabs (AI 탐방) light 7종목·내 보유 from S
}

function renderChips() {
  const st = S.status, ses = S.session;
  const chip = (label, value) => h("span", { class: "chip" }, label, " ", h("b", null, value));
  mount($("#chips"),
    chip("다음 미국장", `${ses.label} ${hhmm(ses.open_kst)}`),
    chip("TIME PDF", md(st.pdf)),
    chip("종가", md(st.last_close)),
    chip("시세", st.quotes_at || "—"),
    chip("환율", nf(S.fx, 1)));
}

function renderWarnings() {
  mount($("#warnings"), S.status.warnings.map((w) =>
    h("div", { class: "banner" }, h("span", { class: "ico", "aria-hidden": "true" }, "!"), h("span", null, w))));
}

/* ------------------------------------------------------------ order ticket */
const targetOf = (tk) => S.targets.find((r) => r.ticker === tk);

function priceLine(r) {
  if (!r || r.price == null) return null;
  const q = r.quote || {};
  const when = q.session ? `${q.session} ${q.time_kst || ""}` : `${md(r.last_close_date)} 종가`;
  return [
    "현재 ", h("b", { class: "mono" }, usd(r.price)), " ", h("span", { class: "muted" }, `(${when})`),
    r.change != null ? [" · ", h("span", { class: cls(r.change) }, arw(r.change) + pct(r.change, 2))] : null,
  ];
}

function orderLine(a, i, hasBook) {
  const r = targetOf(a.ticker);
  const buy = a.side === "BUY";
  const why = r ? r.reasons.join(" · ") : (a.side === "SELL" ? "목표 포트폴리오에서 제외된 종목입니다" : "");
  const qty = a.qty != null
    ? [h("div", { class: "big" }, `${qtyFmt(a.qty)}주`), h("div", { class: "small" }, `정수 주 기준 ${a.whole ?? 0}주 · 약 ${usd(a.amount, 0)}`)]
    : [h("div", { class: "big" }, pctAbs(a.target_weight, 0)), h("div", { class: "small" }, hasBook ? "시세 확인이 필요합니다" : "투자 금액을 입력하면 수량을 계산합니다")];
  const fill = () => prefill(a.ticker, a.side, a.qty, a.price);
  return h("div", {
    class: "order", style: `animation-delay:${80 + i * 70}ms`, tabindex: "0", role: "button",
    "aria-label": `${a.ticker} ${buy ? "매수" : "매도"} 내역을 기록란에 입력`, onclick: fill,
    onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fill(); } },
  },
    h("span", { class: `stamp ${buy ? "buy" : "sell"}` }, buy ? "▲ 매수" : "▼ 매도", h("small", null, buy ? "BUY" : "SELL")),
    h("div", null,
      h("div", null, h("span", { class: "o-ticker" }, a.ticker), h("span", { class: "o-name" }, r ? r.name : "")),
      h("div", { class: "o-why" }, why, r && r.since ? h("span", { class: "muted" }, ` · 전략 편입일 ${md(r.since)}`) : null),
      h("div", { class: "o-price" }, priceLine(r), ` · 목표 ${pctAbs(a.target_weight, 0)}`)),
    h("div", { class: "o-qty" }, qty));
}

function renderTicket() {
  const ses = S.session, acct = S.account;
  const hasBook = acct.positions.length > 0 || acct.funded;
  const flag = ses.rebalance_day
    ? "주간 재조정일 · 7종목 구성과 비중을 조정합니다"
    : `보유 유지 기간 · 다음 재조정일 ${ses.next_rebalance}`;
  let orders = S.actions;
  if (!hasBook) {
    orders = S.targets.map((r) => {
      const amount = planAmount ? r.weight * planAmount : null;
      const q = amount && r.price ? amount / r.price : null;
      return { ticker: r.ticker, side: "BUY", target_weight: r.weight, price: r.price, qty: q, amount, whole: q ? Math.floor(q) : null };
    });
  }
  const active = orders.filter((a) => a.side !== "HOLD");
  const holds = orders.filter((a) => a.side === "HOLD");
  const body = active.map((a, i) => orderLine(a, i, hasBook));
  if (!active.length) {
    body.push(h("div", { class: "order", style: "cursor:default" }, h("span", { class: "stamp hold" }, "— 유지"),
      h("div", { class: "o-why" }, `현재 비중과 목표 비중의 차이가 허용 범위(${pctAbs(S.settings.tolerance, 0)}) 안이므로 추가 주문이 없습니다.`), h("div")));
  }
  if (holds.length) {
    body.push(h("div", { class: "holds" }, h("span", { class: "muted", style: "font-size:12px;align-self:center" }, "유지"),
      holds.map((a) => h("span", { class: "hold-pill", title: "현재 비중과 목표 비중의 차이가 허용 범위 안이므로 유지합니다" },
        h("b", null, a.ticker), `${pctAbs(a.current_weight, 0)} → ${pctAbs(a.target_weight, 0)}`))));
  }
  if (S.cash_target > 0.005) {
    body.push(h("div", { class: "holds" }, h("span", { class: "muted", style: "font-size:12px;align-self:center" }, "현금"),
      h("span", { class: "hold-pill", title: "종목·섹터별 비중 상한을 적용하고 남은 금액은 현금으로 둡니다" },
        h("b", null, pctAbs(S.cash_target, 0)), "비중 상한 적용 후 남은 현금")));
  }
  const steps = hasBook ? null : h("ol", { class: "empty-steps" },
    h("li", null, "아래 ‘매매 기록’에서 계좌의 달러 예수금(주문 가능한 현금)을 ", h("b", null, "입금"), "으로 먼저 기록하세요."),
    h("li", null, "기록한 계좌 내역을 바탕으로 목표 비중과 필요한 주문 수량을 계산합니다."),
    h("li", null, "주문이 체결되면 해당 종목을 누르고, 실제 체결 내역을 확인한 뒤 저장하세요."));
  const plan = hasBook ? null : h("label", { class: "plan" }, "투자 금액 ($)",
    h("input", { id: "plan", class: "mono", type: "number", min: "0", step: "100", value: planAmount ?? "", placeholder: "10000",
                 onchange: (e) => { planAmount = e.target.value ? +e.target.value : null; renderTicket(); } }));
  mount($("#ticket"),
    h("div", { class: "t-head" },
      h("div", null,
        h("div", { class: "eyebrow" }, "주문 안내 · 다음 미국 거래일"),
        h("div", { class: "t-title" }, `${ses.label} 미국장`),
        h("div", { class: "t-meta" }, "개장 ", h("b", null, hhmm(ses.open_kst)), " (한국시간) · 판단 기준 PDF ",
          h("b", null, md(S.status.pdf_used)), " · 종가 ", h("b", null, md(S.status.last_close))),
        h("span", { class: "flag" }, flag)),
      h("div", { class: "t-serial" }, S.strategy, h("br"), `NO. ${ses.date.replaceAll("-", "")}`)),
    steps,
    h("div", { class: "orders" }, body),
    h("div", { class: "t-foot" },
      h("span", { class: "grow" }, hasBook ? "체결 후 해당 종목을 누르면 매매 기록란에 입력됩니다." : "계좌 내역을 기록하기 전에는 목표 비중을 표시합니다."),
      plan));
}

function prefill(tk, side, qty, price) {
  const f = $("#jform");
  f.date.value = S.session.date;
  f.type.value = side === "SELL" ? "sell" : "buy";
  f.ticker.value = tk;
  f.qty.value = qty ? (+qty).toFixed(4) : "";
  f.price.value = price ? (+price).toFixed(2) : "";
  syncForm();
  $("#journal").scrollIntoView({ behavior: "smooth", block: "start" });
  setTimeout(() => f.qty.focus(), 350);
  toast(`${tk} ${side === "SELL" ? "매도" : "매수"} 내역을 입력했습니다. 실제 체결 날짜·수량·단가를 확인한 뒤 저장하세요.`);
}

/* ------------------------------------------------------------------ tiles */
function tile(delay, label, value, foot, valueClass = "") {
  return h("div", { class: "tile", style: `animation-delay:${delay}ms` },
    h("div", { class: "label" }, label), h("div", { class: `value ${valueClass}` }, value), h("div", { class: "foot" }, foot));
}
function renderTiles() {
  const a = S.account, fx = S.fx, st = S.model.stats;
  const pnl = a.unrealized + a.realized;
  const booked = a.positions.length || a.funded;
  mount($("#tiles"),
    tile(60, [swatch("var(--s-account)"), "내 계좌 평가액"],
      booked ? krw(a.equity * fx) : h("span", { style: "font-size:20px" }, "기록 전"),
      booked ? h("span", { class: "mono" }, `${usd(a.equity)} · 현금 ${usd(a.cash)} · 주식 ${usd(a.market_value)}`) : "입금·매수를 기록하면 평가액이 표시됩니다"),
    tile(120, "전체 손익 (매도 손익 + 보유 손익)",
      [arw(pnl) + usd(pnl), a.contributed > 0 ? h("small", null, pct(pnl / a.contributed)) : null],
      `매도한 종목 ${usd(a.realized)} · 보유 종목 ${usd(a.unrealized)}`, cls(pnl)),
    tile(180, [swatch("var(--s-strategy)"), `7종목 전략 모의 성과 · ${md(S.model.start)} 이후 원화`],
      [pct(st.total), h("small", null, `TIME ETF ${pct(S.model.etf_total)} (${md(S.model.etf_through)}까지)`)],
      `전략 최대 하락폭 ${pct(st.mdd)} · 과거 자료로 계산한 성과입니다`),
    tile(240, "전략의 고점 대비 하락폭",
      [st.dd_now > -0.0005 ? "고점" : pct(st.dd_now),
       h("small", null, `최근 20거래일 ETF 대비 ${S.model.vs_etf_20d == null ? "—" : pct(S.model.vs_etf_20d) + "p"}`)],
      "하락폭이 커지면 아래 ‘투자 점검’을 확인하세요"));
}

/* ------------------------------------------------------------- allocation */
function renderAlloc() {
  const rows = new Map();
  S.targets.forEach((r) => rows.set(r.ticker, { tk: r.ticker, tgt: r.weight, cur: 0 }));
  S.account.positions.forEach((p) => {
    const row = rows.get(p.ticker) || { tk: p.ticker, tgt: 0, cur: 0 };
    row.cur = p.weight || 0;
    rows.set(p.ticker, row);
  });
  const list = [...rows.values()].sort((a, b) => b.tgt - a.tgt || b.cur - a.cur);
  const eq = S.account.market_value + Math.max(S.account.cash, 0);
  const cashNow = eq > 0 ? Math.max(S.account.cash, 0) / eq : 0;
  if (S.cash_target > 0.005 || cashNow > 0.005) list.push({ tk: "현금", tgt: S.cash_target, cur: cashNow });
  const max = Math.max(0.1, ...list.map((r) => Math.max(r.tgt, r.cur))) * 1.08;
  $("#alloc-sub").textContent = `가로축 최대 ${Math.round(max * 100)}%`;
  mount($("#alloc-body"), list.length ? list.map((r) =>
    h("div", { class: "alloc-row", title: `현재 ${pctAbs(r.cur)} · 목표 ${pctAbs(r.tgt)}` },
      h("span", { class: "tk" }, r.tk),
      h("div", { class: "track" },
        h("div", { class: "cur", style: `width:${(r.cur / max) * 100}%` }),
        h("div", { class: "tgt", style: `left:calc(${(r.tgt / max) * 100}% - 1px)` })),
      h("span", { class: "nums" }, "현재 ", h("b", null, pctAbs(r.cur, 0)), " → 목표 ", h("b", null, pctAbs(r.tgt, 0)))))
    : h("p", { class: "muted" }, "표시할 목표 비중이 없습니다."));
}

/* ---------------------------------------------------------------- ranking */
function table(head, rows) {
  return h("div", { class: "table-wrap" }, h("table", null,
    h("thead", null, h("tr", null, head.map(([t, r]) => h("th", { class: r ? "r" : null }, t)))),
    h("tbody", null, rows)));
}
function renderRank() {
  const rows = S.rankings.map((r) => h("tr", { class: r.rank === 7 || r.rank === 12 ? "cutline" : null },
    h("td", { class: "num" }, r.rank), h("td", { class: "mono" }, r.ticker),
    h("td", { class: `r ${cls(r.mom120)}` }, pct(r.mom120, 0)), h("td", { class: "r" }, pctAbs(r.time_weight)),
    h("td", null, h("span", { class: r.semis ? "tag semi" : "tag",
      title: `상한 계산 섹터: ${r.sector}${r.semis ? ` · 반도체 ${capPct("semis_cap")} 한도에 포함` : ""}` }, r.detail || r.sector)),
    h("td", null, r.in_target ? h("span", { class: "tag in" }, "목표")
      : r.rank <= 7 ? h("span", { class: "tag", title: "기존 종목이 유지 기준을 충족해 새로 편입할 자리가 없습니다" }, "편입 대기") : null,
      r.rank === 7 ? h("span", { class: "cut-label" }, " — 신규 편입 기준") : null,
      r.rank === 12 ? h("span", { class: "cut-label" }, " — 보유 유지선") : null)));
  mount($("#rank-body"), table([["순위"], ["종목 코드"], ["120일 수익률", 1], ["TIME 비중", 1], ["세부 섹터"], [""]], rows),
    h("p", { class: "muted", style: "font-size:12px;margin:10px 0 0" },
      "매주 첫 미국 거래일에 기존 7종목 중 12위 이내인 종목을 유지하고, 빈자리는 수익률 순위가 높은 종목으로 채웁니다. 기본 비중은 TIME의 보유 비중에 비례합니다. ",
      `한 종목 ${capPct("stock_cap")}, 같은 섹터 ${capPct("sector_cap")}, 반도체(점선 표시, 미분류 포함) 합계 ${capPct("semis_cap")}를 상한으로 적용합니다. 상한을 넘는 금액은 나머지 종목에 배분하고, 배분하지 못한 금액은 현금으로 둡니다. `,
      "세부 섹터는 거인의 어깨의 대표 사업 분류이며, 상한은 이보다 큰 섹터로 계산합니다(이름에 마우스를 올리면 표시)."));
}

/* ------------------------------------------------------------------ charts */
function niceTicks(lo, hi, count = 6) {
  if (!(hi > lo)) hi = lo + 1;
  const span = hi - lo, raw = span / count, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => span / s <= count) || 10 * mag;
  const out = [];
  for (let v = Math.floor(lo / step) * step; v <= hi + step * 0.5; v += step) out.push(+v.toFixed(10));
  return out;
}

function lineChart(host, cfg) {
  host.replaceChildren();
  const W = Math.max(280, host.clientWidth);
  const compact = W < 560;  // phones: legend with latest values instead of end labels
  const endLabels = cfg.endLabels && !compact;
  const H = compact ? Math.min(cfg.height || 300, 240) : cfg.height || 300;
  const m = { l: compact ? 48 : 58, r: endLabels ? 132 : 14, t: endLabels ? 24 : 12, b: 28 };
  const n = cfg.dates.length;
  const xs = (i) => m.l + (n <= 1 ? 0 : (i * (W - m.l - m.r)) / (n - 1));
  let lo = Infinity, hi = -Infinity;
  for (const s of cfg.series) for (const v of s.values) if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  for (const mk of cfg.markers || []) if (mk.value != null) { lo = Math.min(lo, mk.value); hi = Math.max(hi, mk.value); }
  if (!isFinite(lo)) { mount(host, h("p", { class: "muted" }, cfg.empty || "데이터가 없습니다")); return; }
  if (compact && cfg.series.filter((s) => !s.context).length > 1) {
    host.appendChild(h("div", { class: "legend chart-legend" }, cfg.series.filter((s) => !s.context).map((s) => {
      const last = [...s.values].reverse().find((v) => v != null);
      return h("span", null, swatch(s.color), `${s.label} ${last == null ? "" : cfg.fmtTip(last)}`);
    })));
  }
  const padv = (hi - lo) * 0.06 || Math.abs(hi) * 0.05 || 1;
  const ticks = niceTicks(lo - padv, hi + padv);
  const t0 = ticks[0], t1 = ticks[ticks.length - 1];
  const ys = (v) => m.t + (H - m.t - m.b) * (1 - (v - t0) / (t1 - t0));
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": cfg.aria || "" }, host);
  const grid = svg("g", { class: "grid" }, root);
  const axis = svg("g", { class: "axis" }, root);
  const text = (parent, attrs, value) => { const t = svg("text", attrs, parent); t.textContent = value; return t; };
  ticks.forEach((v) => {
    svg("line", { x1: m.l, x2: W - m.r, y1: ys(v), y2: ys(v) }, grid);
    text(axis, { x: m.l - 8, y: ys(v) + 4, "text-anchor": "end" }, cfg.fmtY(v));
  });
  if (n <= 45) {  // short spans: day labels spaced at least ~64px apart
    const every = Math.max(1, Math.ceil(64 / ((W - m.l - m.r) / Math.max(1, n - 1))));
    cfg.dates.forEach((d, i) => {
      if (i % every === 0 || i === n - 1)
        text(axis, { x: xs(i), y: H - 8, "text-anchor": i === 0 ? "start" : i === n - 1 ? "end" : "middle" }, md(d));
    });
  } else {
    let lastMonth = "";
    cfg.dates.forEach((d, i) => {
      const mo = d.slice(0, 7);
      if (mo === lastMonth) return;
      lastMonth = mo;
      if (i === 0 || (xs(i) > m.l + 22 && xs(i) < W - m.r - 10))
        text(axis, { x: xs(i), y: H - 8, "text-anchor": i === 0 ? "start" : "middle" }, `${+d.slice(5, 7)}월`);
    });
  }
  if (cfg.baseline != null && cfg.baseline >= t0 && cfg.baseline <= t1)
    svg("line", { class: "baseline", x1: m.l, x2: W - m.r, y1: ys(cfg.baseline), y2: ys(cfg.baseline) }, root);
  const ordered = [...cfg.series].sort((a, b) => (a.context ? 0 : 1) - (b.context ? 0 : 1));
  for (const s of ordered) {
    let d = "", pen = false;
    s.values.forEach((v, i) => {
      if (v == null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`;
      pen = true;
    });
    svg("path", { d, fill: "none", style: `stroke:${s.color};stroke-width:${s.width || 2};stroke-linejoin:round;stroke-linecap:round` }, root);
  }
  const main = cfg.series.filter((s) => !s.context);
  if (endLabels) {
    const labels = main.map((s) => {
      let i = s.values.length - 1;
      while (i > 0 && s.values[i] == null) i--;
      return { s, v: s.values[i], x: xs(i), y: ys(s.values[i]) };
    }).sort((a, b) => a.y - b.y);
    for (const L of labels) L.y = Math.max(m.t + 6, Math.min(H - m.b - 18, L.y));
    for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 30) labels[i].y = labels[i - 1].y + 30;
    for (const L of labels) {
      svg("circle", { cx: L.x, cy: ys(L.v), r: 3.5, style: `fill:${L.s.color};stroke:var(--surface);stroke-width:2` }, root);
      text(root, { class: "endlabel", x: W - m.r + 12, y: L.y - 2, style: "fill:var(--ink)" }, L.s.label);
      text(root, { class: "endlabel", x: W - m.r + 12, y: L.y + 13, style: "fill:var(--ink-2)" }, cfg.fmtTip(L.v));
    }
  }
  const idx = new Map(cfg.dates.map((d, i) => [d, i]));
  for (const mk of cfg.markers || []) {
    const i = idx.get(mk.date);
    if (i == null || mk.value == null) continue;
    const x = xs(i), y = ys(mk.value), s = 7;
    const pts = mk.shape === "up"
      ? `${x},${y - s} ${x - s},${y + s * 0.8} ${x + s},${y + s * 0.8}`
      : `${x},${y + s} ${x - s},${y - s * 0.8} ${x + s},${y - s * 0.8}`;
    svg("polygon", { points: pts, style: `fill:${mk.fill};stroke:${mk.stroke};stroke-width:1.8;stroke-linejoin:round` }, root);
  }
  const cross = svg("line", { class: "cross", y1: m.t, y2: H - m.b, style: "opacity:0" }, root);
  const dots = main.map((s) => svg("circle", { r: 4, style: `fill:${s.color};stroke:var(--surface);stroke-width:2;opacity:0` }, root));
  const tip = h("div", { class: "tooltip", role: "presentation" });
  host.appendChild(tip);
  host.tabIndex = 0;
  let cur = n - 1;
  const show = (i) => {
    cur = Math.max(0, Math.min(n - 1, i));
    const x = xs(cur);
    cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.style.opacity = 1;
    main.forEach((s, k) => {
      const v = s.values[cur];
      dots[k].style.opacity = v == null ? 0 : 1;
      if (v != null) { dots[k].setAttribute("cx", x); dots[k].setAttribute("cy", ys(v)); }
    });
    const marks = (cfg.markers || []).filter((mk) => mk.date === cfg.dates[cur]);
    mount(tip, h("div", { class: "d" }, cfg.dates[cur]),
      main.map((s) => h("div", { class: "row" }, h("span", null, swatch(s.color), s.label),
        h("b", null, s.values[cur] == null ? "—" : cfg.fmtTip(s.values[cur])))),
      marks.map((mk) => h("div", { class: "row" }, h("span", null, mk.label))));
    tip.style.opacity = 1;
    const tw = tip.offsetWidth;
    tip.style.left = Math.min(W - tw - 4, Math.max(0, x > W * 0.6 ? x - tw - 14 : x + 14)) + "px";
    tip.style.top = m.t + 6 + "px";
  };
  const hide = () => { cross.style.opacity = 0; dots.forEach((d) => (d.style.opacity = 0)); tip.style.opacity = 0; };
  const overlay = svg("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, root);
  overlay.addEventListener("pointermove", (e) => {
    const r = root.getBoundingClientRect();
    const px = ((e.clientX - r.left) * W) / r.width;
    show(Math.round(((px - m.l) / (W - m.l - m.r)) * (n - 1)));
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

function tableView(host, dates, cols) {
  const rows = dates.map((d, i) => h("tr", null, h("td", { class: "mono" }, d),
    cols.map((c) => h("td", { class: "r" }, c.values[i] == null ? "—" : c.fmt(c.values[i]))))).reverse();
  mount(host, h("div", { class: "scroll" }, table([["날짜"], ...cols.map((c) => [c.label, 1])], rows)));
}

function renderModel() {
  const ser = S.model.series;
  const series = [
    { label: "7종목 전략(모의)", values: ser.model, color: "var(--s-strategy)" },
    { label: "TIME ETF", values: ser.etf, color: "var(--s-etf)" },
  ];
  $("#model-sub").textContent = `${md(S.model.start)}부터 누적 수익률 · 원화·종가 기준 · ETF는 해당 미국장 이후의 한국 종가로 비교`;
  const host = $("#model-chart");
  if (tableMode.model) {
    tableView(host, S.model.dates, series.map((s) => ({ label: s.label, values: s.values, fmt: (v) => pct(v / 100 - 1) })));
    return;
  }
  lineChart(host, { dates: S.model.dates, series, fmtY: (v) => pct(v / 100 - 1, 0), fmtTip: (v) => pct(v / 100 - 1),
    endLabels: true, baseline: 100, height: 300, aria: "7종목 전략 모의 성과와 TIME ETF 누적 수익률" });
}

function renderPrice() {
  const tks = Object.keys(S.series);
  if (!priceTicker || !S.series[priceTicker]) priceTicker = (S.targets[0] && S.targets[0].ticker) || tks[0];
  mount($("#price-tickers"), tks.map((t) => h("button", { "aria-pressed": String(t === priceTicker), onclick: () => { priceTicker = t; renderPrice(); } }, t)));
  const tri = (fill, stroke) => { const s = svg("svg", { width: 12, height: 10 }); svg("polygon", { points: "6,1 1,9 11,9", style: `fill:${fill};stroke:${stroke};stroke-width:1.6` }, s); return s; };
  mount($("#price-legend"),
    h("span", null, tri("var(--s-account)", "var(--s-account)"), "내 매수(▲)·매도(▼)"),
    h("span", null, tri("var(--surface)", "var(--s-strategy)"), "7종목 전략 편입·편출"));
  const s = S.series[priceTicker];
  if (!s) { mount($("#price-chart"), h("p", { class: "muted" }, "표시할 종목이 없습니다.")); return; }
  $("#price-title").textContent = `${priceTicker} 차트`;
  const markers = s.marks.map((mk) => {
    const me = mk.kind.startsWith("me_"), buy = mk.kind.endsWith("buy");
    return { date: mk.date, value: mk.price, shape: buy ? "up" : "down",
             fill: me ? "var(--s-account)" : "var(--surface)", stroke: me ? "var(--s-account)" : "var(--s-strategy)",
             label: `${me ? "내" : "전략(모의)"} ${buy ? "매수" : "매도"}${mk.qty ? ` ${qtyFmt(mk.qty)}주` : ""} @ ${usd(mk.price)}` };
  });
  lineChart($("#price-chart"), { dates: s.dates, series: [{ label: priceTicker, values: s.close, color: "var(--ink-2)", width: 1.8 }],
    markers, fmtY: (v) => usd(v, v < 20 ? 2 : 0), fmtTip: (v) => usd(v), height: 280, aria: `${priceTicker} 종가와 매매 시점` });
}

function renderPositions() {
  const a = S.account;
  $("#pos-sub").textContent = a.positions.length ? `${a.positions.length}종목 · 평균단가 기준` : "";
  if (!a.positions.length && !a.trips.length) {
    mount($("#pos-body"), h("p", { class: "muted" }, "아직 보유 기록이 없습니다. 체결 후 ‘매매 기록’에 적으면 여기서 손익과 보유 기간을 보여 줍니다."));
    return;
  }
  const rows = a.positions.map((p) => h("tr", null,
    h("td", { class: "mono" }, p.ticker), h("td", { class: "r" }, qtyFmt(p.qty)), h("td", { class: "r" }, usd(p.avg_cost)),
    h("td", { class: "r" }, usd(p.price)), h("td", { class: `r ${cls(p.pnl)}` }, arw(p.pnl) + pct(p.pnl_pct)),
    h("td", { class: "r" }, p.weight == null ? "—" : pctAbs(p.weight, 0)), h("td", { class: "r" }, `${md(p.opened)} · ${daysSince(p.opened)}일`)));
  const trips = a.trips.length ? [h("h3", { class: "eyebrow", style: "margin:16px 0 6px" }, "전량 매도한 종목"),
    h("table", null, h("tbody", null, a.trips.slice(0, 8).map((t) => h("tr", null,
      h("td", { class: "mono" }, t.ticker), h("td", null, `${md(t.opened)} → ${md(t.closed)}`),
      h("td", { class: `r ${cls(t.gain)}` }, arw(t.gain) + usd(t.gain))))))] : null;
  mount($("#pos-body"),
    table([["종목 코드"], ["수량", 1], ["평균단가", 1], ["현재가", 1], ["손익", 1], ["비중", 1], ["보유 시작일", 1]], rows), trips);
}

function renderAccount() {
  const hs = S.account.history, host = $("#account-chart");
  if (!hs.dates.length) { mount(host, h("p", { class: "muted" }, "입금이나 매수를 기록하면 계좌 평가액의 변화를 보여 줍니다.")); return; }
  if (tableMode.account) {
    tableView(host, hs.dates, [{ label: "평가액(원)", values: hs.krw, fmt: krw }, { label: "평가액($)", values: hs.usd, fmt: (v) => usd(v) }]);
    return;
  }
  lineChart(host, { dates: hs.dates, series: [{ label: "내 계좌", values: hs.krw, color: "var(--s-account)" }],
    fmtY: krwShort, fmtTip: krw, height: 240, aria: "내 계좌 원화 평가액" });
}

/* ---------------------------------------------------------------- journal */
const TYPE_KO = { buy: "매수", sell: "매도", deposit: "입금", withdraw: "출금" };
function syncForm() {
  const f = $("#jform"), cash = f.type.value === "deposit" || f.type.value === "withdraw";
  $$(".f-trade", f).forEach((n) => (n.hidden = cash));
  $(".f-cash", f).hidden = !cash;
}
function renderJournal() {
  const tickers = new Set([...S.targets.map((r) => r.ticker), ...S.rankings.map((r) => r.ticker), ...S.account.positions.map((p) => p.ticker)]);
  mount($("#tickers"), [...tickers].map((t) => h("option", { value: t })));
  const f = $("#jform");
  if (!f.date.value) f.date.value = S.session.date;
  if (!S.journal.length) {
    mount($("#journal-body"), h("p", { class: "muted" }, "기록이 없습니다. 먼저 증권 계좌의 달러 예수금을 ‘입금’으로 적어 두면 주문 수량과 손익이 정확해집니다."));
    return;
  }
  const rows = S.journal.map((e) => {
    const trade = e.type === "buy" || e.type === "sell";
    const del = h("button", { class: "linkbtn" }, "삭제");
    del.addEventListener("click", async () => {
      if (del.dataset.armed !== "1") {
        del.dataset.armed = "1"; del.textContent = "삭제하려면 다시 클릭"; del.classList.add("down");
        setTimeout(() => { del.dataset.armed = ""; del.textContent = "삭제"; del.classList.remove("down"); }, 3500);
        return;
      }
      try { const r = await api(`/api/trades/${encodeURIComponent(e.id)}`, { method: "DELETE" }); S = r.state; render(); toast("기록을 지웠습니다"); }
      catch (err) { toast(err.message); }
    });
    return h("tr", null, h("td", { class: "mono" }, e.date),
      h("td", null, h("span", { class: `tag ${e.type === "buy" ? "buy" : e.type === "sell" ? "sell" : ""}` }, TYPE_KO[e.type])),
      h("td", { class: "mono" }, e.ticker || ""), h("td", { class: "r" }, trade ? qtyFmt(e.qty) : ""),
      h("td", { class: "r" }, trade ? usd(e.price) : ""), h("td", { class: "r" }, usd(trade ? e.qty * e.price : e.amount)),
      h("td", { class: "muted" }, e.note || ""), h("td", { class: "r" }, del));
  });
  mount($("#journal-body"), table([["날짜"], ["종류"], ["종목 코드"], ["수량", 1], ["단가", 1], ["금액", 1], ["메모"], [""]], rows));
}

/* ------------------------------------------------------------------- risk */
function renderRisk() {
  const LV = { warning: ["!", "주의"], serious: ["!!", "경고"], critical: ["×", "위험"] };
  mount($("#risk-body"), S.risk.length ? S.risk.map((a) => {
    const [ico, word] = LV[a.level] || LV.warning;
    return h("div", { class: `alert ${a.level}` }, h("span", { class: "ico", "aria-hidden": "true" }, ico),
      h("div", null, h("div", { class: "t" }, `${word} · ${a.title}`), h("div", { class: "b" }, a.body)));
  }) : h("p", { class: "muted" }, "현재 주의·경고 기준에 해당하는 항목이 없습니다."),
  (S.risk_rules || []).length ? h("div", { class: "rules" }, h("div", { class: "t" }, "매매 지속·투자 금액 점검 기준 · 2026-09-27 확정"),
    h("ol", null, S.risk_rules.map((r) => h("li", null, r)))) : null);
}

function renderNotes() {
  const st = S.model.stats;
  mount($("#notes"),
    h("b", null, "운용 규칙"), " — TIME이 보유한 미국 종목의 최근 120거래일 수익률을 기준으로 7종목을 구성합니다. 기존 종목은 12위 이내면 유지하고, 빈자리를 높은 순위부터 채웁니다. 매주 첫 미국 거래일 개장가를 기준으로 조정하며, 기본 비중은 TIME 보유 비중에 비례합니다. ",
    `비중 상한은 한 종목 ${capPct("stock_cap")}, 같은 섹터 ${capPct("sector_cap")}, 반도체(미분류 포함) 합계 ${capPct("semis_cap")}입니다(반도체는 2026-10-10에 60%에서 낮춤). 배분하고 남은 금액은 현금으로 둡니다. 판단에는 해당 거래일 이전의 PDF와 전일 종가만 사용합니다.`, h("br"),
    h("b", null, `전략 모의 성과 (${md(S.model.start)} 이후, 원화, 수수료 0원, 개장가 체결 가정)`),
    ` — 7종목 ${pct(st.total)} (최대 하락폭 ${pct(st.mdd)}), TIME ETF ${pct(S.model.etf_total)}. `,
    "과거 성과가 좋았던 규칙을 801개 중에서 골랐고(#774·반도체 60%), 반도체 50%는 2026-10-10에 변형 20개를 더 비교해 골랐으므로 결과가 유리하게 보일 수 있습니다. 2023~2025년 수익률은 #774 기준 TIME ETF의 0.9~1.1배였습니다(50%는 다시 재지 않음). 미래 수익을 보장하지 않습니다.", h("br"),
    h("b", null, "자료 출처"), " — 검증한 TIME 운용사 PDF와 Yahoo Finance 일별·1분 시세를 사용합니다. 장전·장후 시세를 포함하며 지연될 수 있습니다. 주문은 증권사에서 직접 실행하세요. ",
    `화면 생성 ${S.generated_at} (한국시간)`);
}

/* ------------------------------------------------------------------ wiring */
function wire() {
  const bar = $(".topbar");
  new ResizeObserver(() => document.documentElement.style.setProperty("--bar-height", `${bar.offsetHeight}px`)).observe(bar);
  document.addEventListener("click", (e) => {
    const button = e.target.closest("[data-jump]");
    if (!button) return;
    jumpTo(button.dataset.jump);
  });
  $("#btn-refresh").addEventListener("click", () => load("refresh"));
  const up = $("#btn-update");
  up.addEventListener("click", async () => {
    if (up.dataset.armed !== "1") {
      up.dataset.armed = "1"; up.textContent = "갱신하려면 다시 클릭";
      setTimeout(() => { up.dataset.armed = ""; up.textContent = "펀드 자료 갱신"; }, 4000);
      return;
    }
    up.disabled = true; up.textContent = "자료 수집 중… (1~3분)";
    try {
      const r = await post("/api/update-data");
      S = r.state; render();
      toast(r.briefing.ok ? "펀드 보유 내역과 브리핑을 갱신했습니다" : "브리핑을 갱신하지 못했습니다. " + (r.briefing.log || []).slice(-1)[0]);
    } catch (e) { toast(e.message); }
    finally { up.disabled = false; up.dataset.armed = ""; up.textContent = "펀드 자료 갱신"; }
  });
  $("#btn-theme").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("desk-theme", next); } catch (_) { /* storage unavailable */ }
    if (S) { renderModel(); renderPrice(); renderAccount(); }
  });
  $("#model-table").addEventListener("click", (e) => { tableMode.model = !tableMode.model; e.target.textContent = tableMode.model ? "차트로 보기" : "표로 보기"; renderModel(); });
  $("#acct-table").addEventListener("click", (e) => { tableMode.account = !tableMode.account; e.target.textContent = tableMode.account ? "차트로 보기" : "표로 보기"; renderAccount(); });
  const f = $("#jform");
  f.type.addEventListener("change", syncForm);
  f.addEventListener("submit", async (e) => {
    e.preventDefault();
    $("#jerr").textContent = "";
    const body = Object.fromEntries(new FormData(f).entries());
    try {
      const r = await post("/api/trades", body);
      S = r.state; render();
      ["ticker", "qty", "price", "amount", "note"].forEach((k) => (f[k].value = ""));
      toast(`${TYPE_KO[r.entry.type]} 기록을 추가했습니다`);
    } catch (err) { $("#jerr").textContent = err.message; }
  });
  let rt;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { if (S) { renderModel(); renderPrice(); renderAccount(); } }, 150); });
  setInterval(() => { if (!document.hidden && !offline) load(); }, 120000);
}

/* ------------------------------------------------------------------ tabs */
/* Every .tab in the top bar names its hash (href) and its panel (aria-controls); a hash no tab
   claims shows the first tab, the desk. A tab's script loads its data on "desk:tab", which first
   fires on DOMContentLoaded — after every script at the end of <body> has subscribed. */
function showTab() {
  const tabs = $$(".tabs .tab");
  const cur = tabs.find((a) => a.getAttribute("href") === location.hash) || tabs[0];
  const name = cur.getAttribute("href").slice(1);
  const previous = document.body.dataset.tab;
  if (previous && previous !== name) tabScroll.set(previous, scrollY);
  for (const a of tabs) {
    a.setAttribute("aria-selected", String(a === cur));
    const panel = document.getElementById(a.getAttribute("aria-controls"));
    if (panel) panel.hidden = a !== cur;
  }
  document.body.dataset.tab = name;
  $("#brand-sub").textContent = cur.dataset.sub || cur.textContent;
  document.title = name === "desk" ? "야간 데스크" : `${cur.textContent} · 야간 데스크`;
  if (name === "desk" && S) { renderModel(); renderPrice(); renderAccount(); }  // charts were sized while hidden
  window.dispatchEvent(new CustomEvent("desk:tab", { detail: name }));
  if (previous !== name) scrollTo({ top: tabScroll.get(name) || 0, behavior: "instant" });
}

try { const t = localStorage.getItem("desk-theme"); if (t) document.documentElement.dataset.theme = t; } catch (_) { /* storage unavailable */ }
{ const q = new URLSearchParams(location.search).get("theme"); if (q === "light" || q === "dark") document.documentElement.dataset.theme = q; }
wire();
syncForm();
load();
window.addEventListener("hashchange", showTab);
document.addEventListener("DOMContentLoaded", showTab);
