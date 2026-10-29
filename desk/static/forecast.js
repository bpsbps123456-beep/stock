"use strict";
/* 예측 점검(§14): a locked forecast scored against later closes. Read-only; never touches orders. */
const FC = { data: null, id: null, request: 0 };
const fcTime = s => s ? new Date(s).toLocaleString("ko-KR", {timeZone: "Asia/Seoul", hour12: false}) + " KST" : "—";
const FC_STATE = {pending: ["대기", ""], on_track: ["지금은 맞는 쪽", "up"], off_track: ["지금은 틀린 쪽", "down"],
  hit: ["적중", "up"], miss: ["빗나감", "down"], manual: ["직접 확인", ""]};
const FC_COLORS = ["var(--muted)", "var(--accent)", "var(--s-account)", "var(--s-etf)", "var(--s-strategy)"];
const FC_SERIES = [["picks", "var(--accent)", 2.6], ["all43", "var(--muted)", 1.4], ["july10", "var(--s-etf)", 1.6],
  ["beta10", "var(--s-account)", 1.6]];

async function fcLoad(refresh = false) {
  const seq = ++FC.request;
  const status = $("#fc-status");
  if (status) status.textContent = refresh ? "가격·전망치 다시 받는 중… (10~30초)" : "예측 원장 읽는 중…";
  try {
    const data = await api("/api/forecast" + (refresh ? "?refresh=1" : ""));
    if (seq !== FC.request) return;
    FC.data = data;
    if (!FC.id || !data.forecasts.some(f => f.id === FC.id)) FC.id = data.forecasts.at(-1)?.id;  // 가장 최근 기록부터
    fcRender();
  } catch (error) {
    if (seq !== FC.request) return;
    const box = h("div", {class: "fc-error", role: "alert"}, error.message,
      h("button", {class: "btn", onclick: () => fcLoad()}, "다시 불러오기"));
    if (FC.data) { fcRender(); $("#fc-status").append(box); } else mount($("#forecast-view"), box);
  }
}

function fcValue(c) {
  const v = c.observed;
  if (v == null) return "—";
  if (c.kind === "benchmark_min_close") return usd(v);
  if (c.kind === "basket_count_beat") return `${v}개`;
  if (c.kind === "factor_rho") return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
  return pct(v, 1);
}

function fcTarget(c) {
  if (c.kind === "manual") return c.how || "";
  const t = c.threshold ?? c.value;
  const v = c.kind === "benchmark_min_close" ? usd(t) : c.kind === "basket_count_beat" ? `${t}개`
    : c.kind === "factor_rho" ? t.toFixed(2) : pct(t, 1);
  return [`기준 ${c.op} ${v}`, c.note && h("small", {}, c.note)];
}

function fcChip(state) {
  const [label, tone] = FC_STATE[state] || [state, ""];
  return h("span", {class: `fc-chip ${tone} s-${state}`}, label);
}

function fcClaimRows(claims) {
  return claims.map(c => h("tr", {},
    h("td", {}, c.text), h("td", {}, Math.round(c.probability * 100) + "%"),
    h("td", {class: c.kind === "factor_rho" || c.kind.endsWith("excess") || c.kind.endsWith("return") ? cls(c.observed) : ""}, fcValue(c)),
    h("td", {class: "fc-dim"}, fcTarget(c)), h("td", {}, fcChip(c.state))));
}

function fcTile(label, value, foot, tone = "") {
  return h("div", {class: "fc-tile"}, h("div", {class: "eyebrow"}, label),
    h("strong", {class: tone}, value), h("p", {class: "fc-dim"}, foot));
}

function fcPickers(data, fc) {
  return data.forecasts.length > 1 && h("nav", {class: "fc-pickers", "aria-label": "예측 기록 선택"},
    data.forecasts.map(f => h("button", {class: "btn" + (f.id === fc.id ? " primary" : ""), "aria-pressed": String(f.id === fc.id),
      onclick: () => { FC.id = f.id; fcRender(); }}, f.type === "earnings" ? `📊 ${f.title}` : f.title)));
}

function fcRender() {
  const root = $("#forecast-view"), data = FC.data;
  const fc = data.forecasts.find(f => f.id === FC.id);
  if (!fc) {
    mount(root, h("section", {class: "page-heading"}, h("div", {}, h("div", {class: "eyebrow"}, "잠긴 예측과 실제 가격 비교"),
      h("h1", {}, "예측 점검"), h("p", {}, "기록된 예측이 없습니다. ‘예측 기록’ 요청으로 만들 수 있습니다."))));
    return;
  }
  if (fc.type === "earnings") return fcRenderEarnings(fc, data);
  const headKey = fc.headline_basket || fc.primary_basket, many = fc.picks.length > 20;
  const r = fc.result, B = r.baskets || {}, P = B[headKey] || {};
  const soxx = r.returns?.[fc.benchmark];
  const byTicker = {};
  for (const c of r.claims) if (c.ticker) (byTicker[c.ticker] ||= []).push(c);
  const resolved = r.claims.filter(c => c.state === "hit" || c.state === "miss");
  const hits = resolved.filter(c => c.state === "hit").length;
  const progress = r.sessions ? Math.min(1, r.elapsed / r.sessions) : 0;
  const waiting = r.status === "기준 종가 대기";
  const label = fc.baskets[headKey].label;

  const head = h("section", {class: "page-heading"},
    h("div", {}, h("div", {class: "eyebrow"}, "잠긴 예측과 실제 가격 비교 · ", fc.author),
      h("h1", {}, fc.title),
      h("p", {}, `기록 ${fcTime(fc.recorded_at)} · 기준 ${md(r.baseline)} 종가 → 마감 ${md(r.target)} 종가 · `,
        h("b", {}, waiting ? "기준 종가 대기" : `${r.elapsed}/${r.sessions}거래일`), r.complete ? " · 완료" : "")),
    h("div", {class: "fc-tools"}, h("button", {class: "btn", onclick: () => fcLoad(true)}, "가격·전망치 갱신")));
  const pickers = fcPickers(data, fc);
  const bar = h("div", {class: "fc-progress", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": r.sessions,
    "aria-valuenow": r.elapsed}, h("i", {style: `width:${(progress * 100).toFixed(1)}%`}));
  const lock = h("p", {class: "fc-lock"}, "🔒 기록 뒤 수정 불가 — 내용 해시 ", h("code", {title: fc.sha256}, fc.sha256.slice(0, 12)),
    " · 원장을 고치면 검사가 실패합니다. 기준가는 기록 뒤 첫 미국 정규장 종가라 기록 중에 본 가격이 점수에 섞이지 않습니다.");

  const tiles = h("section", {class: "fc-tiles"},
    fcTile(`${fc.benchmark} 수익`, waiting ? "—" : pct(soxx), `기준 ${md(r.baseline)} 이후`, cls(soxx)),
    fcTile(label, waiting ? "—" : pct(P.return), `베타 ${fc.baskets[headKey].beta.toFixed(2)}배 · 같은 금액`, cls(P.return)),
    fcTile(`${fc.benchmark} 대비`, waiting ? "—" : pct(P.excess), `${label} − ${fc.benchmark}`, cls(P.excess)),
    fcTile("베타 조정 초과", waiting ? "—" : pct(P.alpha), `바스켓 − 베타 × ${fc.benchmark} · 종목 선택 효과`, cls(P.alpha)),
    fcTile(`${fc.benchmark}를 이긴 종목`, waiting ? "—" : `${P.beat}/${fc.baskets[headKey].tickers.length}`, label),
    fcTile("판정된 예측", `${hits}/${resolved.length}`, r.brier == null ? "판정되면 확률 점수(Brier)가 나옵니다"
      : `Brier ${r.brier.toFixed(3)} · 0에 가까울수록 확률이 정확`));

  const chartHost = h("div", {class: "fc-chart"});
  const chart = h("section", {class: "card"}, h("header", {}, h("h2", {}, "기준가 이후 누적 수익"),
    h("span", {class: "sub"}, "같은 금액으로 사서 들고 있는 바스켓 · 수정 종가")), chartHost);

  const bRows = Object.entries(B).map(([k, b]) => h("tr", {class: k === headKey ? "fc-main" : ""},
    h("td", {}, b.label, fc.baskets[k]?.note ? h("small", {}, fc.baskets[k].note) : null),
    h("td", {}, b.tickers.length), h("td", {}, b.beta == null ? "—" : b.beta.toFixed(2)),
    h("td", {class: cls(b.return)}, pct(b.return)), h("td", {class: cls(b.excess)}, pct(b.excess)),
    h("td", {class: cls(b.alpha)}, pct(b.alpha)), h("td", {}, `${b.beat}/${b.tickers.length}`)));
  const baskets = h("section", {class: "card"}, h("header", {}, h("h2", {}, "바스켓 비교"),
    h("span", {class: "sub"}, "내 판단이 대조군보다 나았는지 · 베타 조정 초과 = 수익 − 바스켓 베타 × SOXX 수익")),
    h("div", {class: "fc-scroll"}, h("table", {class: "fc-table"},
      h("thead", {}, h("tr", {}, ["바스켓", "종목", "베타", "수익", "SOXX 대비", "베타 조정", "SOXX 이김"].map(s => h("th", {}, s)))),
      h("tbody", {}, waiting ? h("tr", {}, h("td", {colspan: 7, class: "fc-dim"}, "기준 종가가 나오면 채워집니다.")) : bRows))));

  const res = Object.fromEntries((r.picks || []).map(p => [p.ticker, p]));
  const pRows = fc.picks.map(p => {
    const x = res[p.ticker] || {};
    const verdict = (byTicker[p.ticker] || []).find(c => c.kind === "ticker_excess");
    return h("tr", {}, h("td", {}, p.rank), h("td", {}, h("b", {}, p.ticker), h("small", {}, p.theme)),
      h("td", {}, verdict ? Math.round(verdict.probability * 100) + "%" : "—"),
      h("td", {}, usd(x.base_close)), h("td", {}, usd(x.last_close)), h("td", {class: cls(x.return)}, pct(x.return)),
      h("td", {class: cls(x.excess)}, pct(x.excess)), h("td", {}, x.rank ? `${x.rank}/${x.of}` : "—"),
      h("td", {class: cls(x.worst)}, pct(x.worst)), h("td", {}, verdict ? fcChip(verdict.state) : "—"));
  });
  const picks = h("section", {class: "card"}, h("header", {}, h("h2", {}, many ? `예측 순위 ${fc.picks.length}종목 성적` : `${label} 성적`),
    h("span", {class: "sub"}, `순위는 이 기록의 ${Object.keys(fc.universe).length}종목 중 기준가 이후 수익 순 · 최악은 기간 중 가장 낮았던 종가`)),
    h("div", {class: "fc-scroll"}, h("table", {class: "fc-table"},
      h("thead", {}, h("tr", {}, ["#", "종목", "이길 확률", "기준 종가", "최근 종가", "수익", "SOXX 대비", "실제 순위", "최악", "판정"].map(s => h("th", {}, s)))),
      h("tbody", {}, pRows))));
  if (many) picks.querySelector(".fc-scroll").classList.add("tall");

  const cards = h("section", {class: "fc-reasons"}, h("h2", {}, "판단 이유와 확인할 것"),
    many && h("p", {class: "fc-dim"}, `카드는 예측 순위 12위까지입니다. 나머지 ${fc.picks.length - 12}종목의 근거는 위 표와 원장에 있습니다.`),
    fc.picks.slice(0, many ? 12 : undefined).map(p => h("details", {class: "card fc-pick", open: p.rank <= 3},
      h("summary", {}, h("span", {class: "fc-rank"}, p.rank), h("b", {}, p.ticker), h("span", {class: "fc-dim"}, p.theme),
        res[p.ticker]?.return != null && h("span", {class: "fc-ret " + cls(res[p.ticker].excess)}, `SOXX 대비 ${pct(res[p.ticker].excess)}`)),
      h("p", {}, p.thesis),
      h("div", {class: "fc-sides"},
        h("div", {}, h("h3", {}, "판단 근거"), h("ul", {}, p.reasons.map(s => h("li", {}, s)))),
        h("div", {}, h("h3", {}, "틀릴 수 있는 이유"), h("ul", {}, p.risks.map(s => h("li", {}, s))))),
      h("div", {class: "fc-scroll"}, h("table", {class: "fc-table fc-mini"}, h("tbody", {}, fcClaimRows(byTicker[p.ticker] || [])))))));

  const groups = [...new Set(fc.claims.map(c => c.group || "기타"))].filter(g => g !== "종목");
  const claimCards = groups.map(g => h("section", {class: "card"}, h("header", {}, h("h2", {}, `예측 — ${g}`),
    h("span", {class: "sub"}, g === "요인" ? "반도체 43종목에서 그 값과 한 달 수익의 순위 상관(ρ)" : "확률은 기록할 때 내가 붙인 값")),
    h("div", {class: "fc-scroll"}, h("table", {class: "fc-table fc-claims"},
      h("thead", {}, h("tr", {}, ["예측", "확률", "지금 값", "판정 기준", "상태"].map(s => h("th", {}, s)))),
      h("tbody", {}, fcClaimRows(r.claims.filter(c => (c.group || "기타") === g)))))));

  const why = h("section", {class: "card fc-summary"}, h("div", {class: "eyebrow"}, "예측 요지"), h("p", {}, fc.summary),
    h("details", {}, h("summary", {}, "기록 당시 상황"), h("ul", {}, (fc.context || []).map(s => h("li", {}, s)))),
    h("details", {}, h("summary", {}, "일부러 뺀 종목"), h("ul", {}, (fc.exclusions || []).map(e => h("li", {}, h("b", {}, e.ticker), " — ", e.reason)))),
    h("details", {open: true}, h("summary", {}, "이렇게 되면 내가 틀린 것"), h("ul", {}, (fc.wrong_if || []).map(s => h("li", {}, s)))));

  mount(root, pickers, head, bar, lock,
    h("div", {id: "fc-status", class: "fc-dim", role: "status"}, "가격 ", data.prices_fetched_at ? fcTime(data.prices_fetched_at) : "미수집",
      " · 전망치 ", data.estimates_fetched_at ? fcTime(data.estimates_fetched_at) : "미수집", " · 최근 종가일 ", r.latest || "—"),
    data.warning && h("p", {class: "fc-error", role: "alert"}, data.warning),
    r.price_issues?.length > 0 && h("p", {class: "fc-note", role: "status"}, `분할 반영 대기: ${r.price_issues.join(", ")} — `,
      "Yahoo가 분할 전 가격을 아직 다시 계산하지 않아 점수에서 잠시 뺐습니다. 다음 갱신에서 돌아옵니다."),
    waiting && h("p", {class: "fc-note"}, `${md(r.baseline)} 미국 정규장 종가가 나오면(한국 다음 날 새벽) 점수가 시작됩니다. 그 전까지는 모든 예측이 대기입니다.`),
    why, tiles, chart, baskets, picks, ...claimCards, cards,
    h("footer", {class: "fc-dim"}, data.method, h("ul", {}, (fc.method_notes || []).map(s => h("li", {}, s))),
      "가격: ", data.price_source || "미수집"));

  if (!waiting && r.days?.length) {
    const flagged = Object.keys(fc.baskets).filter(k => fc.baskets[k].chart);
    const plan = flagged.length ? flagged.map((k, i) => [k, FC_COLORS[i % FC_COLORS.length], k === headKey ? 2.6 : 1.6]) : FC_SERIES;
    const series = plan.filter(([k]) => B[k]).map(([k, color, width]) =>
      ({label: B[k].label, values: B[k].path.map(v => v == null ? null : v * 100), color, width}));
    series.splice(1, 0, {label: fc.benchmark, values: r.paths[fc.benchmark].map(v => v == null ? null : v * 100), color: "var(--ink-2)", width: 2});
    lineChart(chartHost, {dates: r.days, series, baseline: 0, endLabels: true, height: 300,
      fmtY: v => pct(v / 100, Math.abs(v - Math.round(v)) < 1e-9 ? 0 : 1), fmtTip: v => pct(v / 100, 1),
      aria: "기준가 이후 바스켓별 누적 수익"});
  } else mount(chartHost, h("p", {class: "fc-dim"}, "기준 종가 뒤 첫 거래일이 지나면 그려집니다."));
}

/* ---- 실적 반응 예측(§14.2): 발표 전에 잠근 반응일 종가 분포를 실제 종가·저가와 비교한다 ---- */
const FC_EV = {waiting: "반응일 종가 대기", no_price: "가격 대기", split: "분할 반영 대기"};
const fcKst = s => s ? new Date(s).toLocaleString("ko-KR", {timeZone: "Asia/Seoul", month: "numeric", day: "numeric",
  hour: "2-digit", minute: "2-digit", hour12: false}) : "—";
const fcPp = v => v == null ? "—" : `${Math.round(v * 100)}%`;

function fcRangeBar(q, actual, lim) {
  const x = v => Math.max(0, Math.min(100, (v + lim) / (2 * lim) * 100));
  const span = (a, b) => `left:${x(a).toFixed(2)}%;width:${Math.max(0.6, x(b) - x(a)).toFixed(2)}%`;
  return h("div", {class: "fc-range", role: "img", "aria-label": `80% 구간 ${pct(q["0.1"])}~${pct(q["0.9"])}, 가운데 ${pct(q["0.5"])}`
      + (actual == null ? "" : `, 실제 ${pct(actual)}`)},
    h("i", {class: "zero", style: `left:${x(0).toFixed(2)}%`}),
    h("i", {class: "q80", style: span(q["0.1"], q["0.9"])}), h("i", {class: "q50", style: span(q["0.25"], q["0.75"])}),
    h("i", {class: "mid", style: `left:${x(q["0.5"]).toFixed(2)}%`}),
    actual != null && h("b", {class: `act ${cls(actual)}`, style: `left:${x(actual).toFixed(2)}%`}));
}

function fcVerdicts(row) {
  if (row.state !== "scored") return fcChip("pending");
  const f = row.forecast;
  const dir = f.dir_hit == null ? h("span", {class: "fc-chip"}, "방향 안 고름")
    : h("span", {class: `fc-chip ${f.dir_hit ? "up s-hit" : "down s-miss"}`}, f.dir_hit ? "방향 적중" : "방향 빗나감");
  return [dir, " ", h("span", {class: `fc-chip ${f.in80 ? "up s-hit" : "down s-miss"}`}, f.in80 ? "80% 구간 안" : "80% 구간 밖")];
}

function fcScoreTiles(s, scope) {
  const f = s.forecast || {}, m = s.mechanical || {};
  const none = !s.n;
  const pinPp = v => v == null ? "—" : `${(v * 100).toFixed(2)}%p`;
  return h("section", {class: "fc-tiles"},
    fcTile(`채점된 실적${scope}`, `${s.n}건`, none ? "반응일 종가가 나오면 채워집니다" : "반응일 정규장 종가 기준"),
    fcTile("방향 적중(판단)", none || !f.calls ? "—" : `${f.hits}/${f.calls}`, "P(상승) 50%인 종목은 방향을 고르지 않은 것"),
    fcTile("Brier 판단 / 기계", none ? "—" : `${f.brier.toFixed(3)} / ${m.brier.toFixed(3)}`, "낮을수록 좋음 · 무조건 50%면 0.250"),
    fcTile("80% 구간 적중", none ? "—" : fcPp(f.in80), "80%에 가까워야 규모가 맞은 것"),
    fcTile("분위 손실 판단 / 기계", none ? "—" : `${pinPp(f.pinball)} / ${pinPp(m.pinball)}`, "분포 전체의 오차 · 낮을수록 좋음"),
    fcTile("실제 크기 ÷ 예상 규모", none || s.size_ratio == null ? "—" : s.size_ratio.toFixed(2),
      "2026 과거 평균 1.13 · 크게 넘으면 규모를 키운다"));
}

function fcRenderEarnings(fc, data) {
  const root = $("#forecast-view"), r = fc.result, rows = r.events;
  const byT = Object.fromEntries(rows.map(x => [x.ticker, x]));
  const scored = rows.filter(x => x.state === "scored");
  const lim = Math.max(0.1, ...fc.events.flatMap(e => [Math.abs(e.forecast.q["0.1"]), Math.abs(e.forecast.q["0.9"])]),
    ...scored.map(x => Math.abs(x.return))) * 1.05;
  const first = fc.events[0];
  const days = [...new Set(fc.events.map(e => e.reaction_date))].sort(), oneDay = days.length === 1;

  const head = h("section", {class: "page-heading"},
    h("div", {}, h("div", {class: "eyebrow"}, "잠긴 실적 반응 예측 · ", fc.author),
      h("h1", {}, fc.title),
      h("p", {}, `기록 ${fcTime(fc.recorded_at)} · `, oneDay ? `기준 ${md(first.base_date)} 종가 → 반응 ${md(first.reaction_date)} 종가 · `
        : `반응일 ${md(days[0])}~${md(days.at(-1))} · `,
        h("b", {}, `${scored.length}/${rows.length}건 채점`), r.status === "완료" ? " · 완료" : "")),
    h("div", {class: "fc-tools"}, h("button", {class: "btn", onclick: () => fcLoad(true)}, "가격 갱신")));
  const lock = h("p", {class: "fc-lock"}, "🔒 기록 뒤 수정 불가 — 내용 해시 ", h("code", {title: fc.sha256}, fc.sha256.slice(0, 12)),
    " · 발표 전, 그리고 기준일 정규장 마감 전에만 기록할 수 있어 기준가를 보고 쓴 예측이 아닙니다.");
  const why = h("section", {class: "card fc-summary"}, h("div", {class: "eyebrow"}, "예측 요지"), h("p", {}, fc.summary),
    h("details", {}, h("summary", {}, "기록 당시 상황"), h("ul", {}, (fc.context || []).map(s => h("li", {}, s)))),
    h("details", {open: true}, h("summary", {}, "이렇게 되면 내가 틀린 것"), h("ul", {}, (fc.wrong_if || []).map(s => h("li", {}, s)))));

  const evRows = fc.events.map(e => {
    const x = byT[e.ticker] || {}, f = e.forecast, q = f.q;
    return h("tr", {},
      h("td", {}, h("b", {}, e.ticker), h("small", {}, e.name || "", " · ", e.judged ? "판단 넣음" : "기계 기준 그대로")),
      h("td", {class: "fc-dim"}, fcKst(e.announce_at), h("small", {}, e.timing === "amc" ? "장 마감 뒤" : "장 시작 전")),
      h("td", {}, fcPp(f.p_up), h("small", {}, `기계 ${fcPp(e.mechanical.p_up)}`)),
      h("td", {class: "fc-rangecell"}, fcRangeBar(q, x.return ?? null, lim),
        h("small", {}, `${pct(q["0.1"])} ~ ${pct(q["0.9"])} · 가운데 ${pct(q["0.5"])} · 규모 ${pctAbs(e.inputs?.scale)}`)),
      h("td", {class: cls(x.return)}, x.state === "scored" ? pct(x.return) : h("span", {class: "fc-dim"}, FC_EV[x.state] || "—"),
        x.state === "scored" && h("small", {}, `${usd(x.base_close)} → ${usd(x.close)}`)),
      h("td", {class: cls(x.bench_return)}, x.state === "scored" ? pct(x.bench_return) : "—"),
      h("td", {}, fcVerdicts(x)));
  });
  const main = h("section", {class: "card"}, h("header", {}, h("h2", {}, "종목별 예측과 실제"),
    h("span", {class: "sub"}, "막대: 옅은 칸 80% 구간 · 진한 칸 50% 구간 · 세로선 가운데 예상 · 점 실제 반응일 종가 · 가운데 0선")),
    h("div", {class: "fc-scroll"}, h("table", {class: "fc-table fc-earn"},
      h("thead", {}, h("tr", {}, ["종목", "발표(한국)", "P(상승)", "예상 반응(기준 종가 대비)", "실제", fc.benchmark, "판정"].map(s => h("th", {}, s)))),
      h("tbody", {}, evRows))));

  const maxZ = Math.max(...fc.events.map(e => (e.levels || []).length));
  const zCols = Array.from({length: maxZ}, (_, i) => `Z${i + 1}`);
  const lvRows = fc.events.map(e => {
    const x = byT[e.ticker] || {}, got = Object.fromEntries((x.levels || []).map(l => [l.name, l]));
    return h("tr", {}, h("td", {}, h("b", {}, e.ticker)), zCols.map(name => {
      const l = (e.levels || []).find(v => v.name === name);
      if (!l) return h("td", {class: "fc-dim"}, "—");
      const s = got[name];
      const result = !s ? null : s.touched ? h("small", {class: cls(s.to_close)}, `닿음 → 종가 ${pct(s.to_close)}`)
        : h("small", {class: "fc-dim"}, s.touched_post ? "애프터만 닿음" : "안 닿음");
      return h("td", {}, usd(l.price), h("small", {}, `${pct(s ? s.dist : l.dist_ref)} · ${fcPp(l.p_touch)}`), result);
    }));
  });
  const levels = h("section", {class: "card"}, h("header", {}, h("h2", {}, "지지 구역 지정가 (160분봉)"),
    h("span", {class: "sub"}, "가격 = 구역 하단 · 거리(채점 전에는 기록 때 기준가, 채점 뒤에는 기준 종가 대비) · 반응일 프리장+정규장에 닿을 확률 · 결과")),
    h("div", {class: "fc-scroll"}, h("table", {class: "fc-table fc-levels"},
      h("thead", {}, h("tr", {}, ["종목", ...zCols].map(s => h("th", {}, s)))), h("tbody", {}, lvRows))),
    h("p", {class: "fc-dim"}, "Z1이 지금 가격에 가장 가까운 지지입니다. 닿음은 프리장 1시간봉(한 번 찍힌 가격은 거름)이나 정규장 저가가 지정가를 0.1% 넘게 뚫은 경우입니다. ",
      "프리·애프터는 Yahoo 1시간봉이라 한국 낮 주간거래(뉴욕 20:00~04:00) 체결은 보이지 않습니다. 이 칸은 체결 확률 예측이고 매수 추천이 아닙니다."));

  const cards = h("section", {class: "fc-reasons"}, h("h2", {}, "판단 이유와 확인할 것"),
    fc.events.map(e => {
      const i = e.inputs || {}, x = byT[e.ticker] || {};
      const facts = [i.implied_move != null && `옵션 내재 ${pctAbs(i.implied_move)}(${md(i.expiry)} 만기 ATM, 사건 몫 ${pctAbs(i.implied_move_event)})`,
        i.s_hist != null && `최근 ${(i.hist || []).length}번 반응 평균 ${pctAbs(i.s_hist)}`, `규모 ${pctAbs(i.scale)}`,
        `지지 구역 ${i.zones_asof || "—"} 기준 · 기준가 ${usd(i.spot_ref)}(${md(i.spot_ref_date)})`].filter(Boolean).join(" · ");
      return h("details", {class: "card fc-pick", open: e.judged},
        h("summary", {}, h("b", {}, e.ticker), h("span", {class: "fc-dim"}, e.name || ""),
          h("span", {class: "fc-dim"}, `P(상승) ${fcPp(e.forecast.p_up)} · 80% ${pct(e.forecast.q["0.1"])}~${pct(e.forecast.q["0.9"])}`),
          x.state === "scored" && h("span", {class: "fc-ret " + cls(x.return)}, `실제 ${pct(x.return)}`)),
        h("p", {}, e.thesis), h("p", {class: "fc-dim"}, facts),
        h("div", {class: "fc-sides"},
          h("div", {}, h("h3", {}, "판단 근거"), h("ul", {}, e.reasons.map(s => h("li", {}, s)))),
          h("div", {}, h("h3", {}, "틀릴 수 있는 이유"), h("ul", {}, e.risks.map(s => h("li", {}, s))))));
    }));

  const hist = (data.earnings || {}).events || [];
  const cum = h("section", {class: "card"}, h("header", {}, h("h2", {}, "누적 성적표 — 고도화 근거"),
    h("span", {class: "sub"}, "모든 실적 반응 기록의 채점된 사건 · 판단이 기계 기준보다 나은지, 규모가 맞는지 본다")),
    fcScoreTiles((data.earnings || {}).score || {n: 0}, "(누적)"),
    hist.length ? h("div", {class: "fc-scroll tall"}, h("table", {class: "fc-table"},
      h("thead", {}, h("tr", {}, ["반응일", "종목", "P(상승)", "80% 구간", "실제", "방향", "구간"].map(s => h("th", {}, s)))),
      h("tbody", {}, hist.map(x => h("tr", {}, h("td", {}, md(x.reaction_date)), h("td", {}, h("b", {}, x.ticker)),
        h("td", {}, fcPp(x.forecast.p_up)), h("td", {class: "fc-dim"}, `${pct(x.forecast.q["0.1"])} ~ ${pct(x.forecast.q["0.9"])}`),
        h("td", {class: cls(x.return)}, pct(x.return)),
        h("td", {}, x.forecast.dir_hit == null ? "안 고름" : x.forecast.dir_hit ? "적중" : "빗나감"),
        h("td", {}, x.forecast.in80 ? "안" : "밖"))))))
      : h("p", {class: "fc-dim"}, "아직 채점된 사건이 없습니다. 첫 반응일 종가(10/30, 한국 10/31 05:00) 뒤 20분이 지나면 채워집니다."));

  mount(root, fcPickers(data, fc), head, lock,
    h("div", {id: "fc-status", class: "fc-dim", role: "status"}, "가격 ", data.prices_fetched_at ? fcTime(data.prices_fetched_at) : "미수집"),
    data.warning && h("p", {class: "fc-error", role: "alert"}, data.warning),
    why, fcScoreTiles(r.score, ""), main, levels, cards, cum,
    h("footer", {class: "fc-dim"},
      h("p", {}, "계산: 기계 기준은 P(상승) 50%(2026년 실적 4,701건의 반응일 상승 비율)와, 옵션 내재 변동(사건 몫)과 최근 8번 반응 평균을 반씩 섞은 규모에 ",
        "2026 반응 분포의 분위(규모 단위)를 곱한 구간입니다. 판단은 근거와 함께 P(상승)을 옮기고 분포를 그만큼 평행 이동합니다."),
      fc.model?.judgment && h("p", {}, fc.model.judgment),
      h("details", {}, h("summary", {}, "기록된 계산 설명(원문)"), h("pre", {class: "fc-model"}, fc.model?.text || "")),
      "가격: ", data.price_source || "미수집"));
}

window.addEventListener("desk:tab", e => {
  if (e.detail === "forecast") fcLoad();
});
