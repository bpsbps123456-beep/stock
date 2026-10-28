"use strict";
/* 예측 점검(§14): a locked forecast scored against later closes. Read-only; never touches orders. */
const FC = { data: null, id: null, request: 0 };
const fcTime = s => s ? new Date(s).toLocaleString("ko-KR", {timeZone: "Asia/Seoul", hour12: false}) + " KST" : "—";
const FC_STATE = {pending: ["대기", ""], on_track: ["지금은 맞는 쪽", "up"], off_track: ["지금은 틀린 쪽", "down"],
  hit: ["적중", "up"], miss: ["빗나감", "down"], manual: ["직접 확인", ""]};
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
    if (!FC.id || !data.forecasts.some(f => f.id === FC.id)) FC.id = data.forecasts.at(-1)?.id;
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

function fcRender() {
  const root = $("#forecast-view"), data = FC.data;
  const fc = data.forecasts.find(f => f.id === FC.id);
  if (!fc) {
    mount(root, h("section", {class: "page-heading"}, h("div", {}, h("div", {class: "eyebrow"}, "잠긴 예측과 실제 가격 비교"),
      h("h1", {}, "예측 점검"), h("p", {}, "기록된 예측이 없습니다. ‘예측 기록’ 요청으로 만들 수 있습니다."))));
    return;
  }
  const r = fc.result, B = r.baskets || {}, P = B[fc.primary_basket] || {}, M = B.all43;
  const soxx = r.returns?.[fc.benchmark];
  const byTicker = {};
  for (const c of r.claims) if (c.ticker) (byTicker[c.ticker] ||= []).push(c);
  const resolved = r.claims.filter(c => c.state === "hit" || c.state === "miss");
  const hits = resolved.filter(c => c.state === "hit").length;
  const progress = r.sessions ? Math.min(1, r.elapsed / r.sessions) : 0;
  const waiting = r.status === "기준 종가 대기";

  const head = h("section", {class: "page-heading"},
    h("div", {}, h("div", {class: "eyebrow"}, "잠긴 예측과 실제 가격 비교 · ", fc.author),
      h("h1", {}, fc.title),
      h("p", {}, `기록 ${fcTime(fc.recorded_at)} · 기준 ${md(r.baseline)} 종가 → 마감 ${md(r.target)} 종가 · `,
        h("b", {}, waiting ? "기준 종가 대기" : `${r.elapsed}/${r.sessions}거래일`), r.complete ? " · 완료" : "")),
    h("div", {class: "fc-tools"},
      data.forecasts.length > 1 && h("select", {onchange: e => { FC.id = e.target.value; fcRender(); }},
        data.forecasts.map(f => h("option", {value: f.id, selected: f.id === fc.id}, f.title))),
      h("button", {class: "btn", onclick: () => fcLoad(true)}, "가격·전망치 갱신")));
  const bar = h("div", {class: "fc-progress", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": r.sessions,
    "aria-valuenow": r.elapsed}, h("i", {style: `width:${(progress * 100).toFixed(1)}%`}));
  const lock = h("p", {class: "fc-lock"}, "🔒 기록 뒤 수정 불가 — 내용 해시 ", h("code", {title: fc.sha256}, fc.sha256.slice(0, 12)),
    " · 원장을 고치면 검사가 실패합니다. 기준가는 기록 뒤 첫 미국 정규장 종가라 기록 중에 본 가격이 점수에 섞이지 않습니다.");

  const tiles = h("section", {class: "fc-tiles"},
    fcTile(`${fc.benchmark} 수익`, waiting ? "—" : pct(soxx), `기준 ${md(r.baseline)} 이후`, cls(soxx)),
    fcTile(fc.baskets[fc.primary_basket].label, waiting ? "—" : pct(P.return),
      `베타 ${fc.baskets[fc.primary_basket].beta.toFixed(2)}배 · 같은 금액`, cls(P.return)),
    fcTile(`${fc.benchmark} 대비`, waiting ? "—" : pct(P.excess), "탑픽 − SOXX", cls(P.excess)),
    fcTile("베타 조정 초과", waiting ? "—" : pct(P.alpha), "탑픽 − 베타 × SOXX · 종목 선택 효과", cls(P.alpha)),
    fcTile("SOXX를 이긴 종목", waiting ? "—" : `${P.beat}/${fc.picks.length}`, "기준가 이후 수익 기준"),
    fcTile("판정된 예측", `${hits}/${resolved.length}`, r.brier == null ? "판정되면 확률 점수(Brier)가 나옵니다"
      : `Brier ${r.brier.toFixed(3)} · 0에 가까울수록 확률이 정확`));

  const chartHost = h("div", {class: "fc-chart"});
  const chart = h("section", {class: "card"}, h("header", {}, h("h2", {}, "기준가 이후 누적 수익"),
    h("span", {class: "sub"}, "같은 금액으로 사서 들고 있는 바스켓 · 수정 종가")), chartHost);

  const bRows = Object.entries(B).map(([k, b]) => h("tr", {class: k === fc.primary_basket ? "fc-main" : ""},
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
      h("td", {}, usd(x.base_close)), h("td", {}, usd(x.last_close)), h("td", {class: cls(x.return)}, pct(x.return)),
      h("td", {class: cls(x.excess)}, pct(x.excess)), h("td", {}, x.rank ? `${x.rank}/${x.of}` : "—"),
      h("td", {class: cls(x.worst)}, pct(x.worst)), h("td", {}, verdict ? fcChip(verdict.state) : "—"));
  });
  const picks = h("section", {class: "card"}, h("header", {}, h("h2", {}, "탑픽 10 성적"),
    h("span", {class: "sub"}, "순위는 반도체 43종목 중 한 달 수익 순 · 최악은 기간 중 가장 낮았던 종가")),
    h("div", {class: "fc-scroll"}, h("table", {class: "fc-table"},
      h("thead", {}, h("tr", {}, ["#", "종목", "기준 종가", "최근 종가", "수익", "SOXX 대비", "순위", "최악", "판정"].map(s => h("th", {}, s)))),
      h("tbody", {}, pRows))));

  const cards = h("section", {class: "fc-reasons"}, h("h2", {}, "고른 이유와 확인할 것"),
    fc.picks.map(p => h("details", {class: "card fc-pick", open: p.rank <= 3},
      h("summary", {}, h("span", {class: "fc-rank"}, p.rank), h("b", {}, p.ticker), h("span", {class: "fc-dim"}, p.theme),
        res[p.ticker]?.return != null && h("span", {class: "fc-ret " + cls(res[p.ticker].excess)}, `SOXX 대비 ${pct(res[p.ticker].excess)}`)),
      h("p", {}, p.thesis),
      h("div", {class: "fc-sides"},
        h("div", {}, h("h3", {}, "근거"), h("ul", {}, p.reasons.map(s => h("li", {}, s)))),
        h("div", {}, h("h3", {}, "위험"), h("ul", {}, p.risks.map(s => h("li", {}, s))))),
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

  mount(root, head, bar, lock,
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
    const series = FC_SERIES.filter(([k]) => B[k]).map(([k, color, width]) =>
      ({label: B[k].label, values: B[k].path.map(v => v == null ? null : v * 100), color, width}));
    series.splice(1, 0, {label: fc.benchmark, values: r.paths[fc.benchmark].map(v => v == null ? null : v * 100), color: "var(--ink-2)", width: 2});
    lineChart(chartHost, {dates: r.days, series, baseline: 0, endLabels: true, height: 300,
      fmtY: v => pct(v / 100, Math.abs(v - Math.round(v)) < 1e-9 ? 0 : 1), fmtTip: v => pct(v / 100, 1),
      aria: "기준가 이후 바스켓별 누적 수익"});
  } else mount(chartHost, h("p", {class: "fc-dim"}, "기준 종가 뒤 첫 거래일이 지나면 그려집니다."));
}

window.addEventListener("desk:tab", e => {
  if (e.detail === "forecast") fcLoad();
});
