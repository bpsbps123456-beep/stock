"use strict";
/* 실적 뒤 흐름(§15) — renders /api/erules. 2026년에 강한 실적 후 조건부 규칙 8개(매수 K1~K5·회피 X1~X3)에 맞는 종목을
   자동으로 쌓고, 다음 날(또는 6거래일째) 정규장 시가 진입 기준 SPY 대비 성적을 채점한다. 관찰용 화면이다: 매수 추천·목표 비중·
   주문 수량이 없다. 상승 = red(up), 하락 = blue(down), 부호를 함께 써서 색만으로 읽지 않게 한다. app.js의 h()/mount/$/api/md/cls를 쓴다. */
const ER = { data: null, request: 0, onlyMatched: true, onlyAI: false };
const ER_STATE = { building: ["쌓는 중", ""], holding: ["유지", "up"], alarm: ["경보", "down"] };
const ER_SIDE = { buy: "매수 쪽", avoid: "피할 쪽" };
const ER_SRC = { sec: "SEC 접수", nasdaq: "나스닥", yf: "yfinance" };
const erPct = (v, d = 1) => (v == null ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}%`);
const erRate = (v) => (v == null ? "—" : `${Math.round(v * 100)}%`);
const erTime = (s) => (s ? new Date(s).toLocaleString("ko-KR", { timeZone: "Asia/Seoul", hour12: false }) + " KST" : "—");

async function erLoad(refresh = false) {
  const seq = ++ER.request;
  const status = $("#er-status");
  if (status) status.textContent = refresh ? "실적 달력·SEC 접수 시각·가격을 다시 받는 중… (30초~2분)" : "원장 읽는 중…";
  try {
    const data = await api("/api/erules" + (refresh ? "?refresh=1" : ""));
    if (seq !== ER.request) return;
    ER.data = data;
    erRender();
  } catch (error) {
    if (seq !== ER.request) return;
    const box = h("div", { class: "er-error", role: "alert" }, error.message,
      h("button", { class: "btn", onclick: () => erLoad() }, "다시 불러오기"));
    if (ER.data) { erRender(); $("#er-status").append(box); } else mount($("#erules-view"), box);
  }
}

function erChip(state, n, need) {
  const [label, tone] = ER_STATE[state] || [state, ""];
  return h("span", { class: `er-chip ${tone} s-${state}` }, state === "building" ? `${label} ${n}/${need}` : label);
}

function erResearch(r) {
  const x = r.research, [q1, q2, q3] = x.seasons;
  return h("p", { class: "er-dim" },
    `연구(다음 날 시가 진입) · 2026 승률 ${erRate(x.win)}(${x.n}건, 기준 ${erRate(x.base)}) · 중앙 ${erPct(x.median)} · `,
    `1~3월 ${erRate(q1[0])} · 4~6월 ${erRate(q2[0])} · 7~9월 ${erRate(q3[0])} · 대형주만 ${erRate(x.big[0])}(${x.big[1]}건) · `,
    `2025(참고) ${erRate(x.y2025[0])}(기준 ${erRate(x.y2025[1])})`);
}

function erCard(r, need) {
  const lv = r.live;
  const verdict = lv.state === "alarm"
    ? (r.side === "buy" ? "실전에서 절반 넘게 SPY에 졌습니다 — 장세가 바뀐 신호로 보고 이 규칙을 멈춥니다."
      : "실전에서 절반 넘게 SPY를 이겼습니다 — 이 회피 규칙이 안 맞는 장세입니다.")
    : lv.state === "holding" ? "실전 성적이 연구 방향과 같습니다." : `닫힌 사례가 ${need}건 쌓이면 판정합니다.`;
  const tone = lv.win == null ? "" : (r.side === "buy" ? cls(lv.win - 0.5) : cls(0.5 - lv.win));
  return h("section", { class: `card er-rule side-${r.side} st-${lv.state}` },
    h("header", {}, h("span", { class: "er-code" }, r.code), h("h2", {}, r.label),
      h("span", { class: `er-side ${r.side}` }, ER_SIDE[r.side]), erChip(lv.state, lv.n, need)),
    h("ul", { class: "er-cond" }, r.conditions.map((c) => h("li", {}, c))),
    h("p", { class: "er-dim" }, `진입 ${r.entry} · ${r.horizon}`),
    h("div", { class: "er-live" },
      h("div", {}, h("div", { class: "eyebrow" }, "실전 승률"), h("strong", { class: tone }, erRate(lv.win)),
        h("p", { class: "er-dim" }, `닫힘 ${lv.n}건 · 같은 칸 전체 ${erRate(lv.base_win)}`)),
      h("div", {}, h("div", { class: "eyebrow" }, "중앙 초과수익"), h("strong", { class: cls(lv.median) }, erPct(lv.median)),
        h("p", { class: "er-dim" }, `평균 ${erPct(lv.mean)}`)),
      h("div", {}, h("div", { class: "eyebrow" }, "보유 중"), h("strong", {}, `${r.open.length}건`),
        h("p", { class: "er-dim" }, "기간이 끝나면 닫힘으로"))),
    h("p", { class: `er-verdict st-${lv.state}` }, verdict), erResearch(r));
}

function erFeat(row) {
  const bits = [];
  if (row.mcap_b != null) bits.push(`시총 ${row.mcap_b >= 100 ? Math.round(row.mcap_b) : row.mcap_b.toFixed(1)}B`);
  if (row.vol_ratio != null) bits.push(`거래량 ${row.vol_ratio.toFixed(1)}배`);
  if (row.eps != null) bits.push(`EPS ${erPct(row.eps, 0)}`);
  if (row.gap != null) bits.push(`갭 ${erPct(row.gap)}·장중 ${erPct(row.o2c)}`);
  if (row.d5_x != null) bits.push(`뒤 5일 ${erPct(row.d5_x)}`);
  if (row.pre60_x != null) bits.push(`전 60일 ${erPct(row.pre60_x, 0)}`);
  if (row.vol60 != null) bits.push(`하루 변동 ${row.vol60.toFixed(1)}%`);
  return bits.join(" · ");
}

function erTicker(row) {
  return [h("b", { class: "mono" }, row.ticker), row.ai && h("span", { class: "tag", title: "AI 유니버스(watchlist.AI_SUBSECTOR)" }, row.ai),
    h("small", {}, row.name)];
}

function erTable(head, rows) {
  return h("div", { class: "er-scroll" }, h("table", { class: "er-table" },
    h("thead", {}, h("tr", {}, head.map(([t, c]) => h("th", { class: c === true ? "er-feat" : c || null }, t)))),
    h("tbody", {}, rows)));
}

function erUpcoming(data) {
  const rows = data.upcoming;
  return h("section", { class: "card er-up" },
    h("header", {}, h("h2", {}, "다음 정규장 시가 진입 예정"),
      h("span", { class: "sub" }, "조건이 확정됐고 진입일 시가가 아직 오지 않은 종목입니다. 한국 시각 22:30(11/2부터 23:30) 정규장 시가 기준입니다.")),
    rows.length ? erTable([["진입일"], ["규칙", "er-rc"], ["종목", "er-tk"], ["반응일"], ["실적날 SPY 대비"], ["조건 값", true]],
      rows.map((row) => h("tr", { class: row.side === "buy" ? "er-hit" : "er-avoid" },
        h("td", {}, md(row.entry_date)),
        h("td", { class: "er-rc" }, h("span", { class: `er-tag ${row.side}` }, row.code), h("small", { class: "er-dim" }, ` ${row.label}`)),
        h("td", { class: "er-tk" }, erTicker(row)), h("td", {}, md(row.reaction)), h("td", { class: cls(row.r0_x) }, erPct(row.r0_x)),
        h("td", { class: "er-feat" }, erFeat(row)))))
      : h("p", { class: "er-dim" }, "지금 진입을 기다리는 종목이 없습니다."));
}

function erRuleTables(r) {
  const head = [["종목", "er-tk"], ["반응일"], ["진입일"], ["실적날 SPY 대비"], ["조건 값", true]];
  const body = (rows, closed) => rows.map((row) => h("tr", {},
    h("td", { class: "er-tk" }, erTicker(row)), h("td", {}, md(row.reaction)), h("td", {}, md(row.entry_date)),
    h("td", { class: cls(row.r0_x) }, erPct(row.r0_x)), h("td", { class: "er-feat" }, erFeat(row)),
    h("td", { class: cls(row.x) }, erPct(row.x)),
    h("td", {}, `${row.days ?? 0}거래일${row.capped && closed ? " (다음 실적 못 찾음)" : ""}`),
    h("td", { class: "er-dim" }, ER_SRC[row.timing] || row.timing)));
  return h("details", { class: "card er-detail", open: r.side === "buy" },
    h("summary", {}, h("b", {}, `${r.code} ${r.label}`), h("span", { class: "er-dim" }, ` — 보유 중 ${r.open.length}건 · 닫힘 ${r.closed.length}건 · ${r.entry} 진입`)),
    r.open.length ? erTable([...head, ["지금까지(SPY 대비)"], ["경과"], ["시각"]], body(r.open, false))
      : h("p", { class: "er-dim" }, "보유 중인 사례가 없습니다."),
    r.closed.length > 0 && h("details", { class: "er-closed" }, h("summary", {}, `닫힌 사례 ${r.closed.length}건`),
      erTable([...head, ["결과(SPY 대비)"], ["경과"], ["시각"]], body(r.closed, true))));
}

function erRecent(data) {
  let rows = data.recent;
  if (ER.onlyMatched) rows = rows.filter((r) => r.hits.length || r.waits.length);
  if (ER.onlyAI) rows = rows.filter((r) => r.ai);
  const seg = (label, on, set) => h("button", { role: "switch", "aria-checked": String(on), class: `er-toggle ${on ? "on" : ""}`, onclick: set }, label);
  const side = (c) => (c.startsWith("X") ? "avoid" : "buy");
  return h("section", { class: "card" },
    h("header", {}, h("h2", {}, "최근 10거래일 실적 발표"),
      h("span", { class: "sub" }, `대상(지금 시총 30억 달러 이상) ${data.recent.length}건 중 ${rows.length}건 표시`),
      h("div", { class: "right" },
        seg("규칙에 맞은 것만", ER.onlyMatched, () => { ER.onlyMatched = !ER.onlyMatched; erRender(); }),
        seg("AI 종목만", ER.onlyAI, () => { ER.onlyAI = !ER.onlyAI; erRender(); }))),
    rows.length ? h("div", { class: "er-scroll tall" }, h("table", { class: "er-table" },
      h("thead", {}, h("tr", {}, [["종목", "er-tk"], ["반응일"], ["실적날 SPY 대비"], ["규칙", "er-rc"], ["조건 값", "er-feat"]]
        .map(([t, c]) => h("th", { class: c || null }, t)))),
      h("tbody", {}, rows.map((row) => h("tr", { class: row.hits.some((c) => c.startsWith("K")) ? "er-hit" : "" },
        h("td", { class: "er-tk" }, erTicker(row)), h("td", {}, md(row.reaction)), h("td", { class: cls(row.r0_x) }, erPct(row.r0_x)),
        h("td", { class: "er-rc" }, row.hits.map((c) => h("span", { class: `er-tag ${side(c)}` }, c)),
          row.waits.map((c) => h("span", { class: "er-tag wait", title: "5일 뒤 값이나 EPS가 나오면 판정합니다" }, `${c} 대기`))),
        h("td", { class: "er-feat" }, erFeat(row))))))) :
      h("p", { class: "er-dim" }, "조건에 맞는 발표가 없습니다."));
}

function erRender() {
  const data = ER.data, root = $("#erules-view");
  if (!data) return;
  const need = data.alarm_min_n;
  const counts = data.status_counts || {};
  const buy = data.rules.filter((r) => r.side === "buy"), avoid = data.rules.filter((r) => r.side === "avoid");
  mount(root,
    h("div", { class: "page-heading" }, h("div", { class: "eyebrow" }, "실적 뒤 흐름 · 2026 엄선 규칙 자동 추적"),
      h("h1", {}, "실적 뒤 흐름"),
      h("p", {}, "2026년에 강했던 실적 후 조건부 규칙 8개(매수 쪽 K1~K5, 피할 쪽 X1~X3)에 맞는 종목을 자동으로 쌓고, ",
        "실제로 살 수 있는 다음 날(또는 6거래일째) 정규장 시가에 들어갔을 때의 SPY 대비 성적을 채점합니다. ",
        `실전 성적은 연구 구간이 끝난 ${md(data.live_start)} 반응일부터만 셉니다. 관찰용이며 매수 추천이 아닙니다.`)),
    h("div", { class: "er-tools" },
      h("button", { class: "btn", onclick: () => erLoad(true) }, "달력·가격 다시 받기"),
      h("span", { id: "er-status", class: "er-dim", role: "status" },
        `갱신 ${erTime(data.updated_at)} · 최근 확정 종가 ${data.latest_close || "—"} · 채점한 발표 ${data.events_n}건`,
        counts.waiting ? ` · 반응일 종가 대기 ${counts.waiting}건` : "",
        data.unknown_timing?.length ? ` · 발표 시각 미상으로 뺀 ${data.unknown_timing.length}건` : "")),
    data.warning && h("p", { class: "er-note", role: "status" }, data.warning),
    erUpcoming(data),
    h("h2", { class: "er-group" }, "매수 쪽 규칙"),
    h("div", { class: "er-rules" }, buy.map((r) => erCard(r, need))),
    h("h2", { class: "er-group" }, "피할 쪽 규칙"),
    h("div", { class: "er-rules" }, avoid.map((r) => erCard(r, need))),
    ...data.rules.map(erRuleTables),
    erRecent(data),
    h("details", { class: "notes-panel" }, h("summary", {}, "계산 방법과 경보 기준"),
      h("ul", {},
        h("li", {}, data.method),
        h("li", {}, `경보: 닫힌 실전 사례가 ${need}건 이상일 때 매수 쪽은 승률 50% 미만, 피할 쪽은 50% 초과면 경보입니다. 경보가 뜬 규칙은 멈춥니다.`),
        h("li", {}, "엄선 기준: 2026 전방 검증을 통과한 규칙 325개를 다음 날 시가 진입으로 다시 재서, 2026 승률이 기준보다 15%p 이상 높고(피할 쪽은 낮고) 1~3월·4~6월·7~9월 세 시즌 모두 8%p 이상, 중앙 초과수익 ±2% 이상, 대형주(사건 시점 100억 달러 이상)에서도 같은 쪽인 것만 남긴 뒤(51개) 같은 뜻끼리 묶어 대표 8개를 골랐습니다. 2025년 성적은 고르는 데 쓰지 않았고 참고로만 적습니다."),
        h("li", {}, "근거 연구: data/derived/analysis/earnings_trend/rules_report.md §7. 정본: docs/desk_system.md §15."),
        h("li", {}, "승률 60~70%는 10번 중 3~4번은 틀린다는 뜻입니다. 규칙은 후보를 거르는 체이지 확신의 근거가 아닙니다."))));
}

window.addEventListener("desk:tab", (e) => {
  if (e.detail === "erules") erLoad();
});
