"use strict";
/* 실적 뒤 흐름(§15) — renders /api/erules. 실적 발표 뒤 조건부 규칙 B1·B4·A2에 맞는 종목을 자동으로 쌓고
   SPY 대비 성적을 채점한다. 관찰용 화면이다: 매수 추천·목표 비중·주문 수량이 없다. 상승 = red(up), 하락 = blue(down),
   부호를 함께 써서 색만으로 읽지 않게 한다. app.js의 h()/mount/$/api/md/cls를 쓴다. */
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
  const [a, b, p] = ["2026A", "2026B", "2025"].map((k) => r.research[k]);
  return h("p", { class: "er-dim" }, `연구 승률 · 2026 상반기 ${erRate(a[0])}(${a[1]}건, 기준 ${erRate(a[2])}) · `,
    `7~9월 ${erRate(b[0])}(${b[1]}건, 기준 ${erRate(b[2])}) · 2025 ${erRate(p[0])}(기준 ${erRate(p[2])})`);
}

function erCard(r, need) {
  const lv = r.live;
  const verdict = lv.state === "alarm"
    ? (r.side === "buy" ? "실전에서 절반 넘게 SPY에 졌습니다 — 장세가 바뀐 신호로 보고 이 규칙을 내립니다."
      : "실전에서 절반 넘게 SPY를 이겼습니다 — 추격 회피 규칙이 안 맞는 장세입니다.")
    : lv.state === "holding" ? "실전 성적이 연구 방향과 같습니다." : `닫힌 사례가 ${need}건 쌓이면 판정합니다.`;
  return h("section", { class: `card er-rule side-${r.side} st-${lv.state}` },
    h("header", {}, h("span", { class: "er-code" }, r.code), h("h2", {}, r.label),
      h("span", { class: `er-side ${r.side}` }, ER_SIDE[r.side]), erChip(lv.state, lv.n, need)),
    h("ul", { class: "er-cond" }, r.conditions.map((c) => h("li", {}, c))),
    h("p", { class: "er-dim" }, `진입 ${r.entry} · 기간 ${r.horizon} · 성적은 SPY 대비 초과수익`),
    h("div", { class: "er-live" },
      h("div", {}, h("div", { class: "eyebrow" }, "실전 승률"), h("strong", { class: lv.win == null ? "" : (r.side === "buy" ? cls(lv.win - 0.5) : cls(0.5 - lv.win)) }, erRate(lv.win)),
        h("p", { class: "er-dim" }, `닫힘 ${lv.n}건 · 같은 칸 전체 ${erRate(lv.base_win)}(${lv.base_n}건)`)),
      h("div", {}, h("div", { class: "eyebrow" }, "중앙 초과수익"), h("strong", { class: cls(lv.median) }, erPct(lv.median)),
        h("p", { class: "er-dim" }, `평균 ${erPct(lv.mean)}`)),
      h("div", {}, h("div", { class: "eyebrow" }, "진행 중"), h("strong", {}, `${r.open.length}건`),
        h("p", { class: "er-dim" }, "기간이 끝나면 닫힘으로 옮깁니다"))),
    h("p", { class: `er-verdict st-${lv.state}` }, verdict), erResearch(r));
}

function erFeat(row) {
  const bits = [];
  if (row.mcap_b != null) bits.push(`시총 ${row.mcap_b >= 100 ? Math.round(row.mcap_b) : row.mcap_b.toFixed(1)}B`);
  if (row.dist_hi != null) bits.push(`고점 ${erPct(row.dist_hi, 0)}`);
  if (row.vol_ratio != null) bits.push(`거래량 ${row.vol_ratio.toFixed(1)}배`);
  if (row.clv != null) bits.push(`종가 위치 ${Math.round(row.clv * 100)}%`);
  if (row.vol60 != null) bits.push(`하루 변동 ${row.vol60.toFixed(1)}%`);
  return bits.join(" · ");
}

function erTicker(row) {
  return [h("b", { class: "mono" }, row.ticker), row.ai && h("span", { class: "tag", title: "AI 유니버스(watchlist.AI_SUBSECTOR)" }, row.ai),
    h("small", {}, row.name)];
}

function erEntryTable(rows, r, closed) {
  if (!rows.length) return h("p", { class: "er-dim" }, closed ? "아직 닫힌 사례가 없습니다." : "진행 중인 사례가 없습니다.");
  return h("div", { class: "er-scroll" }, h("table", { class: "er-table" },
    h("thead", {}, h("tr", {}, ["종목", "반응일", "실적날 SPY 대비", r.code === "A2" ? "뒤 5일" : "조건 값", closed ? "결과(SPY 대비)" : "지금까지", "경과", "시각"]
      .map((s, i) => h("th", { class: i === 3 && r.code !== "A2" ? "er-feat" : null }, s)))),
    h("tbody", {}, rows.map((row) => h("tr", {},
      h("td", {}, erTicker(row)), h("td", {}, md(row.reaction)),
      h("td", { class: cls(row.r0_x) }, erPct(row.r0_x)),
      h("td", { class: r.code === "A2" ? cls(row.d5_x) : "er-feat" }, r.code === "A2" ? erPct(row.d5_x) : erFeat(row)),
      h("td", { class: cls(row.x) }, erPct(row.x)),
      h("td", {}, `${row.days ?? 0}거래일${row.capped && closed ? " (다음 실적 못 찾음)" : ""}`),
      h("td", { class: "er-dim" }, ER_SRC[row.timing] || row.timing))))));
}

function erRecent(data) {
  let rows = data.recent;
  if (ER.onlyMatched) rows = rows.filter((r) => r.matches.length || r.wait_a2);
  if (ER.onlyAI) rows = rows.filter((r) => r.ai);
  const seg = (label, on, set) => h("button", { role: "switch", "aria-checked": String(on), class: `er-toggle ${on ? "on" : ""}`, onclick: set }, label);
  return h("section", { class: "card" },
    h("header", {}, h("h2", {}, "최근 10거래일 실적 발표"),
      h("span", { class: "sub" }, `유니버스(지금 시총 30억 달러 이상) ${data.recent.length}건 중 ${rows.length}건 표시`),
      h("div", { class: "right" },
        seg("규칙에 맞은 것만", ER.onlyMatched, () => { ER.onlyMatched = !ER.onlyMatched; erRender(); }),
        seg("AI 종목만", ER.onlyAI, () => { ER.onlyAI = !ER.onlyAI; erRender(); }))),
    rows.length ? h("div", { class: "er-scroll tall" }, h("table", { class: "er-table" },
      h("thead", {}, h("tr", {}, ["종목", "반응일", "실적날 SPY 대비", "규칙", "특징", "20일 성적(지금까지)"]
        .map((s, i) => h("th", { class: i === 4 ? "er-feat" : null }, s)))),
      h("tbody", {}, rows.map((row) => h("tr", { class: row.matches.length ? "er-hit" : "" },
        h("td", {}, erTicker(row)), h("td", {}, md(row.reaction)), h("td", { class: cls(row.r0_x) }, erPct(row.r0_x)),
        h("td", {}, row.matches.map((m) => h("span", { class: `er-tag ${m === "A2" ? "avoid" : "buy"}` }, m)),
          row.wait_a2 && h("span", { class: "er-tag wait", title: "A2는 반응 뒤 5거래일 종가로 판정합니다" }, "A2 대기")),
        h("td", { class: "er-feat" }, erFeat(row)), h("td", { class: cls(row.x) }, erPct(row.x))))))) :
      h("p", { class: "er-dim" }, "조건에 맞는 발표가 없습니다."));
}

function erRender() {
  const data = ER.data, root = $("#erules-view");
  if (!data) return;
  const need = data.alarm_min_n;
  const counts = data.status_counts || {};
  mount(root,
    h("div", { class: "page-heading" }, h("div", { class: "eyebrow" }, "실적 뒤 흐름 · 조건부 규칙 자동 추적"),
      h("h1", {}, "실적 뒤 흐름"),
      h("p", {}, "실적 발표 뒤 조건부 규칙 세 개(B1·B4·A2)에 맞는 종목을 자동으로 쌓고, 그 뒤 SPY 대비 성적을 채점합니다. ",
        `실전 성적은 연구 구간이 끝난 ${md(data.live_start)} 반응일부터만 셉니다. 관찰용이며 매수 추천이 아닙니다.`)),
    h("div", { class: "er-tools" },
      h("button", { class: "btn", onclick: () => erLoad(true) }, "달력·가격 다시 받기"),
      h("span", { id: "er-status", class: "er-dim", role: "status" },
        `갱신 ${erTime(data.updated_at)} · 최근 확정 종가 ${data.latest_close || "—"} · 채점한 발표 ${data.events_n}건`,
        counts.waiting ? ` · 반응일 종가 대기 ${counts.waiting}건` : "",
        data.unknown_timing?.length ? ` · 발표 시각 미상으로 뺀 ${data.unknown_timing.length}건` : "")),
    data.warning && h("p", { class: "er-note", role: "status" }, data.warning),
    h("div", { class: "er-rules" }, data.rules.map((r) => erCard(r, need))),
    ...data.rules.map((r) => h("section", { class: "card" },
      h("header", {}, h("h2", {}, `${r.code} ${r.label} — 진행 중`), h("span", { class: "sub" }, `${r.entry} 진입, ${r.horizon}`)),
      erEntryTable(r.open, r, false),
      h("details", { class: "er-closed" }, h("summary", {}, `닫힌 사례 ${r.closed.length}건`), erEntryTable(r.closed, r, true)))),
    erRecent(data),
    h("details", { class: "notes-panel" }, h("summary", {}, "계산 방법과 경보 기준"),
      h("ul", {},
        h("li", {}, data.method),
        h("li", {}, `경보: 닫힌 실전 사례가 ${need}건 이상일 때 매수 쪽 규칙(B1·B4)은 승률 50% 미만, 피할 쪽 규칙(A2)은 50% 초과면 경보입니다. B4·A2는 2025년에 반대로 움직였던 규칙이라 장세 전환을 가장 먼저 알려 줄 칸입니다.`),
        h("li", {}, "근거 연구: data/derived/analysis/earnings_trend/rules_report.md(1,629종목 12,441건, 2026 상반기로 고르고 7~9월로 확인, 우연 기준선·보통 날·대형주 점검). 정본: docs/desk_system.md §15."),
        h("li", {}, "승률 60% 안팎은 10번 중 4번은 틀린다는 뜻입니다. 규칙은 후보를 거르는 체이지 확신의 근거가 아닙니다."))));
}

window.addEventListener("desk:tab", (e) => {
  if (e.detail === "erules") erLoad();
});
