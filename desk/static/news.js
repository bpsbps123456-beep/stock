"use strict";
/* Source evidence is reviewed by the researcher; this UI never changes orders. */
const NW = { data: null, episode: null, basis: "public", day: "", request: 0 };
const nwTime = s => s ? new Date(s).toLocaleString("ko-KR", {timeZone: "Asia/Seoul", hour12: false}) + " KST" : "미확인";
const nwPct = v => v == null ? "—" : pct(v, 2);

async function nwLoad(refresh = false) {
  const seq = ++NW.request;
  const params = new URLSearchParams({basis: NW.basis});
  if (NW.day) params.set("as_of", NW.day + "T23:59:59+09:00");
  if (refresh) params.set("refresh", "1");
  const status = $("#nw-status");
  if (status) status.textContent = refresh ? "가격 갱신 중…" : "저장된 뉴스 읽는 중…";
  try {
    const data = await api("/api/news?" + params);
    if (seq !== NW.request) return;
    NW.data = data;
    if (!NW.episode) NW.episode = data.episodes.at(-1)?.id;
    nwRender();
  } catch (error) {
    if (seq !== NW.request) return;
    const errorBox = h("div", {class: "nw-error", role: "alert"}, error.message,
      h("button", {class: "btn", onclick: () => nwLoad()}, "다시 불러오기"));
    if (NW.data) { nwRender(); $("#nw-status").append(errorBox); }
    else mount($("#nw-body"), errorBox);
  }
}

function nwArticle(event, data) {
  const direction = {support: "소진 가설 지지", against: "위험·반대 근거", conditional: "조건부 반등 재료"};
  const facts = [["대상", event.scope], ["측정값", event.measurement], ["집계 기간", event.window]];
  return h("article", {class: "nw-article card", id: "nw-e-" + event.id},
    h("div", {class: "nw-article-top"}, h("span", {class: "nw-tag"}, data.kinds[event.kind]),
      h("span", {class: "sub"}, direction[event.stance], event.duplicate_source ? " · 같은 원자료 재보도" : "")),
    h("h3", {}, h("a", {href: event.url, target: "_blank", rel: "noopener noreferrer"}, event.title, " ↗")),
    h("p", {class: "nw-meta"}, event.source, " · ", data.qualities[event.quality], " · 기사일 ", event.published_date),
    h("p", {}, event.summary),
    h("p", {class: "nw-interpretation"}, h("b", {}, "판단에 주는 의미 "), event.interpretation),
    h("p", {}, h("b", {}, "다음 확인 "), event.next_check),
    h("details", {}, h("summary", {}, "표본·수치·시각 확인"),
      h("dl", {}, facts.map(([label, value]) => [h("dt", {}, label), h("dd", {}, value)])),
      h("p", {class: "nw-meta"}, "조회에 쓰는 시각: ", nwTime(event.available_at), " · 첫 기록: ", nwTime(event.recorded_at)),
      h("p", {class: "nw-meta"}, event.time_note)));
}

function nwRender() {
  const data = NW.data, root = $("#nw-body");  // #bd-view(바닥 감지기, bottom.js) 아래
  const ep = data.episodes.find(e => e.id === NW.episode) || data.episodes[0];
  const controls = h("div", {class: "nw-controls"},
    h("label", {}, "비교 국면", h("select", {id: "nw-episode", onchange: e => { NW.episode = e.target.value; nwRender(); }},
      data.episodes.map(e => h("option", {value: e.id, selected: e.id === ep?.id}, e.label)))),
    h("label", {}, "자료 기준", h("select", {id: "nw-basis", onchange: e => { NW.basis = e.target.value; nwLoad(); }},
      [["public", "기사 공개 기준 복습"], ["recorded", "실제 기록 기준"]].map(([v,t]) => h("option", {value:v, selected:NW.basis===v},t)))),
    h("label", {}, "그날까지 보기 · 한국시간", h("input", {id: "nw-day", type: "date", value: NW.day,
      onchange: e => { NW.day = e.target.value; nwLoad(); }})),
    h("button", {class:"btn", onclick: () => { NW.day = ""; nwLoad(); }}, "현재 기록"),
    h("button", {class:"btn", onclick: () => nwLoad(true)}, "가격 갱신"));
  const review = ep?.assessment;
  const sourceTitle = id => ep.events.find(e => e.id === id)?.title || id;
  const reviewCard = h("section", {class: "nw-review card"},
    h("div", {class:"eyebrow"}, "지금까지의 판단 · ", review ? nwTime(review.recorded_at) : "해당 시각 기록 없음"),
    h("h2", {}, review?.conclusion || "이 시각에 남겨진 판단이 없습니다."),
    review && h("div", {class:"nw-evidence-sides"},
      h("div", {}, h("h3", {}, "소진·반등을 뒷받침하는 기사"), h("ul", {}, review.supports.map(id => h("li", {}, sourceTitle(id))))),
      h("div", {}, h("h3", {}, "함께 봐야 할 반대 근거"), h("ul", {}, review.contradicts.map(id => h("li", {}, sourceTitle(id)))))),
    review && h("div", {class:"nw-next"}, h("h3", {}, "다음 기사에서 확인할 것"),
      h("ul", {}, review.next_checks.map(s => h("li", {}, s)))));
  const quote = ep?.quote;
  const board = h("section", {class: "nw-board"}, Object.entries(data.kinds).map(([key, label]) => {
    const articles = (ep?.events || []).filter(e => e.kind === key);
    return h("article", {class: "card"}, h("div", {class: "eyebrow"}, label),
      h("strong", {}, articles.length ? `${ep.counts[key]}개 원자료` : "추가 확인 필요"),
      h("p", {}, articles.at(-1)?.title || "이 원장에 확인된 기사가 없습니다."));
  }));
  const allEvents = data.episodes.flatMap(e => e.events.map(r => ({...r, label:e.label})));
  const compareRows = allEvents.map(e => h("tr", {},
    h("td", {}, h("small", {}, e.label), h("a", {href:e.url,target:"_blank",rel:"noopener noreferrer"}, e.title)),
    h("td", {}, e.reaction.anchor || e.reaction.reason),
    ["1","3","5"].map(n => h("td", {class:cls(e.reaction.returns[n])}, nwPct(e.reaction.returns[n]))),
    h("td", {class:cls(e.reaction.worst_close_5)}, nwPct(e.reaction.worst_close_5))));
  const compare = h("section", {class:"card nw-comparison"}, h("h2", {}, "기사 공개 후 가격 반응 · 국면 비교"),
    h("p", {class:"sub"}, data.reaction_note, " 매매 체결 성과가 아닌 비교 관측입니다."),
    h("div", {class:"scroll"}, h("table", {},
      h("thead", {}, h("tr", {}, ["국면·기사", "기준 종가일", "+1일", "+3일", "+5일", "5일 내 최대 하락"].map(s=>h("th",{},s)))),
      h("tbody", {}, compareRows))));
  mount(root,
    h("section", {class:"page-heading"}, h("div", {}, h("div", {class:"eyebrow"}, "중요 뉴스와 가격의 반응을 함께 추적"),
      h("h1", {}, "바닥을 찾는 뉴스 근거"), h("p", {}, "매도 압력 → 매물 소화 단서 → 가격이 버티는지 확인"))),
    controls,
    h("div", {id:"nw-status", class:"nw-meta", role:"status"}, "기사 첫 기록 최신: ", nwTime(data.last_recorded_at),
      " · 조회 기준: ", nwTime(data.as_of), " · 새 기사 추가는 ‘뉴스 근거 업데이트’로 요청하세요."),
    data.warning && h("p", {class:"nw-error",role:"alert"}, data.warning),
    h("p", {class:"nw-note"}, NW.basis === "public" ? data.replay_note : "각 시각까지 실제로 기록한 기사만 표시합니다. 과거 기사를 나중에 수집한 경우 이전 날짜에는 나타나지 않습니다."),
    reviewCard, board,
    h("section", {class:"card nw-quote"}, h("h2", {}, ep?.benchmark || "가격"),
      quote ? h("p", {}, `${quote.day} 종가 ${usd(quote.close)} · 하루 ${nwPct(quote.day_return)} · 직전 5거래일 최저 종가 대비 ${nwPct(quote.above_prior5_low)}`) : h("p", {}, "가격 자료가 없습니다. 가격 갱신을 눌러 주세요."),
      h("p", {class:"sub"}, "해당 조회 시각에 장이 끝난 종가만 사용합니다. 가격 갱신은 기사를 새로 수집하지 않습니다.")),
    compare,
    h("section", {class:"nw-articles"}, h("h2", {}, ep?.label || "기사"),
      (ep?.events || []).slice().reverse().map(e => nwArticle(e,data)),
      !ep?.events.length && h("p", {}, "해당 시각까지 확인된 기사가 없습니다.")),
    h("footer", {class:"nw-meta"}, data.method, " · 가격: ", data.price_source || "미수집", " · 수집 시각: ", nwTime(data.prices_fetched_at),
      h("p", {}, "현재는 2026년 7월·10월 두 국면을 기록한 관찰판입니다. 근거 개수는 바닥 확률 점수가 아닙니다.")));
}

window.addEventListener("desk:tab", e => {
  if (e.detail === "news") nwLoad();
});
