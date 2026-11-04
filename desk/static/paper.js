"use strict";
/* Frozen SOXL/SOXS paper records. GET reads; POST catches up completed sessions. */
const PP = { data: null, busy: false, error: null };
const ppPct = v => v == null || !Number.isFinite(v) ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(2)}%`;
const ppCount = v => Array.isArray(v) ? v.length : Number(v || 0);
const ppTime = (s, zone = "Asia/Seoul") => {
  if (!s) return "—";
  const d = new Date(s);
  return Number.isNaN(d.valueOf()) ? s : d.toLocaleString("ko-KR", {
    timeZone: zone, month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false,
  });
};

async function ppLoad(update = false) {
  if (PP.busy) return;
  PP.busy = true;
  PP.error = null;
  ppRender();
  try {
    PP.data = await api(update ? "/api/paper/update" : "/api/paper", update ? { method: "POST", body: "{}" } : {});
  } catch (error) {
    PP.error = error.message;
  } finally {
    PP.busy = false;
    ppRender();
  }
}

function ppTable(head, rows, caption) {
  return h("div", { class: "pp-scroll", tabindex: "0", "aria-label": caption },
    h("table", { class: "pp-table" }, h("caption", { class: "pp-sr" }, caption),
      h("thead", {}, h("tr", {}, head.map(s => h("th", { scope: "col" }, s)))), h("tbody", {}, rows)));
}

function ppMetric(label, value, foot, tone = "") {
  return h("div", {}, h("div", { class: "eyebrow" }, label), h("strong", { class: tone }, value),
    foot && h("p", { class: "pp-dim" }, foot));
}

function ppRule(rule, data) {
  const ready = Number(rule.sessions || 0) > 0;
  const mean = ready ? rule.mean_daily_pct : null;
  const total = ready ? rule.total_return_pct : null;
  const ci = ready ? rule.ci95_pct : null;
  const months = rule.months || [];
  return h("article", { class: "card pp-rule" },
    h("header", {}, h("span", { class: "pp-code" }, rule.id), h("h2", {}, rule.name), h("span", { class: "pp-tag" }, "종이 기록")),
    h("ul", { class: "pp-conditions" }, (rule.conditions || []).map(s => h("li", {}, s))),
    h("div", { class: "pp-metrics" },
      ppMetric("하루 평균", ppPct(mean), "신호 없는 날 0 포함", cls(mean)),
      ppMetric("누적 손익", ppPct(total), "이 규칙의 가상 경로", cls(total)),
      ppMetric("평가일 / 매매", `${rule.sessions || 0}일 / ${rule.trades || 0}건`, "자료가 완전한 날만")),
    h("dl", { class: "pp-stats" },
      h("div", {}, h("dt", {}, "하루 평균 95% 구간"), h("dd", {}, ci ? `${ppPct(ci[0])} ~ ${ppPct(ci[1])}` : "—")),
      h("div", {}, h("dt", {}, "호가 비용 2배 · 하루 평균"), h("dd", {}, ready ? ppPct(rule.cost2_mean_pct) : "—")),
      h("div", {}, h("dt", {}, "상위 2일 제외 · 하루 평균"), h("dd", {}, ready ? ppPct(rule.drop_top2_mean_pct) : "—"))),
    h("p", { class: "pp-review" }, rule.review_ready
      ? `${data.review_sessions || 60}개 평가일이 쌓였습니다. 두 규칙의 성적과 자료 누락을 함께 점검할 때입니다.`
      : `${rule.sessions || 0}/${data.review_sessions || 60}개 평가일 · 첫 점검을 위해 쌓는 중입니다.`),
    months.length > 0 && h("details", { class: "pp-months" }, h("summary", {}, "월별 기록"),
      ppTable(["월", "평가일", "가상 손익"], months.map(m => h("tr", {}, h("td", {}, m.month),
        h("td", {}, `${m.days}일`), h("td", { class: cls(m.return_pct) }, ppPct(m.return_pct)))), `${rule.id} 월별 기록`)));
}

function ppTrades(day) {
  const groups = Object.entries(day.rules || {});
  return h("details", { class: "pp-trades" }, h("summary", {}, "매매별 가상 체결"),
    ...groups.map(([id, r]) => h("div", {}, h("h3", {}, id), (r.trades || []).length
      ? ppTable(["종목", "진입 KST", "청산 KST", "청산 사유", "가상 손익"], r.trades.map(t => h("tr", {},
        h("td", {}, t.symbol),
        h("td", { title: `ET ${t.entry_at_et || t.entry_bar_et || "—"}` }, ppTime(t.entry_at_kst || t.entry_at_et)),
        h("td", { title: `ET ${t.exit_at_et || t.exit_bar_et || "—"}` }, ppTime(t.exit_at_kst || t.exit_at_et), t.why === "target" && " 봉 안"),
        h("td", {}, ({ target: "목표 도달", cutoff: "시간 마감", signal: "신호 청산", stop: "손절" })[t.why] || t.why),
        h("td", { class: cls(t.return_pct) }, ppPct(t.return_pct)))), `${day.session} ${id} 가상 체결`)
      : h("p", { class: "pp-dim" }, "신호 없음 · 매매 0건"))),
    h("p", { class: "pp-dim" }, "1분봉으로 계산한 가상 체결입니다. 시장가 시각은 봉 마감, 목표 도달 시각은 해당 봉 안을 뜻합니다."));
}

function ppDays(data) {
  const days = [...(data.days || [])].sort((a, b) => b.session.localeCompare(a.session));
  return h("section", { class: "card pp-days", id: "pp-days" },
    h("header", {}, h("h2", {}, "날짜별 원장"), h("span", { class: "sub" }, "미국 거래일 기준 · 최근 날짜부터")),
    !days.length ? h("p", { class: "pp-empty" },
      data.initialized ? `${data.start_session}부터 기록합니다. 마감 후 자료가 준비되면 첫 성적이 표시됩니다.` : "등록된 종이 기록이 없습니다.")
      : ppTable(["미국 거래일", "상태", "S1 손익", "S1 매매", "S2 손익", "S2 매매", "상세"], days.map(day => {
        const valid = day.status === "valid", s1 = day.rules?.S1, s2 = day.rules?.S2;
        return h("tr", { class: valid ? "" : "pp-missing" }, h("td", {}, day.session),
          h("td", {}, valid ? "평가 완료" : "미평가", !valid && h("small", {}, day.error || "자료 누락")),
          h("td", { class: valid ? cls(s1?.return_pct) : "" }, valid ? ppPct(s1?.return_pct) : "—"),
          h("td", {}, valid ? `${s1?.trades?.length || 0}건` : "—"),
          h("td", { class: valid ? cls(s2?.return_pct) : "" }, valid ? ppPct(s2?.return_pct) : "—"),
          h("td", {}, valid ? `${s2?.trades?.length || 0}건` : "—"),
          h("td", { class: "pp-detail-cell" }, valid ? ppTrades(day) : "자료를 다시 받으면 재확인"));
      }), "종이 기록 날짜별 손익과 자료 상태"));
}

function ppRender() {
  const data = PP.data, root = $("#paper-view");
  if (!root) return;
  const head = h("section", { class: "page-heading" },
    h("div", {}, h("div", { class: "eyebrow" }, "SOXL · SOXS / 앞으로 쌓이는 날짜"), h("h1", {}, "종이 기록"),
      h("p", {}, "조건을 고정한 두 규칙의 가상 손익을 매일 따로 확인합니다. 아직 확정된 매매 규칙은 없습니다.")),
    h("div", { class: "pp-tools" }, h("button", { class: "btn", disabled: PP.busy || !data?.initialized,
      onclick: () => ppLoad(true) }, PP.busy ? "기록 확인 중…" : "기록 갱신")));
  const error = PP.error && h("div", { class: "pp-error", role: "alert" }, PP.error,
    h("button", { class: "btn", onclick: () => ppLoad() }, "다시 불러오기"));
  if (!data) {
    mount(root, head, error, h("p", { role: "status", class: "pp-dim" }, PP.busy ? "종이 기록을 읽는 중…" : "기록을 불러오지 못했습니다."));
    return;
  }
  const n = ppCount(data.valid_sessions), need = data.review_sessions || 60;
  const info = h("section", { class: "pp-overview" },
    ppMetric("기록 시작", data.start_session || "등록 대기", "미국 거래일 기준"),
    ppMetric("완료한 평가일", `${n} / ${need}`, "신호 없는 날도 포함"),
    ppMetric("자료 누락 / 대기", `${ppCount(data.missing_sessions)} / ${ppCount(data.pending_sessions)}`, "누락은 손익 0으로 넣지 않음"),
    ppMetric("매매 마감", "한국 24:00", "겨울 10:00 · 여름 11:00 ET"));
  const note = h("div", { class: "pp-note", role: "status" },
    PP.busy ? "완료한 거래일의 자료를 확인하고 있습니다…" : (data.status || (data.initialized ? "앞으로 쌓이는 거래일을 기다립니다." : "종이 기록 등록 대기")),
    h("p", {}, "두 규칙은 서로 독립된 가상 경로입니다. 실제 계좌 성적과 분리하며, 첫 60개 평가일 점검은 실전 채택 기준이 아닙니다."));
  const warnings = (data.warnings || []).length > 0 ? h("div", { class: "pp-warnings", role: "status" },
    data.warnings.map(s => h("p", {}, s))) : null;
  mount(root, head, error, info, note, warnings,
    h("section", { class: "pp-rules", "aria-label": "고정한 두 종이 기록 규칙" }, (data.rules || []).map(r => ppRule(r, data))),
    ppDays(data),
    h("details", { class: "notes-panel pp-notes" }, h("summary", {}, "기록 방법과 고정 조건"),
      h("ul", {},
        h("li", {}, `규칙·비용표 고정 ${ppTime(data.frozen_at)} KST · 기록 시작 이전에 본 날짜는 새 성적에서 제외합니다.`),
        h("li", {}, "완료한 장초 구간의 SIP 1분봉을 사용합니다. 신호 다음 봉 종가에 시장가 체결을 가정하고, 시장가에는 고정한 반 호가 비용을 반영합니다."),
        h("li", {}, "목표는 체결 다음 봉부터 확인합니다. 신호가 없으면 그날 손익은 0, 자료가 누락되면 미평가로 남깁니다."),
        h("li", {}, "첫 60개 완전한 거래일에 두 규칙을 함께 점검합니다. 중간 성적에 따라 조건을 바꾸지 않으며 실전 매매로 자동 전환하지 않습니다."),
        h("li", {}, "이 규칙들은 여러 후보를 비교한 뒤 남은 관찰 대상입니다. 과거 숫자는 새 날짜의 수익을 보장하지 않습니다.")),
      data.manifest_sha256 && h("p", { class: "pp-dim" }, "고정 기록 해시 ", h("code", { title: data.manifest_sha256 }, data.manifest_sha256.slice(0, 16)))));
}

window.addEventListener("desk:tab", e => { if (e.detail === "paper") ppLoad(); });
