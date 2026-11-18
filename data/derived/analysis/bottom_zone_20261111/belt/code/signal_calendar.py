"""A standalone day-by-day view of the registered BB-1 signal."""
from datetime import date
from html import escape
from itertools import groupby


def state(row):
    if row["on"] is None:
        return "unknown", "수급 미확인"
    return ("on", "켜짐") if row["on"] else ("off", "꺼짐")


def details(row):
    parts = [row["day"], state(row)[1]]
    if row.get("basis") == "reconstructed":
        parts.append("사후 복원")
    drawdown, percentile = row["dd60"], row["percentile"]
    parts.append("SOXX 낙폭 " + ("미확인" if drawdown is None else f"{drawdown:.1%}"))
    parts.append("SOXL 수급 " + ("미확인" if percentile is None else f"{percentile:.1f}백분위"))
    return " · ".join(parts)


def render(rows, checked_at):
    """Render current-year US exchange sessions; a missing observation breaks an ON run."""
    if not rows:
        return '<!doctype html><html lang="ko"><meta charset="utf-8"><title>BB-1</title><p>올해 확인된 거래일이 없습니다.</p></html>'
    last = rows[-1]
    year = date.fromisoformat(last["day"]).year
    latest_class, latest_text = state(last)
    known = sum(r["on"] is not None for r in rows)
    unknown = len(rows) - known
    active = sum(bool(r["on"]) for r in rows)
    runs = [list(group) for key, group in groupby(rows, key=lambda r: bool(r["on"])) if key]
    ranges = []
    for group in runs:
        first, end = group[0]["day"][5:].replace("-", "/"), group[-1]["day"][5:].replace("-", "/")
        label = first if len(group) == 1 else f"{first}–{end}"
        ranges.append(f'<li>{label} <small>{len(group)}일</small></li>')
    run_list = '<ul class="runs">' + ''.join(ranges) + '</ul>' if ranges else '<p>이 기간에 확인된 신호가 없습니다.</p>'

    months, views = [], []
    latest_month = last["day"][:7]
    for key, records in groupby(rows, key=lambda r: r["day"][:7]):
        group = list(records)
        month = int(key[-2:])
        count = sum(bool(r["on"]) for r in group)
        current = key == latest_month
        months.append(
            f'<button type="button" class="month" data-month="{month}" aria-controls="m{month}" '
            f'aria-pressed="{str(current).lower()}" aria-label="{month}월, 신호 {count}일">'
            f'{month}월 <b>{count}일</b></button>'
        )
        cells = []
        for row in group:
            d = date.fromisoformat(row["day"])
            week = (d.day - 1 + d.replace(day=1).weekday()) // 7
            cell_class, label = state(row)
            desc = escape(details(row), quote=True)
            mark = ' aria-current="date"' if row["day"] == last["day"] else ''
            cells.append(
                f'<button type="button" class="date {cell_class}" style="grid-column:{d.weekday()+1};grid-row:{week+2}" '
                f'data-detail="{desc}" aria-label="{desc}" title="{desc}"{mark}>{d.day}</button>'
            )
        headings = ''.join(f'<small>{day}</small>' for day in "월화수목금")
        hidden = '' if current else ' hidden'
        views.append(
            f'<section id="m{month}" class="days" aria-label="{month}월 미국 거래일"{hidden}>'
            + headings + ''.join(cells) + '</section>'
        )
    last_badge = ' · 사후 복원' if last.get("basis") == "reconstructed" else ''
    drawdown = '미확인' if last['dd60'] is None else f'{last["dd60"]:.1%}'
    strength = '미확인' if last['percentile'] is None else f'{last["percentile"]:.1f}백분위'
    return f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BB-1 신호 달력 · {year}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:var(--background,#fff);color:var(--foreground,#172326);font:14px/1.55 var(--font-sans,system-ui),sans-serif}}
main{{max-width:940px;margin:auto;padding:18px}}h1,h2,p{{margin:0}}h1{{font-size:22px;letter-spacing:-.04em}}h2{{font-size:16px}}
.header,.latest,.legend,.months,.runs{{display:flex;align-items:center;flex-wrap:wrap;gap:6px 12px}}.header{{justify-content:space-between}}
small,.muted{{color:var(--muted-foreground,#565f63)}}.latest{{margin:7px 0 12px}}.status{{border-radius:6px;padding:2px 8px;font-weight:700}}
.on,.swatch.on{{background:#166853;color:white}}.off,.swatch.off{{background:var(--muted,#e9edec);color:var(--foreground,#233532)}}
.unknown,.swatch.unknown{{background:var(--background,#fff);color:var(--muted-foreground,#606266);border:1px dashed currentColor!important}}
.readings{{display:flex;flex-wrap:wrap;gap:4px 18px}}.readings b{{font-variant-numeric:tabular-nums}}.frequency{{margin:12px 0 6px}}
.months{{gap:4px}}button{{font:inherit;cursor:pointer;color:inherit}}.month{{border:1px solid var(--border,#d0d8d6);border-radius:5px;background:var(--background,#fff);padding:3px 7px}}
.month b{{font-weight:500;color:var(--muted-foreground,#56615c)}}.month[aria-pressed=true]{{box-shadow:inset 0 -2px var(--primary,#266951);border-color:var(--primary,#266951)}}
.month:hover{{background:var(--accent,#e9efed);color:var(--accent-foreground,#26352e)}}button:focus-visible{{outline:2px solid var(--ring,#307ca7);outline-offset:1px}}
.body{{display:grid;grid-template-columns:minmax(150px,270px) minmax(180px,1fr);gap:18px;margin-top:15px}}.days{{display:grid;gap:3px;grid-template-columns:repeat(5,1fr);align-content:start}}.days[hidden]{{display:none}}.days small{{text-align:center}}
.date{{height:26px;border:0;border-radius:4px;font-size:13px;font-variant-numeric:tabular-nums}}.date[aria-current=date]{{outline:2px solid var(--foreground,#31463f);outline-offset:-2px}}
.legend{{gap:4px 10px;font-size:12px;margin:6px 0}}.swatch{{width:12px;height:10px;display:inline-block;vertical-align:baseline;border-radius:2px}}
.runs{{list-style:none;padding:0;margin:5px 0 8px;gap:4px 14px;font-variant-numeric:tabular-nums}}.runs li{{white-space:nowrap}}
.detail{{border-top:1px solid var(--border,#d0d8d6);margin-top:9px;padding-top:7px;min-height:25px}}.stamp{{font-size:12px;margin-top:7px}}
@media(max-width:470px){{main{{padding:12px}}.body{{grid-template-columns:1fr;gap:12px}}.days{{max-width:330px}}.header{{align-items:baseline}}}}
</style></head><body><main>
<div class="header"><h1>BB-1 신호 달력</h1><small>{year}년 · 미국 거래일</small></div>
<div class="latest"><strong>{last["day"]}</strong><span class="status {latest_class}">{latest_text}{last_badge}</span></div>
<div class="readings"><span>SOXX 60거래일 고점 대비 <b>{drawdown}</b> <small>(기준 −20% 이하)</small></span>
<span>한국인 SOXL 매수 비중 <b>{strength}</b> <small>(기준 80백분위 이상)</small></span></div>
<p class="frequency"><b>신호 {active}거래일 · 연속 {len(runs)}구간</b> <small>판정 가능 {known}일 · 수급 미확인 {unknown}일</small></p>
<nav class="months" aria-label="월별 신호 일수">{''.join(months)}</nav>
<div class="body"><div>{''.join(views)}<div class="legend">
<span><i class="swatch on"></i> 켜짐</span><span><i class="swatch off"></i> 꺼짐</span>
<span><i class="swatch unknown"></i> 수급 미확인</span></div></div>
<div><h2>신호가 이어진 거래일</h2>{run_list}
<small>미국장 휴일과 주말은 건너뜁니다. 꺼짐이나 수급 미확인일을 만나면 구간을 나눕니다.</small></div></div>
<p class="detail" id="detail" aria-live="polite">{escape(details(last))}</p>
<p class="stamp muted">조회 {escape(str(checked_at))} · 날짜를 눌러 그날의 두 조건을 확인하세요.</p>
</main><script>
const months=document.querySelectorAll('.month'), views=document.querySelectorAll('.days');
const detail=document.getElementById('detail');
for(const button of months)button.addEventListener('click',()=>{{
  for(const item of months)item.setAttribute('aria-pressed',String(item===button));
  const id=button.getAttribute('aria-controls');
  for(const panel of views)panel.hidden=panel.id!==id;
  const target=document.querySelector('#'+id+' .date.on')||document.querySelector('#'+id+' .date');
  detail.textContent=target.dataset.detail;
}});
for(const button of document.querySelectorAll('.date'))button.addEventListener('click',()=>{{detail.textContent=button.dataset.detail;}});
</script></body></html>'''
