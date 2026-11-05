import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import crypto from 'node:crypto';
import { gzipSync } from 'node:zlib';
import { createRequire } from 'node:module';
import os from 'node:os';

// Run with the Codex primary runtime's node executable. The public package is
// Windows executable: %USERPROFILE%/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe
// Example: <runtime-node.exe> data/derived/analysis/seibro_bottoms/code/soxs_ledger_presentation.mjs
// resolved only from its dependency bundle; repository node_modules are unused.
// Optional: SOXS_ARTIFACT_NODE_MODULES overrides the runtime node_modules path.
// A task runtime node_modules junction was used for initial authoring.
const modulesDir=process.env.SOXS_ARTIFACT_NODE_MODULES||path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
const runtimeRequire=createRequire(path.join(modulesDir,'..','package.json'));
const {Workbook,SpreadsheetFile}=await import(pathToFileURL(runtimeRequire.resolve('@oai/artifact-tool')).href);

// Presentation only: all ranks, flags and price responses come from the CSVs.
let repo=process.env.SOXS_LEDGER_REPO_ROOT||path.dirname(fileURLToPath(import.meta.url));
while(true){try{await fs.access(path.join(repo,'AGENTS.md'));break;}catch{const up=path.dirname(repo);if(up===repo)throw new Error('Set SOXS_LEDGER_REPO_ROOT to the repository root.');repo=up;}}
const outputDir = path.join(repo,'outputs/soxs_ledger_20261105');
const sourceDir = path.join(repo,'data/derived/analysis/seibro_bottoms/results/soxs_daily_ledger_20261105');
const wb = Workbook.create();
const font = 'Arial';
const colors = { ink:'#203047', navy:'#243C5A', pale:'#EDF2F7', on:'#D9EEE8', unknown:'#FFF0CB', off:'#EEF1F5', red:'#FCE3E3' };
const keys = ['gross_usd_L1','gross_usd_L5','net_usd_L1','net_usd_L5','buy_share_L1','buy_share_L5'];
const labels = {gross_usd_L1:'매수 1일',gross_usd_L5:'매수 5일',net_usd_L1:'순매수 1일',net_usd_L5:'순매수 5일',buy_share_L1:'자기 매수비율 1일',buy_share_L5:'자기 매수비율 5일'};
const dictionary = JSON.parse(await fs.readFile(path.join(sourceDir,'column_dictionary.json'),'utf8'));
const ko={source_date:'원자료 미국 거래일',entry:'미국 진입일',month:'월',key:'신호 정의',SOXS_buy:'SOXS 매수 (USD)',SOXS_sell:'SOXS 매도 (USD)',SOXS_net:'SOXS 순매수 (USD)',SOXS_tot:'SOXS 매수+매도 (USD, 원본 보고)',cens_SOXS:'TOP50 밖 가려짐',part_SOXS:'부분 절단',known_settle_kst:'공개 규약 시각 (KST)',entry_settle:'연결 미국 진입일',source_observed:'원자료 관측됨',selected_for_entry:'최신 source로 선택',selected_source_for_same_entry:'같은 진입일의 최신 source',selection_reason:'선택 사유',previous_close:'진입 전일 SOXX 종가',previous_dd60_pct:'진입 전일 DD60 (%)',SOXX_Open:'SOXX 시가',SOXX_Close:'SOXX 종가',SOXX_daily_return_pct:'SOXX 당일 수익률 (%)',dd60_pct:'SOXX DD60 (%)',source_sessions:'source 세션 수',entry_sessions:'entry 세션 수',source_on:'source ON 수',source_off:'source OFF 수',source_unknown:'source UNKNOWN 수',source_unpublished:'source 미공개 수',source_shadowed_on:'source ON 중 덮임',entry_latest_on:'entry ON 수',entry_latest_off:'entry OFF 수',entry_unknown:'entry UNKNOWN 수',entry_no_new_source:'새 source 없는 entry',dd10_on:'DD10 상태 ON 수',entry_on_price_pending:'가격 대기 중 entry ON'};
for(const [filename,meta] of Object.entries(dictionary.files||{}))for(const [column,info]of Object.entries(meta.columns||{})){
  if(ko[column])info.ko=ko[column];
  for(const k of keys)if(column===k||column.startsWith(k+'_')){const suffix=column.slice(k.length),suffixLabel={'':' 값'+(k.startsWith('buy_share')?' (0~1)':' (USD)'),_P:' 순위 (0~100)',_rank_n:' 유효 관측 수',_status:' 상태',_on:' ON 여부',_transition:' 상태 전환',_on_streak_sessions:' ON 연속 세션',_status_run_sessions:' 같은 상태 연속 세션',_dd10_on:' DD10 ON',_dd10_eligible:' DD10 평가 유효',_dd10_status:' DD10 상태',_event5:' 5세션 간격 사건',_event10:' 10세션 간격 사건',_event20:' 20세션 간격 사건'}[suffix];if(suffixLabel!==undefined)info.ko=labels[k]+suffixLabel;}
  const m=column.match(/^(R|MAE|MFE)(5|10|20)_(pct|complete|end)$/);if(m)info.ko=m[2]+'세션 '+({R:'SOXX 반응',MAE:'최대 추가하락',MFE:'최대 상승'}[m[1]])+({pct:' (%)',complete:' 전창 완성',end:' 종료일'}[m[3]]);
}
const summary = JSON.parse(await fs.readFile(path.join(sourceDir,'summary.json'),'utf8'));
const coverage = JSON.parse(await fs.readFile(path.join(sourceDir,'coverage.json'),'utf8'));
const sha = s => crypto.createHash('sha256').update(s).digest('hex');
function col(i){let s='';for(i++;i;i=Math.floor((i-1)/26))s=String.fromCharCode(65+(i-1)%26)+s;return s;}
const excelValue=v=>v instanceof Date?+v/86400000+25569:v;
function dateType(name){return /(^entry$|^month$|date$|^entry_settle$|^known_settle$|_kst$|^unit_first$|^unit_last$|^R\d+_end$|^selected_source_for_same_entry$)/.test(name);}
function convert(name,v){
  if(v === '' || v === null || v === undefined || /^(nan|nat|none|null)$/i.test(String(v))) return null;
  if(v instanceof Date)return new Date(+v+(/_kst$/.test(name)?9*3600000:0));
  if(dateType(name) && /^\d{4}-\d{2}(-\d{2})?/.test(String(v))) {
    const m=String(v).match(/^(\d{4})-(\d{2})(?:-(\d{2}))?(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?/);
    return new Date(Date.UTC(+m[1],+m[2]-1,+(m[3]||1),+(m[4]||0),+(m[5]||0),+(m[6]||0)));
  }
  if(/^(true|false)$/i.test(String(v)))return String(v).toLowerCase()==='true';
  if(/^[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?$/i.test(String(v)) && !/^unit_id$|isin|reason|_status$/.test(name))return Number(v);
  return String(v);
}
async function readCsv(filename){
  const txt=await fs.readFile(path.join(sourceDir,filename),'utf8');
  const imp=await Workbook.fromCSV(txt,{sheetName:'Import'});
  const vals=imp.worksheets.getItemAt(0).getUsedRange().values;
  const headers=vals[0].map(String);
  const rows=vals.slice(1).filter(r=>r.some(v=>v!==''&&v!==null)).map(r=>Object.fromEntries(headers.map((h,i)=>[h,convert(h,r[i])])));
  return {filename,headers,rows,sha256:sha(txt)};
}
const source=await readCsv('source_daily.csv');
const entry=await readCsv('entry_daily.csv');
const monthly=await readCsv('monthly.csv');
const warmup=await readCsv('warmup_daily.csv');
function pick(ds,columns,rows=ds.rows){return {...ds,headers:columns.filter(h=>ds.headers.includes(h)),rows};}
const sheets={};
for(const name of ['요약','최근일별','전체source','전체entry','월별','2023준비','열설명']){
  const s=wb.worksheets.add(name);s.showGridLines=false;sheets[name]=s;
}
function base(s,range){s.getRange(range).format={font:{name:font,size:10,color:colors.ink},verticalAlignment:'center',rowHeight:22};}
function title(s,text,context=''){
  s.getRange('A2').values=[[text]];s.getRange('A2').format={font:{name:font,size:15,bold:true,color:colors.navy},rowHeight:27};
  if(context){s.getRange('A3').values=[[context]];s.getRange('A3').format={font:{name:font,size:10,color:'#54657A',italic:true},rowHeight:23};}
}
function formatColumn(s,header,index,start,end){
  const r=s.getRange(`${col(index)}${start}:${col(index)}${end}`);
  r.format.columnWidth=17;
  if(dateType(header)){r.setNumberFormat(/kst|known_settle/.test(header)?'yyyy-mm-dd hh:mm':'month'===header?'yyyy-mm':'yyyy-mm-dd');r.format.columnWidth=/kst|known_settle/.test(header)?24:16;}
  else if(/_status$|^state$/.test(header)){r.format.columnWidth=20;}
  else if(/_pct$/.test(header)){r.setNumberFormat('0.00"%"');r.format.horizontalAlignment='right';}
  else if(/buy_share_L[15]$/.test(header)){r.setNumberFormat('0.0%');r.format.horizontalAlignment='right';}
  else if(/_P$/.test(header)){r.setNumberFormat('0.0');r.format.horizontalAlignment='right';}
  else if(/SOXS_(buy|sell|net|tot)$|gross_usd_L[15]$|net_usd_L[15]$/.test(header)){r.setNumberFormat('"$"#,##0;[Red]("$"#,##0);"$"0');r.format.columnWidth=21;r.format.horizontalAlignment='right';}
  else if(/SOXX_|previous_close/.test(header)){r.setNumberFormat('0.00');r.format.horizontalAlignment='right';}
  else if(/reason|source_for|transition/.test(header)){r.format.columnWidth=34;}
  else if(/raw_paths|raw_source_url/.test(header)){r.format.columnWidth=100;r.format.wrapText=true;}
  else if(/_n$|_sessions$|_on$|^source_|^entry_|dd10_on/.test(header)){r.setNumberFormat('#,##0');}
  if(/_status$/.test(header)){
    for(const [status,fill] of [['ON',colors.on],['OFF',colors.off],['UNKNOWN',colors.unknown],['UNPUBLISHED',colors.unknown],['NO_NEW_SOURCE',colors.pale]])r.conditionalFormats.add('cellIs',{operator:'equal',formula:`"${status}"`,format:{fill,font:{bold:status==='ON',color:colors.ink}}});
  }
}
let tableId=0;
const placements=[];
function table(s,ds,start=6,{friendly=false}={}){
  const end=start+ds.rows.length,last=col(ds.headers.length-1);
  const headers=ds.headers.map(h=>friendly?(dictionary.files?.[ds.filename]?.columns?.[h]?.ko||h):h);
  s.getRange(`A${start}:${last}${end}`).values=[headers,...ds.rows.map(r=>ds.headers.map(h=>excelValue(r[h]??null)))];
  base(s,`A${start}:${last}${end}`);
  s.getRange(`A${start}:${last}${start}`).format={fill:colors.navy,font:{name:font,size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:friendly?48:40,horizontalAlignment:'center',verticalAlignment:'center'};
  ds.headers.forEach((h,i)=>formatColumn(s,h,i,start+1,end));
  if(ds.headers.includes('raw_paths_d1_4'))s.getRange(`A${start+1}:${last}${end}`).format.autofitRows();
  const t=s.tables.add(`A${start}:${last}${end}`,true,`LedgerTable${++tableId}`);t.showFilterButton=true;
  s.freezePanes.freezeRows(start);s.freezePanes.freezeColumns(1);
  placements.push({sheet:s.name,table:tableId,startRow:start,columns:ds.headers,rows:ds.rows.length,source:ds.filename,records:ds.rows});
  return end;
}

title(sheets['전체source'],'SOXS 원자료 날짜별 전체 원장','미국 거래일 기준. USD 금액. 빈칸은 결측이며 실제 0과 구분됩니다.');
sheets['전체source'].getRange('A4').values=[['출처: 한국예탁결제원 SEIBRO 해외주식 결제 상위50 원자료와 기존 고정 연구 가격판. https://seibro.or.kr/']];
sheets['전체source'].getRange('A5').values=[['시점 규약: 다음 한국 결제영업일 09:00 KST를 사용합니다. 과거 실제 공개시각을 확인한 기록은 아닙니다.']];
table(sheets['전체source'],source,7);
title(sheets['전체entry'],'최신 공개자료를 연결한 미국 진입일별 전체 원장','같은 entry의 여러 source 중 최신 source 하나를 유지합니다. source 원자료 금액의 합계와 섞지 않습니다.');
table(sheets['전체entry'],entry,6);
title(sheets['월별'],'월별 신호 횟수','source 날짜별 횟수와 최신 source를 사용한 entry 날짜별 횟수를 분리했습니다. 금액 합계가 아닙니다.');
table(sheets['월별'],monthly,6,{friendly:true});
title(sheets['2023준비'],'2023년 순위 준비 입력 전체','250행은 2024~2026 신호의 인과적 순위 준비용 입력입니다. 2023년 성과를 평가하거나 합격 판정에 사용하지 않습니다.');
table(sheets['2023준비'],warmup,6);

const recentSource=source.rows.filter(r=>r.source_date instanceof Date && r.source_date.toISOString().slice(0,10)>='2026-09-01').toReversed();
const recentEntry=entry.rows.filter(r=>r.entry instanceof Date && r.entry.toISOString().slice(0,10)>='2026-09-01').toReversed();
const sourceRecent=pick(source,['source_date','SOXS_buy','SOXS_sell','SOXS_net','gross_usd_L1_P','gross_usd_L1_status','gross_usd_L5','gross_usd_L5_P','gross_usd_L5_status','known_settle_kst','entry_settle','selected_for_entry','selection_reason'],recentSource);
const entryRecent=pick(entry,['entry','source_date','gross_usd_L1','gross_usd_L1_P','gross_usd_L1_status','gross_usd_L5','gross_usd_L5_P','gross_usd_L5_status','previous_dd60_pct','R5_pct','R10_pct','R20_pct','R5_complete','R10_complete','R20_complete'],recentEntry);
title(sheets['최근일별'],'최근 모든 일별 기록','2026-09-01 이후 전 행. 최신 날짜부터. ON=당일 포함 최근250세션 P80 이상. UNKNOWN·미공개를 OFF로 치환하지 않습니다.');
sheets['최근일별'].getRange('A5').values=[['원자료 날짜별. 미공개 날짜도 포함합니다.']];
const recentSourceEnd=table(sheets['최근일별'],sourceRecent,6,{friendly:true});
sheets['최근일별'].getRange(`A${recentSourceEnd+3}`).values=[['미국 entry 날짜별. 실제 공개시각을 측정한 기록이 아닌 09:00 KST 공개 규약을 적용합니다.']];
table(sheets['최근일별'],entryRecent,recentSourceEnd+4,{friendly:true});
sheets['최근일별'].freezePanes.freezeRows(6);

const dictRows=[];
for(const [filename,meta] of Object.entries(dictionary.files||{}))for(const [column,info] of Object.entries(meta.columns||{}))dictRows.push({file:filename,column,ko:info.ko||column,description:info.description||''});
title(sheets['열설명'],'열 이름과 정의','전체 원장의 각 열을 원본 CSV 이름으로 찾아볼 수 있습니다.');
table(sheets['열설명'],{filename:'dictionary',headers:['file','column','ko','description'],rows:dictRows},6);
sheets['열설명'].getRange(`A7:A${dictRows.length+6}`).format.columnWidth=22;
sheets['열설명'].getRange(`B7:B${dictRows.length+6}`).format.columnWidth=35;
sheets['열설명'].getRange(`C7:C${dictRows.length+6}`).format.columnWidth=27;
sheets['열설명'].getRange(`D7:D${dictRows.length+6}`).format.columnWidth=105;
sheets['열설명'].getRange(`D7:D${dictRows.length+6}`).format.wrapText=true;
sheets['열설명'].getRange(`A7:D${dictRows.length+6}`).format.autofitRows();

title(sheets['요약'],'한국인 SOXS 매수 일별 기록','기준일 2026-11-05. 원자료 일자와 미국 진입 가능일을 나누어 읽는 기록입니다.');
const sy=sheets['요약'];sy.tabColor=colors.navy;base(sy,'A5:H29');
sy.getRange('A5:D5').values=[['원장','기록 수','첫 날짜','마지막 날짜']];
sy.getRange('A6:D9').values=[['원자료 날짜별',source.rows.length,source.rows[0].source_date,source.rows.at(-1).source_date],['미국 entry 날짜별',entry.rows.length,entry.rows[0].entry,entry.rows.at(-1).entry],['최근 source 날짜별',recentSource.length,recentSource.at(-1).source_date,recentSource[0].source_date],['최근 entry 날짜별',recentEntry.length,recentEntry.at(-1).entry,recentEntry[0].entry]].map(r=>r.map(excelValue));
sy.getRange('C6:D9').setNumberFormat('yyyy-mm-dd');
sy.getRange('A11:H11').values=[['정의','source ON','entry ON','source UNKNOWN','entry UNKNOWN','source 미공개','source ON 중 덮임','새 source 없는 entry']];
sy.getRange('A12:H17').values=keys.map(k=>{const c=summary.all_period_counts[k];return[labels[k],c.source_on,c.entry_latest_on,c.source_unknown,c.entry_unknown,c.source_unpublished,c.source_shadowed_on,c.entry_no_new_source];});
for(const r of ['A5:D5','A11:H11'])sy.getRange(r).format={fill:colors.navy,font:{name:font,size:10,bold:true,color:'#FFFFFF'},rowHeight:40,horizontalAlignment:'center',wrapText:true};
sy.getRange('A19').values=[['읽는 기준']];sy.getRange('A19').format.font={bold:true,color:colors.navy};
const notes=[
  'ON: 해당 정의의 당일 포함 최근 250세션 순위가 P80 이상인 날입니다. OFF: 유효한 자료에서 P80 미만입니다.',
  'UNKNOWN: 순위 준비 부족 또는 원자료 결측입니다. UNPUBLISHED: 아직 공개되지 않은 source 날짜입니다.',
  'NO_NEW_SOURCE: entry 날짜에 새로 연결할 공개자료가 없습니다. 빈칸 금액은 매수 0달러를 뜻하지 않습니다.',
  'source 원장은 미국 거래일 기준 원자료입니다. entry 원장은 같은 진입일에 최신 source 하나만 연결합니다.',
  '공개시각은 다음 한국 결제영업일 09:00 KST 규약입니다. 과거 실제 공개시각을 확인한 기록은 아닙니다.',
  'R5·R10·R20은 entry 시가 대비 SOXX 종가 반응입니다. 전 구간 가격이 완성되지 않으면 공란입니다.',
  '2024~2026을 본평가합니다. 2023준비 시트의 250행은 인과적 순위 준비 입력이며 성과 평가에서 제외합니다.'
  ,'unit_shared_source=TRUE는 합산 결제액을 거래일 수로 균등 배분한 행입니다. 실제 개별 일별 금액은 식별할 수 없습니다.'
];
notes.forEach((v,i)=>{sy.getRange(`A${20+i}`).values=[[v]];});
sy.getRange('A20:A27').format.rowHeight=26;
sy.getRange('A5:A27').format.columnWidth=25;sy.getRange('B5:H27').format.columnWidth=19;

// Reconcile every copied cell against the typed CSV values before export.
let compared=0;const mismatches=[];
for(const p of placements){const values=sheets[p.sheet].getRange(`A${p.startRow+1}:${col(p.columns.length-1)}${p.startRow+p.rows}`).values;for(let r=0;r<p.rows;r++)for(let c=0;c<p.columns.length;c++){const expected=p.records[r][p.columns[c]]??null;const actual=values[r][c]??null;const eq=expected instanceof Date?typeof actual==='number'?Math.abs(actual-(+expected/86400000+25569))<1e-7:+new Date(actual)===+expected : typeof expected==='number'?typeof actual==='number'&&Math.abs(actual-expected)<=Math.max(1e-10,Math.abs(expected)*1e-12):actual===expected;if(!eq&&mismatches.length<20)mismatches.push({sheet:p.sheet,row:r+p.startRow+1,column:p.columns[c],expected:String(expected),actual});compared++;}}
if(mismatches.length)throw new Error(JSON.stringify({message:'CSV cell mismatch',compared,mismatches}));

await wb.recalculate();
const inspections=[];
for(const name of Object.keys(sheets))inspections.push({sheet:name,inspection:(await wb.inspect({kind:'table',range:`'${name}'!A1:L12`,include:'values,formulas',tableMaxRows:12,tableMaxCols:12,maxChars:4500})).ndjson});
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},summary:'final formula error scan',maxChars:3000});
await fs.mkdir(path.join(outputDir,'qa'),{recursive:true});
await fs.writeFile(path.join(outputDir,'qa','inspections.json'),JSON.stringify(inspections,null,2));
const renderSheets=process.env.SOXS_LEDGER_RENDER_SHEETS?new Set(process.env.SOXS_LEDGER_RENDER_SHEETS.split(',')):null;
for(const [name,range]of [['요약','A1:H27'],['최근일별','A1:I17'],['전체source','A1:J16'],['전체entry','A1:J16'],['월별','A1:I16'],['2023준비','A1:J12'],['열설명','A1:D12']]){
  if(renderSheets&&!renderSheets.has(name))continue;
  const blob=await wb.render({sheetName:name,range,scale:1.4,format:'png'});
  await fs.writeFile(path.join(outputDir,'qa',`${name}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const out=await SpreadsheetFile.exportXlsx(wb);await out.save(path.join(outputDir,'SOXS_일별전체원장_20261105.xlsx'));
await fs.unlink(path.join(outputDir,'SOXS_일별전체원장_20261105.xlsx.inspect.ndjson')).catch(e=>{if(e.code!=='ENOENT')throw e;});
await fs.writeFile(path.join(outputDir,'qa','presentation_audit.json'),JSON.stringify({built_at:new Date().toISOString(),inputs:[source,entry,monthly,warmup].map(d=>({filename:d.filename,rows:d.rows.length,columns:d.headers.length,sha256:d.sha256})),copied_cells_compared:compared,mismatches:[],formula_error_scan:errors.ndjson,sheets:Object.keys(sheets),recent:{source:recentSource.length,entry:recentEntry.length},publication_timing:'next KSD business day 09:00 KST convention; actual historical publication timestamps unavailable',calculations_added:false},null,2));

// HTML is a local, read-only view of the same CSV rows and values.
const cleanRows=ds=>ds.rows.map(r=>Object.fromEntries(Object.entries(r).map(([k,v])=>[k,v instanceof Date?v.toISOString().slice(0,/kst|known_settle/.test(k)?16:10).replace('T',' '):v])));
const compact=ds=>({columns:ds.headers,rows:cleanRows(ds).map(r=>ds.headers.map(c=>r[c]??null))});
const payload={source:compact(source),entry:compact(entry),monthly:compact(monthly),dictionary:dictionary.files,keys,labels};
const json=JSON.stringify(gzipSync(Buffer.from(JSON.stringify(payload))).toString('base64'));
const html=`<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SOXS 일별 전체 원장</title><style>
:root{font-family:Arial,'Malgun Gothic',sans-serif;color:#203047;background:#f5f7fa;font-size:14px}*{box-sizing:border-box}body{margin:0}header,main{max-width:1500px;margin:auto;padding:24px 28px}header{padding-bottom:12px}h1{font-size:28px;letter-spacing:-.6px;margin:0 0 9px}p{line-height:1.65;margin:6px 0}.sub{color:#5d6d82}.note{background:#fff2d8;border-left:4px solid #d6a44b;padding:12px 16px;margin:16px 0;font-size:13px}.tabs{display:flex;gap:8px;margin:12px 0}.tabs button{border:1px solid #c5ced9;background:white;padding:10px 16px;border-radius:7px;font-weight:700}.tabs button.active{background:#243c5a;color:white;border-color:#243c5a}.filters{display:flex;flex-wrap:wrap;align-items:end;gap:12px;padding:16px;background:white;border:1px solid #dce2e9;border-radius:9px}label{display:flex;flex-direction:column;gap:6px;font-size:12px;color:#526377}input,select,button{font:inherit;color:#203047}input,select{height:36px;padding:6px 9px;border:1px solid #bdc8d6;border-radius:5px;background:white}button{cursor:pointer}button.action{height:36px;background:#edf2f7;border:1px solid #c5ced9;border-radius:5px;padding:5px 12px}.meta{display:flex;justify-content:space-between;gap:15px;margin:16px 0 10px;align-items:center}.badge{display:inline-block;border-radius:4px;padding:4px 7px;font-size:12px;font-weight:700;background:#eef1f5}.ON{background:#d9eee8;color:#24584d}.UNKNOWN,.UNPUBLISHED{background:#fff0cb;color:#795717}.NO_NEW_SOURCE{background:#e9edf2;color:#526377}.legend{font-size:12px;display:flex;gap:10px;flex-wrap:wrap}.tablebox{overflow:auto;max-height:70vh;background:white;border:1px solid #dce2e9;border-radius:8px}table{border-collapse:separate;border-spacing:0;font-size:12px;white-space:nowrap;width:100%}th{position:sticky;top:0;background:#243c5a;color:white;padding:12px 13px;border-right:1px solid #4b617b;z-index:2;text-align:center;max-width:250px;white-space:normal;min-width:115px}td{padding:9px 13px;border-bottom:1px solid #edf1f5;text-align:right;height:39px}td:first-child,th:first-child{position:sticky;left:0;z-index:1;background:#f8fafc;text-align:left;border-right:1px solid #dce2e9;font-variant-numeric:tabular-nums}th:first-child{z-index:3;background:#243c5a}tbody tr:hover td{background:#eef4fa}td.text{text-align:left;max-width:360px;overflow:hidden;text-overflow:ellipsis}td.blank{color:#8e9aac}details{margin-top:20px;background:white;padding:14px 16px;border:1px solid #dce2e9;border-radius:8px}summary{cursor:pointer;font-weight:700}.definitions{columns:2;column-gap:32px;font-size:12px}.definitions p{break-inside:avoid;margin-bottom:9px}#empty{padding:28px;text-align:center;color:#68788d;display:none}@media(max-width:700px){header,main{padding:18px 14px}h1{font-size:22px}.definitions{columns:1}.meta{display:block}.legend{margin-top:10px}}
</style><header><h1>SOXS 매수 일별 전체 원장</h1><p class="sub">2026-11-05 기준. 매수·순매수·자기 매수비율의 1일·5일 정의를 같은 원장에서 확인합니다.</p><div class="note">공개시각은 <b>다음 한국 결제영업일 09:00 KST 규약</b>입니다. 과거 실제 공개시각을 측정한 자료는 없습니다. 원자료 날짜별 금액과 최신 원자료를 연결한 entry 날짜별 금액의 합계를 섞지 않습니다.</div></header><main><nav class="tabs" aria-label="원장 종류"><button data-tab="source" class="active">원자료 날짜별</button><button data-tab="entry">미국 entry 날짜별</button><button data-tab="monthly">월별 횟수</button></nav><div class="filters"><label>시작일<input type="date" id="from" value="2026-09-01"></label><label>종료일<input type="date" id="to" value="2026-11-05"></label><label>정의<select id="signal"></select></label><label>상태<select id="status"><option value="">모두</option><option>ON</option><option>OFF</option><option>UNKNOWN</option><option>UNPUBLISHED</option><option>NO_NEW_SOURCE</option></select></label><label>열 표시<select id="fields"><option value="main">주요 열</option><option value="all">모든 원본 열</option></select></label><label>결측 필터<select id="missing"><option value="">전체</option><option value="missing">결측·미공개만</option><option value="valid">유효 ON·OFF만</option></select></label><button class="action" id="recent">최근 9~11월</button><button class="action" id="all">전체 기간·모든 행</button></div><div class="meta"><div><b id="count"></b><p id="grain" class="sub"></p></div><div class="legend"><span class="badge ON">ON</span><span class="badge">OFF</span><span class="badge UNKNOWN">UNKNOWN</span><span class="badge UNPUBLISHED">미공개</span><span class="badge NO_NEW_SOURCE">새 자료 없음</span></div></div><div class="tablebox"><table id="ledger"><thead></thead><tbody></tbody></table><div id="empty">조건에 맞는 기록이 없습니다.</div></div><details><summary>정의와 열 설명</summary><p>ON은 선택한 정의의 직전 250세션 순위 P80 이상입니다. UNKNOWN은 결측 또는 순위 준비 부족입니다. 빈칸 금액은 매수 0달러를 뜻하지 않습니다. 가격 반응의 전 구간이 완성되지 않은 행은 반응값이 공란입니다.</p><div id="definitions" class="definitions"></div></details></main><script type="application/json" id="data">${json}</script><script>
const packed=JSON.parse(document.querySelector('#data').textContent), bytes=Uint8Array.from(atob(packed),c=>c.charCodeAt(0)), plain=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
const data=JSON.parse(plain),$=s=>document.querySelector(s);for(const name of ['source','entry','monthly']){const ds=data[name];ds.rows=ds.rows.map(r=>Object.fromEntries(ds.columns.map((c,i)=>[c,r[i]])));}let active='source';
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
for(const key of data.keys)$('#signal').insertAdjacentHTML('beforeend','<option value="'+key+'">'+esc(data.labels[key])+'</option>');
function def(c){return data.dictionary[(active==='source'?'source_daily':active==='entry'?'entry_daily':'monthly')+'.csv']?.columns?.[c]||{ko:c,description:''};}
function fmt(v,c){if(v===null||v==='')return '<span class="blank">—</span>';if(typeof v==='boolean')return v?'TRUE':'FALSE';if(typeof v==='number'){if(/_pct$/.test(c))return v.toFixed(2)+'%';if(/buy_share_L[15]$/.test(c))return(v*100).toFixed(1)+'%';if(/_P$/.test(c))return v.toFixed(1);if(/SOXS_(buy|sell|net|tot)$|gross_usd_L[15]$|net_usd_L[15]$/.test(c))return '$'+v.toLocaleString('en-US',{maximumFractionDigits:0});return v.toLocaleString('en-US',{maximumFractionDigits:4});}if(/_status$/.test(c))return '<span class="badge '+esc(v)+'">'+esc(v)+'</span>';return esc(v);}
function render(){const ds=data[active],key=$('#signal').value,from=$('#from').value,to=$('#to').value,st=$('#status').value,miss=$('#missing').value,dateCol=active==='source'?'source_date':active==='entry'?'entry':'month';let rows=ds.rows.filter(r=>{const d=String(r[dateCol]||''),s=r[key+'_status'];return(!from||d.slice(0,10)>=from.slice(0,active==='monthly'?7:10))&&(!to||d.slice(0,10)<=to.slice(0,active==='monthly'?7:10))&&(active==='monthly'||!st||s===st)&&(active==='monthly'||!miss||(miss==='missing'?!['ON','OFF'].includes(s):['ON','OFF'].includes(s)))&&(active!=='monthly'||r.key===key);}).toReversed();let cols=$('#fields').value==='all'?ds.columns:active==='source'?['source_date','SOXS_buy','SOXS_sell','SOXS_net',key,key+'_P',key+'_status','known_settle_kst','entry_settle','selected_for_entry','selection_reason']:active==='entry'?['entry','source_date',key,key+'_P',key+'_status','previous_dd60_pct','R5_pct','R10_pct','R20_pct',key+'_transition',key+'_on_streak_sessions']:ds.columns;cols=cols.filter(c=>ds.columns.includes(c));$('#ledger thead').innerHTML='<tr>'+cols.map(c=>'<th title="'+esc(c+' — '+def(c).description)+'">'+esc(def(c).ko)+'</th>').join('')+'</tr>';$('#ledger tbody').innerHTML=rows.map(r=>'<tr>'+cols.map(c=>'<td class="'+(typeof r[c]==='string'&&!/^\\d{4}-/.test(r[c])?'text':'')+'" title="'+esc(r[c]??'결측')+'">'+fmt(r[c],c)+'</td>').join('')+'</tr>').join('');$('#count').textContent=rows.length.toLocaleString()+'행 표시 / 전체 '+ds.rows.length.toLocaleString()+'행';$('#grain').textContent=active==='source'?'미국 source 거래일 기준. 최신 날짜부터.':active==='entry'?'공개 규약을 적용한 미국 entry 날짜. 같은 entry의 최신 source 하나.':'source 기준과 entry 기준의 월별 신호 횟수.';$('#empty').style.display=rows.length?'none':'block';$('#definitions').innerHTML=cols.map(c=>'<p><b>'+esc(def(c).ko)+'</b> ('+esc(c)+')<br>'+esc(def(c).description)+'</p>').join('');window.ledgerState={active,rows:rows.length,total:ds.rows.length,columns:cols,first:rows[0]?.[dateCol],last:rows.at(-1)?.[dateCol]};}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{active=b.dataset.tab;document.querySelectorAll('[data-tab]').forEach(t=>t.classList.toggle('active',t===b));render();});document.querySelectorAll('.filters input,.filters select').forEach(e=>e.onchange=render);$('#recent').onclick=()=>{$('#from').value='2026-09-01';$('#to').value='2026-11-05';$('#status').value='';$('#missing').value='';render();};$('#all').onclick=()=>{$('#from').value='';$('#to').value='';$('#status').value='';$('#missing').value='';render();};render();
</script></html>`;
const themeCss=`:root{background:var(--background,#f5f7fa);color:var(--foreground,#203047)}.sub,label{color:var(--muted-foreground,#5d6d82)}.filters,.tablebox,details,.tabs button,input,select{background:var(--card,#fff);color:var(--foreground,#203047);border-color:var(--border,#dce2e9)}button.action{background:var(--secondary,#edf2f7);color:var(--secondary-foreground,#203047);border-color:var(--border,#c5ced9)}.tabs button.active,th,th:first-child{background:var(--primary,#243c5a);color:var(--primary-foreground,#fff)}.tablebox{max-height:580px}td:first-child{background:var(--card,#f8fafc)}tbody tr:hover td{background:var(--accent,#eef4fa);color:var(--accent-foreground,#203047)}.note{background:var(--secondary,#fff2d8);color:var(--secondary-foreground,#203047)}td{border-color:var(--border,#edf1f5)}.badge{background:var(--muted,#eef1f5);color:var(--muted-foreground,#526377)}.ON{background:color-mix(in srgb,var(--chart-2,#29826b) 20%,var(--card,#fff));color:var(--foreground,#203047)}.UNKNOWN,.UNPUBLISHED{background:color-mix(in srgb,var(--chart-4,#d6a44b) 22%,var(--card,#fff));color:var(--foreground,#203047)}.recordgrid{display:grid;grid-template-columns:minmax(150px,1fr) minmax(200px,3fr);gap:1px;background:var(--border,#dce2e9);margin-top:12px;font-size:12px}.recordgrid div{padding:9px;background:var(--card,#fff);overflow-wrap:anywhere}.recordgrid div:nth-child(odd){font-weight:700}.tablebox tbody tr{cursor:pointer}`;
let finalHtml=html.replace('</style>',themeCss+'</style>').replace('<script>\nconst packed','<script type="module">\nconst packed').replace('max-height:70vh','max-height:580px').replace('window.ledgerState={','window.currentRows=rows;window.ledgerState={').replace('<details><summary>정의와 열 설명</summary>','<details id="record"><summary>하루를 선택하면 전체 열을 표시합니다</summary><div id="recordbody" class="recordgrid"></div></details><details><summary>정의와 열 설명</summary>').replace("document.querySelectorAll('[data-tab]')",`$('#ledger').onclick=e=>{const tr=e.target.closest('tbody tr');if(!tr)return;const idx=Array.from(tr.parentNode.children).indexOf(tr),row=window.currentRows[idx];$('#record summary').textContent=String(row[active==='source'?'source_date':active==='entry'?'entry':'month'])+' 전체 열';$('#recordbody').innerHTML=data[active].columns.map(c=>'<div>'+esc(def(c).ko)+'<br><span class="sub">'+esc(c)+'</span></div><div>'+fmt(row[c],c)+'</div>').join('');$('#record').open=true;};document.querySelectorAll('[data-tab]')`);
finalHtml=finalHtml.replace('<option value="missing">결측·미공개만</option>','<option value="missing">결측·미공개만</option><option value="censored">TOP50 가려짐·부분절단</option>').replace("(miss==='missing'?!['ON','OFF'].includes(s):['ON','OFF'].includes(s))","(miss==='censored'?(r.cens_SOXS===true||r.part_SOXS===true):miss==='missing'?!['ON','OFF'].includes(s):['ON','OFF'].includes(s))");
finalHtml=finalHtml.replaceAll('d.slice(0,10)>=','d.slice(0,active===\'monthly\'?7:10)>=').replaceAll('d.slice(0,10)<=','d.slice(0,active===\'monthly\'?7:10)<=');
finalHtml=finalHtml.replace('원자료 날짜별 금액과 최신 원자료를 연결한 entry 날짜별 금액의 합계를 섞지 않습니다.','원자료 날짜별 금액과 최신 원자료를 연결한 entry 날짜별 금액의 합계를 섞지 않습니다. 합산 결제 묶음은 unit_n_trades로 균등 배분했으며 실제 개별 일별 금액을 식별한 값이 아닙니다. unit_shared_source·unit_id·raw_paths_d1_4로 확인할 수 있습니다.');
finalHtml=finalHtml.replaceAll('직전 250세션','당일 포함 최근 250세션');
finalHtml=finalHtml.replace('function render(){const ds=',"function render(){$('#record').open=false;$('#record summary').textContent='하루를 선택하면 전체 열을 표시합니다';$('#recordbody').innerHTML='';const ds=");
await fs.writeFile(path.join(outputDir,'SOXS_일별전체원장_20261105.html'),finalHtml,'utf8');
if(Buffer.byteLength(finalHtml)>=512*1024)throw new Error(`Inline HTML exceeds 512KB: ${Buffer.byteLength(finalHtml)}`);
const scratch=process.env.ORCA_CHAT_SCRATCH_DIR;
if(scratch){await fs.writeFile(path.join(scratch,'soxs-daily-ledger-20261105-c7b2.html'),finalHtml,'utf8');}
console.log(JSON.stringify({outputDir,source_rows:source.rows.length,entry_rows:entry.rows.length,monthly_rows:monthly.rows.length,copied_cells_compared:compared,recent_source:recentSource.length,recent_entry:recentEntry.length,formula_error_scan:errors.ndjson}));
