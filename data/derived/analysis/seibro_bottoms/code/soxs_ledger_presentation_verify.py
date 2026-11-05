import base64, csv, gzip, json, math, pathlib, re, zipfile, xml.etree.ElementTree as ET
from datetime import datetime

# Read-only standard-library verification. Run with py <this file> from any cwd.
REPO=next(p for p in pathlib.Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
OUT=REPO/'outputs/soxs_ledger_20261105'
SRC=REPO/'data/derived/analysis/seibro_bottoms/results/soxs_daily_ledger_20261105'
N={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
book=OUT/'SOXS_일별전체원장_20261105.xlsx'
z=zipfile.ZipFile(book)
shared=[]
if 'xl/sharedStrings.xml' in z.namelist():
    shared=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',N)]
root=ET.fromstring(z.read('xl/workbook.xml'))
names=[x.attrib['name'] for x in root.findall('s:sheets/s:sheet',N)]
checks={'xlsx_bytes':book.stat().st_size,'sheet_names':names,'tables':0,'csv_cells_compared':0,'mismatches':[],'panes':{},'date_cells':0,'booleans':0}

def cell_value(cell):
    kind=cell.get('t');v=cell.find('s:v',N)
    if kind=='s':return shared[int(v.text)]
    if kind=='inlineStr':return ''.join(cell.find('s:is',N).itertext())
    if v is None:return None
    if kind=='b':return v.text=='1'
    if kind in ('str','d'):return v.text
    return float(v.text)

def column(i):
    result='';i+=1
    while i:i,r=divmod(i-1,26);result=chr(65+r)+result
    return result

for i,name in enumerate(names,1):
    sheet=ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml'))
    pane=sheet.find('s:sheetViews/s:sheetView/s:pane',N)
    checks['panes'][name]=pane.attrib if pane is not None else None
    checks['tables']+=len(sheet.findall('s:tableParts/s:tablePart',N))
    if name not in ('전체source','전체entry','월별','2023준비'):continue
    fname,start={'전체source':('source_daily.csv',8),'전체entry':('entry_daily.csv',7),'월별':('monthly.csv',7),'2023준비':('warmup_daily.csv',7)}[name]
    cells={c.attrib['r']:c for c in sheet.findall('s:sheetData/s:row/s:c',N)}
    with (SRC/fname).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);headers=reader.fieldnames
        for row_i,row in enumerate(reader,start):
            for j,key in enumerate(headers):
                raw=row[key];cell=cells.get(f'{column(j)}{row_i}');got=cell_value(cell) if cell is not None else None
                checks['csv_cells_compared']+=1
                if raw=='':expected=None;ok=got is None
                elif key=='unit_id':expected=raw;ok=got==expected
                elif raw.lower() in ('true','false'):
                    expected=raw.lower()=='true';ok=got is expected;checks['booleans']+=1
                else:
                    try:expected=float(raw);ok=isinstance(got,(int,float)) and not isinstance(got,bool) and math.isclose(got,expected,rel_tol=1e-12,abs_tol=1e-10)
                    except ValueError:
                        if raw[:4].isdigit() and raw[4:5]=='-' and (key in ('entry','month','entry_settle','known_settle','unit_first','unit_last','selected_source_for_same_entry') or key.endswith(('date','_kst','_end'))):
                            dt=datetime.fromisoformat(raw+'-01' if len(raw)==7 else raw).replace(tzinfo=None)
                            expected=(dt-datetime(1899,12,30)).total_seconds()/86400
                            ok=isinstance(got,(int,float)) and math.isclose(got,expected,abs_tol=1e-7)
                            checks['date_cells']+=1
                        else:expected=raw;ok=got==expected
                if not ok and len(checks['mismatches'])<20:checks['mismatches'].append({'sheet':name,'cell':f'{column(j)}{row_i}','column':key,'expected':expected,'actual':got})
checks['formula_cells']=sum(len(ET.fromstring(z.read(n)).findall('.//s:f',N)) for n in z.namelist() if n.startswith('xl/worksheets/sheet') and n.endswith('.xml'))
checks['native_table_filters']=all(ET.fromstring(z.read(n)).find('s:autoFilter',N) is not None for n in z.namelist() if n.startswith('xl/tables/table') and n.endswith('.xml'))
assert checks['tables']==7,checks
assert checks['native_table_filters'],checks
assert len(names)==7,checks
assert not checks['mismatches'],checks
for name,pane in checks['panes'].items():
    if name!='요약':assert pane and pane.get('state')=='frozen',checks
html=(OUT/'SOXS_일별전체원장_20261105.html').read_text(encoding='utf-8')
compressed=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',html,re.S).group(1))
payload=json.loads(gzip.decompress(base64.b64decode(compressed)))
checks['html_bytes']=len(html.encode('utf-8'))
checks['html_csv_cells_compared']=0
checks['html_mismatches']=[]
for key,fname in [('source','source_daily.csv'),('entry','entry_daily.csv'),('monthly','monthly.csv')]:
    ds=payload[key]
    with (SRC/fname).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==ds['columns']
        raw_rows=list(reader)
    assert len(raw_rows)==len(ds['rows'])
    for i,(raw_row,values) in enumerate(zip(raw_rows,ds['rows'])):
        for column,actual in zip(ds['columns'],values):
            raw=raw_row[column];checks['html_csv_cells_compared']+=1
            if raw=='':expected=None;ok=actual is None
            elif column=='unit_id':expected=raw;ok=actual==expected
            elif raw.lower() in ('true','false'):expected=raw.lower()=='true';ok=actual is expected
            else:
                try:expected=float(raw);ok=isinstance(actual,(int,float)) and not isinstance(actual,bool) and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-10)
                except ValueError:
                    if column.endswith('_kst'):expected=raw[:16].replace('T',' ');ok=actual==expected
                    elif column=='month':expected=raw+'-01';ok=actual==expected
                    else:expected=raw;ok=actual==expected
            if not ok and len(checks['html_mismatches'])<20:checks['html_mismatches'].append({'file':fname,'row':i,'column':column,'expected':expected,'actual':actual})
assert checks['html_bytes']<512*1024
assert not checks['html_mismatches'],checks
assert "function render(){$('#record').open=false;" in html
assert '당일 포함 최근 250세션' in html
assert '100vh' not in html and '70vh' not in html
(OUT/'qa/export_audit.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False))
