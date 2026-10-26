import json, sys
d=json.load(open(f'../results/{sys.argv[1]}.json',encoding='utf-8'))
g=lambda w,h,k: (w or {}).get(h,{}).get(k)
fmt=lambda v: f"{v:+.3f}" if isinstance(v,(int,float)) else str(v)
for r in d['rows']:
    R=r.get('result') or {}; f=R.get('front',{}); b=R.get('back',{}); y=R.get('y2025',{}); l=R.get('late',{})
    if not f: continue
    print(f"{r['name'][:22]:22s} fIC20 {fmt(g(f,'h20','ic'))} top {fmt(g(f,'h20','top20'))} | bIC20 {fmt(g(b,'h20','ic'))} top {fmt(g(b,'h20','top20'))} | 25 {fmt(g(y,'h20','ic'))} 24 {fmt(g(R.get('y2024'),'h20','ic'))} | late10 {fmt(g(l,'h10','ic'))} | turn {f.get('turnover')} | S f {f.get('port_S',{}).get('ret')} b {b.get('port_S',{}).get('ret')} | la {r.get('no_lookahead')}")
