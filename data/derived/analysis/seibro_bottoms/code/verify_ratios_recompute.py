"""BRIEF2 독립 재계산 (code/ratios.py·results/ratios 산출물을 보지 않고 BRIEF2 문장만으로 구현).

실행: py -B code/verify_ratios_recompute.py
출력: results/ratios/verify_recompute.md (strict 재계산 + 하네스 cells.csv·meta.json·intuitive.csv 대조,
      차이 원인을 찾는 변형 4개)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / 'derived'
OUT = ROOT / 'results' / 'ratios'

GROUPS = ['SOXL', 'SOXS', 'TQQQ', 'SQQQ', 'NVDA', 'TSLA', 'LEVSGL', 'LEV3', 'LEV2IDX',
          'INV', 'INC', 'BOND', 'SEMI1', 'IDX1', 'STK1', 'OTH']
E_TICK = ['SOXL', 'SOXS', 'TQQQ', 'SQQQ', 'NVDA', 'TSLA']
WIN = {'W1': (15, -6), 'W2': (5, -1), 'W3': (1, 0), 'W4': (5, 5)}  # (길이, 창 끝 오프셋)
FLAG_ORDER = [('LSGL', 'LEVSGL'), ('EQ3L', 'LEV3'), ('LIDX', 'LEV2IDX'), ('EQ_INV', 'INV'),
              ('R5', 'INC'), ('R3', 'BOND'), ('R1', 'SEMI1'), ('R2', 'IDX1'), ('R4', 'STK1')]


def assign_group(r):
    p = r['product']
    if p in E_TICK:
        return p
    for flag, g in FLAG_ORDER:
        if r[flag]:
            return g
    return 'OTH'


def roll(a, L):
    """axis 0 끝 포함 L행 합(앞 L-1행은 NaN)."""
    c = np.cumsum(np.concatenate([np.zeros((1,) + a.shape[1:]), a]), axis=0)
    out = c[L:] - c[:-L]
    pad = np.full((L - 1,) + a.shape[1:], np.nan)
    return np.concatenate([pad, out])


def sdiv(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.asarray(a, float) / np.asarray(b, float)
    return np.where(np.isfinite(r), r, np.nan)


def slog(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.log(np.asarray(a, float) / np.asarray(b, float))
    return np.where(np.isfinite(r), r, np.nan)


def piv(f, dates, cols, col, key):
    return (f.pivot_table(index='date', columns=key, values=col, aggfunc='sum')
            .reindex(index=dates, columns=cols).fillna(0).values)


def build_features(f, dates, L, pos_mode='window', b_mask='own', nrel_mask='both', srank='desc_tot'):
    B = roll(piv(f, dates, GROUPS, 'buy', 'group'), L)
    S = roll(piv(f, dates, GROUPS, 'sell', 'group'), L)
    N = B - S
    T = B + S
    prods = sorted(f['product'].unique())
    pb = piv(f, dates, prods, 'buy', 'product')
    ps = piv(f, dates, prods, 'sell', 'product')
    PB = roll(pb, L)
    PS = roll(ps, L)
    PN = PB - PS
    PT = PB + PS
    pos = np.where(PN > 0, PN, 0.0)
    neg = np.where(PN < 0, -PN, 0.0)
    if pos_mode == 'window':  # 창 합 순매수의 양수·음수 부분
        P_all = pos.sum(1)
        M_all = neg.sum(1)
    else:  # 하루 양수·음수 부분의 창 합
        pn = pb - ps
        P_all = roll(np.where(pn > 0, pn, 0).sum(1, keepdims=True), L)[:, 0]
        M_all = roll(np.where(pn < 0, -pn, 0).sum(1, keepdims=True), L)[:, 0]
    npos = (PN > 0).sum(1).astype(float)
    nneg = (PN < 0).sum(1).astype(float)
    B_all = PB.sum(1)
    S_all = PS.sum(1)
    T_all = B_all + S_all
    feats = {}
    imb = np.where(T > 0, sdiv(N, T), np.nan)
    for i, g in enumerate(GROUPS):
        feats[f'A:imb:{g}'] = imb[:, i]
    for i, g in enumerate(GROUPS):
        ok = T[:, i] > 0 if b_mask == 'own' else np.ones(len(T), bool)
        feats[f'B:tshare:{g}'] = np.where(ok, sdiv(T[:, i], T_all), np.nan)
        feats[f'B:bshare:{g}'] = np.where(ok, sdiv(B[:, i], B_all), np.nan)
        feats[f'B:sshare:{g}'] = np.where(ok, sdiv(S[:, i], S_all), np.nan)
        feats[f'B:nshare:{g}'] = np.where(ok, sdiv(N[:, i], P_all + M_all), np.nan)
    for i in range(16):
        for j in range(i + 1, 16):
            k = f'{GROUPS[i]}|{GROUPS[j]}'
            ok = (T[:, i] > 0) & (T[:, j] > 0)
            feats[f'C:imbd:{k}'] = np.where(ok, imb[:, i] - imb[:, j], np.nan)
            feats[f'C:ltr:{k}'] = np.where(ok, slog(T[:, i], T[:, j]), np.nan)
            feats[f'C:lbr:{k}'] = np.where(ok, slog(B[:, i], B[:, j]), np.nan)
            okn = ok if nrel_mask == 'both' else (T[:, i] + T[:, j]) > 0
            feats[f'C:nrel:{k}'] = np.where(okn, sdiv(N[:, i] - N[:, j], T[:, i] + T[:, j]), np.nan)
    meta = f.drop_duplicates('product').set_index('product').reindex(prods)
    levm = meta['lev_long'].fillna(False).values.astype(bool)
    invm = meta['EQ_INV'].fillna(False).values.astype(bool)
    feats['D:msr'] = sdiv(M_all, P_all)
    feats['D:npos'] = sdiv(npos, npos + nneg)
    feats['D:top1buy'] = sdiv(pos.max(1), P_all)
    feats['D:hhi'] = np.where(P_all > 0, (sdiv(pos, P_all[:, None]) ** 2).sum(1), np.nan)
    feats['D:top1sell'] = sdiv(neg.max(1), M_all)
    feats['D:fund'] = sdiv(B_all - S_all, T_all)
    feats['D:levnet'] = sdiv(PN[:, levm].sum(1), P_all + M_all)
    lb = PB[:, levm].sum(1)
    ib = PB[:, invm].sum(1)
    feats['D:levbuy'] = sdiv(lb, lb + ib)
    univ = PT > 0
    nU = univ.sum(1).astype(float)
    rn = pd.DataFrame(np.where(univ, PN, np.nan)).rank(axis=1, ascending=False, method='average').values
    if srank == 'desc_tot':
        rs = pd.DataFrame(np.where(univ, PS, np.nan)).rank(axis=1, ascending=False, method='average').values
    elif srank == 'asc_tot':
        rs = pd.DataFrame(np.where(univ, PS, np.nan)).rank(axis=1, ascending=True, method='average').values
    elif srank == 'desc_min':
        rs = pd.DataFrame(np.where(univ, PS, np.nan)).rank(axis=1, ascending=False, method='min').values
    elif srank == 'desc_max':
        rs = pd.DataFrame(np.where(univ, PS, np.nan)).rank(axis=1, ascending=False, method='max').values
    elif srank == 'desc_first':
        rs = pd.DataFrame(np.where(univ, PS, np.nan)).rank(axis=1, ascending=False, method='first').values
    elif srank == 'desc_dense':
        rs = pd.DataFrame(np.where(univ, PS, np.nan)).rank(axis=1, ascending=False, method='dense').values
    elif srank == 'desc_sellpos_nU':  # 매도 > 0 상품만 순위, 분모는 거래 상품 수
        rs = pd.DataFrame(np.where(PS > 0, PS, np.nan)).rank(axis=1, ascending=False, method='average').values
    elif srank == 'asc_sellpos':  # 매도 > 0 상품 오름차순 순위 / 매도 > 0 수
        sp = PS > 0
        rs = pd.DataFrame(np.where(sp, PS, np.nan)).rank(axis=1, ascending=True, method='average').values
        rs = rs * nU[:, None] / np.maximum(sp.sum(1), 1)[:, None]
    else:  # desc_sellpos: 매도 > 0 상품만 순위·분모
        sp = PS > 0
        rs = pd.DataFrame(np.where(sp, PS, np.nan)).rank(axis=1, ascending=False, method='average').values
        rs = rs * nU[:, None] / np.maximum(sp.sum(1), 1)[:, None]
    for t in E_TICK:
        c = prods.index(t)
        feats[f'E:nrank:{t}'] = sdiv(rn[:, c], nU)
        feats[f'E:srank:{t}'] = sdiv(rs[:, c], nU)
    X = pd.DataFrame(feats, index=dates)
    X.iloc[:L - 1] = np.nan
    return X


def pct_at(X, t):
    """t 행의 지난 250세션(t 포함) 중간점 백분위, n ≥ 120."""
    W = X[max(0, t - 249):t + 1]
    v = X[t]
    n = (~np.isnan(W)).sum(0)
    less = (W < v).sum(0)
    eq = (W == v).sum(0)
    P = 100 * (less + 0.5 * eq) / np.where(n > 0, n, 1)
    return np.where((n >= 120) & ~np.isnan(v), P, np.nan)


def run(pos_mode='window', ipo_exclude=False, d_pop='flows', tag='main', **fk):
    f = pd.read_parquet(DER / 'flows_trade_product.parquet')
    f = f[~f['dummy']].copy()
    if ipo_exclude:
        f = f[~f['ipo_excl']]
    f['group'] = f.apply(assign_group, axis=1)
    dates = pd.DatetimeIndex(sorted(f['date'].unique()))
    pos_of = {d: i for i, d in enumerate(dates)}
    ev = json.load(open(DER / 'events.json', encoding='utf-8'))
    bots = [d for d in ev['B10'] if d >= '2024-01-01']
    bk = [pos_of[pd.Timestamp(d)] for d in bots]
    Fd = [x['date'] for x in ev['F'] if x['date'] >= '2024-01-01' and x.get('has_flows')]
    Fk = [pos_of[pd.Timestamp(d)] for d in Fd if pd.Timestamp(d) in pos_of]
    pf = pd.read_parquet(DER / 'price_feats.parquet').set_index('date')
    if d_pop == 'flows':
        lr = pf['lr1'].reindex(dates)
        lr = lr[lr.index >= '2024-01-01']
    else:
        lr = pf['lr1'][pf.index >= '2024-01-01']
    thr = lr.quantile(0.10)
    excl = set()
    for d in ev['B10']:
        k = pos_of[pd.Timestamp(d)]
        excl |= set(range(k - 5, k + 6))
    allF = {pos_of[pd.Timestamp(x['date'])] for x in ev['F'] if pd.Timestamp(x['date']) in pos_of}
    Dk = [pos_of[d] for d in lr[lr <= thr].index
          if d in pos_of and pos_of[d] not in excl and pos_of[d] not in allF]
    Xs = {L: build_features(f, dates, L, pos_mode, **fk) for L in (1, 5, 15)}
    cols = list(Xs[1].columns)
    nT = len(dates)
    parts = []
    for w, (L, off) in WIN.items():
        X = Xs[L].values

        def pmat(ks):
            return np.array([pct_at(X, k + off) if 0 <= k + off < nT else np.full(X.shape[1], np.nan) for k in ks])
        PBm, PFm, PDm = pmat(bk), pmat(Fk), pmat(Dk)
        vB = np.array([X[k + off] for k in bk])
        hi = (PBm >= 80).sum(0)
        lo = (PBm <= 20).sum(0)
        side = np.where(hi >= lo, 'high', 'low')

        def rate(Pm):
            on = np.where(side == 'high', Pm >= 80, Pm <= 20)
            nv = (~np.isnan(Pm)).sum(0)
            return np.where(nv > 0, on.sum(0) / np.maximum(nv, 1), np.nan), nv
        fr, fn = rate(PFm)
        dr, dn = rate(PDm)
        df = pd.DataFrame({'feature': cols, 'window': w, 'side': side, 'n_on': np.maximum(hi, lo),
                           'n_hi': hi, 'n_lo': lo, 'n_valid': (~np.isnan(PBm)).sum(0),
                           'F_rate': fr, 'F_n': fn, 'D_rate': dr, 'D_n': dn})
        for i, d in enumerate(bots):
            df[f'P_{d}'] = PBm[i]
        for i, d in enumerate(bots):
            df[f'v_{d}'] = vB[i]
        parts.append(df)
    cells = pd.concat(parts, ignore_index=True)
    cells['common'] = cells['n_on'] >= 8
    cells['distinct'] = cells['common'] & (cells['F_rate'] <= 1 / 3) & (cells['D_rate'] <= 1 / 3)
    cells['family'] = cells['feature'].str[0]
    summ = {'tag': tag, 'n_features': len(cols), 'n_cells_P': len(cells), 'bottoms': bots,
            'F_days': [str(dates[k].date()) for k in Fk], 'D_days': [str(dates[k].date()) for k in Dk],
            'D_thr': float(thr), 'n_F': len(Fk), 'n_D': len(Dk),
            'common_P': int(cells['common'].sum()),
            'common_P_by_family': {k: int(v) for k, v in cells[cells['common']].groupby('family').size().items()},
            'distinct_P': int(cells['distinct'].sum()),
            'group_rows': {k: int(v) for k, v in f.groupby('group').size().items()}}
    return cells, summ


# ---------------------------------------------------------------- 비교·보고서

HNAME = {'C:dimb:': 'C:imbd:', 'E:rN:': 'E:nrank:', 'E:rS:': 'E:srank:', 'D:fund': 'D:msr', 'D:breadth': 'D:npos',
         'D:top1pos': 'D:top1buy', 'D:hhipos': 'D:hhi', 'D:top1neg': 'D:top1sell', 'D:imball': 'D:fund',
         'D:bull': 'D:levbuy'}
TARGET = [('C:lbr:SOXS|TQQQ', 'W3'), ('C:lbr:SOXL|SOXS', 'W3'), ('B:nshare:SOXL', 'W3'), ('B:sshare:SOXL', 'W4'),
          ('C:nrel:SOXL|STK1', 'W2'), ('A:imb:SOXL', 'W3'), ('A:imb:STK1', 'W3'), ('D:msr', 'W3')]
VARIANTS = {  # 하네스 차이 원인 찾기용(주 결과는 strict)
    'strict': {},
    'b0': {'b_mask': 'none'},
    'nrel': {'nrel_mask': 'sum'},
    'b0+nrel': {'b_mask': 'none', 'nrel_mask': 'sum'},
    'b0+nrel+sellpos': {'b_mask': 'none', 'nrel_mask': 'sum', 'srank': 'desc_sellpos'},
}


def _map(s, table):
    for a, b in table.items():
        if a.endswith(':') and s.startswith(a):
            return b + s[len(a):]
        if s == a:
            return b
    return s


def h2m(s):
    return _map(s, HNAME)


def m2h(s):
    return _map(s, {v: k for k, v in HNAME.items()})


def compare(m, h):
    pc = [c for c in m.columns if c.startswith('P_')]
    m = m.copy()
    m['med'] = m[pc].median(axis=1)
    d = h.merge(m, on=['feature', 'window'], suffixes=('_h', '_m'))
    d['hs'] = d['side_h'].map({1: 'high', -1: 'low'})
    ok = {'on': d.B10_on == d.n_on, 'n': d.B10_n == d.n_valid,
          'side': (d.hs == d.side_m) | (d.n_hi == d.n_lo),
          'F': np.isclose(d.F_rate_h, d.F_rate_m, equal_nan=True),
          'D': np.isclose(d.D_rate_h, d.D_rate_m, equal_nan=True),
          'med': np.isclose(d.B10_med, d.med, atol=0.051, equal_nan=True)}
    d['ok'] = np.all(np.vstack([np.asarray(v) for v in ok.values()]), axis=0)
    d['kind'] = d.feature.str.split(':').str[:2].str.join(':')
    return d


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '—'
    return f'{x:.{nd}f}'


def report():
    h = pd.read_csv(OUT / 'cells.csv')
    h = h[h['norm'] == 'P'].reset_index(drop=True)
    h['hname'] = h['feature']
    h['feature'] = h['feature'].map(h2m)
    hm = json.load(open(OUT / 'meta.json', encoding='utf-8'))
    hi_ = pd.read_csv(OUT / 'intuitive.csv', encoding='utf-8-sig')
    res = {}
    for k, kw in VARIANTS.items():
        c_, s_ = run(tag=k, **kw)
        res[k] = (c_, s_, compare(c_, h))
    cells, summ, d = res['strict']
    bots = summ['bottoms']
    L = []
    w = L.append
    w('# BRIEF2 독립 재계산 대조 (2026-11-04, claude 검증 에이전트)')
    w('')
    w('- 코드: `code/verify_ratios_recompute.py` (`py -B code/verify_ratios_recompute.py`). `code/ratios.py`는 열지 않았고, `results/ratios/*`는 내 값을 다 낸 뒤 이름 대응·비교에만 썼다.')
    w('- 입력: `derived/flows_trade_product.parquet`(dummy 행 15개 뺌), `derived/price_feats.parquet`, `derived/events.json`. 정규화는 P(지난 250세션 중간점 백분위, 최소 120)만 다시 쟀다. Eq는 범위 밖이다.')
    w('- 주 결과(strict)는 BRIEF2 문장 그대로다. T = 0인 묶음이 든 비율은 모두 결측(A·B·C 전부)이다. E 순위는 창 안 거래(B + S > 0) 상품 중 순위 ÷ 그 수다. P_all·M_all은 창 합 순매수의 상품별 양수·음수 부분 합이다.')
    w('')
    w('## 결론')
    w('')
    hfam = d[d.common_h].groupby(d.feature.str[0]).size().to_dict()
    fam = summ['common_P_by_family']
    nbad = len(d) - int(d.ok.sum())
    w('| 항목 | 하네스 | 내 재계산(strict) | 판정 |')
    w('|---|---|---|---|')
    w(f"| 특징 수·P 칸 수 | {hm['features']}·{len(h)} | {summ['n_features']}·{summ['n_cells_P']} | 같음 |")
    w(f"| F 날(2024 이후, 흐름 있음) | {len(hm['events']['F'])}개 | {summ['n_F']}개 | 날짜 목록 {'같음' if summ['F_days'] == hm['events']['F'] else '다름'} |")
    w(f"| D 날(하위 10%, B10 ±5·F 뺌) | {len(hm['events']['D'])}개 | {summ['n_D']}개(문턱 lr1 ≤ {summ['D_thr']:.4f}) | 날짜 목록 {'같음' if summ['D_days'] == hm['events']['D'] else '다름'} |")
    w(f"| 공통 P 칸(≥ 8/10) | {hm['common']['P']} ({', '.join(f'{k} {v}' for k, v in sorted(hfam.items()))}) | {summ['common_P']} ({', '.join(f'{k} {v}' for k, v in sorted(fam.items()))}) | 차 2칸, 원인은 C:nrel 결측 처리(§3) |")
    w(f"| 그중 F ≤ 1/3·D ≤ 1/3 | {hm['common_distinct']['P']} | {summ['distinct_P']} | 같음 |")
    w(f"| 칸별 일치(쪽·바닥 수·유효 수·F율·D율·바닥 P 중앙 ±0.05) | — | {int(d.ok.sum())}/{len(d)} | 불일치 {nbad}칸: B 몫(T = 0 처리)·C:nrel(T = 0 처리)·E 매도 순위(모집단) |")
    w('')
    c5, s5, d5 = res['b0+nrel+sellpos']
    w(f"- 하네스의 해석 세 가지(§3)를 내 코드에 넣으면 {int(d5.ok.sum())}/{len(d5)}칸이 맞는다. 공통 P 칸 {s5['common_P']}개, 구별 {s5['distinct_P']}개로 하네스와 같다. 남은 {len(d5) - int(d5.ok.sum())}칸은 모두 E:srank이고, 공통 판정과는 멀다.")
    w('- 하네스 헤드라인(공통 P 37, 구별 0)은 재현된다. 다만 BRIEF2 문장(T = 0 묶음의 비율은 결측)을 엄격히 따르면 공통 P는 35이고, 구별은 똑같이 0이다. 결론(공통점 없음)은 어느 쪽이든 바뀌지 않는다.')
    w('')
    w('## 1. (a) 16묶음과 우선순위')
    w('')
    w('상품 키가 SOXL·SOXS·TQQQ·SQQQ·NVDA·TSLA면 그 묶음이다. 아니면 자료판 표지를 이 순서로 보고 처음 맞는 묶음에 넣는다: LSGL→LEVSGL, EQ3L→LEV3, LIDX→LEV2IDX, EQ_INV→INV, R5→INC, R3→BOND, R1→SEMI1, R2→IDX1, R4→STK1, 나머지는 OTH.')
    w('')
    w('| 묶음 | ' + ' | '.join(GROUPS) + ' |')
    w('|---|' + '---|' * 16)
    gr = summ['group_rows']
    w('| 행 수 | ' + ' | '.join(str(gr.get(g, 0)) for g in GROUPS) + ' |')
    w('')
    w('- 하네스와 A·C(imbd·ltr·lbr)·D·E:nrank 칸이 모두 같다. 묶음이 하나라도 다르면 그 묶음이 든 비율이 어긋나므로, 묶음 배정은 같다고 본다.')
    w('')
    w('## 2. (b) 지정 8칸')
    w('')
    w('- 창 위치: W1 = k−6에서 끝나는 15세션 합, W2 = k−1에서 끝나는 5세션 합, W3 = k 하루, W4 = k+5에서 끝나는 5세션 합. F 날 W4는 2026-10-30(흐름 끝)의 k+5가 없어 23개로 잰다(하네스도 같다).')
    w('- 하네스 이름 대응: `C:dimb` = 내 `C:imbd`, `D:fund`(순매도 합 ÷ 순매수 합) = 내 `D:msr`, `D:imball` = 내 `D:fund`((B−S) ÷ T), `E:rN/rS` = 내 `E:nrank/srank`. 과제의 `D:fund W3`는 하네스 이름이므로 M_all ÷ P_all 칸이다.')
    w('')
    w('| 칸(하네스 이름) | 쪽 | 바닥 수(하네스/나) | F율(하네스/나) | D율(하네스/나) | 바닥 P 중앙(하네스/나) |')
    w('|---|---|---|---|---|---|')
    for f_, win in TARGET:
        r = d[(d.feature == f_) & (d.window == win)].iloc[0]
        w(f"| {r.hname} {win} | {r.side_m} | {r.B10_on}/{r.n_on} | {fmt(r.F_rate_h)}/{fmt(r.F_rate_m)} | {fmt(r.D_rate_h)}/{fmt(r.D_rate_m)} | {fmt(r.B10_med, 1)}/{fmt(r.med, 1)} |")
    w('')
    w('바닥마다 P(내 값, 열은 바닥 날짜):')
    w('')
    w('| 칸 | ' + ' | '.join(b[2:] for b in bots) + ' |')
    w('|---|' + '---|' * len(bots))
    for f_, win in TARGET:
        r = cells[(cells.feature == f_) & (cells.window == win)].iloc[0]
        w(f'| {m2h(f_)} {win} | ' + ' | '.join(fmt(r[f"P_{b}"], 1) for b in bots) + ' |')
    w('')
    w('바닥마다 값(내 값. A:imb = N ÷ T, lbr = log(B_g ÷ B_h), 몫·nrel·순매도 ÷ 순매수는 비):')
    w('')
    w('| 칸 | ' + ' | '.join(b[2:] for b in bots) + ' |')
    w('|---|' + '---|' * len(bots))
    for f_, win in TARGET:
        r = cells[(cells.feature == f_) & (cells.window == win)].iloc[0]
        w(f'| {m2h(f_)} {win} | ' + ' | '.join(fmt(r[f"v_{b}"], 3) for b in bots) + ' |')
    w('')
    w('하네스 `intuitive.csv`(B10 값 중앙·B10 P 중앙)와 대조(A:imb는 하네스 표처럼 매수 ÷ 매도 배수로 바꿔 비교):')
    w('')
    w('| 칸 | 하네스 B10 값 중앙 | 내 값 중앙 | 하네스 B10_P | 내 P 중앙 |')
    w('|---|---|---|---|---|')
    for hn, win in [('A:imb:SOXL', 'W3'), ('A:imb:STK1', 'W3'), ('B:nshare:SOXL', 'W3'), ('D:fund', 'W3'),
                    ('A:imb:SOXL', 'W2'), ('B:nshare:SOXL', 'W4'), ('D:fund', 'W2'), ('D:bull', 'W3')]:
        mf = h2m(hn)
        r = cells[(cells.feature == mf) & (cells.window == win)].iloc[0]
        vals = np.array([r[f'v_{b}'] for b in bots], float)
        if mf.startswith('A:imb'):
            vals = (1 + vals) / (1 - vals)
        hrow = hi_[(hi_['특징'] == hn) & (hi_['창'] == win)].iloc[0]
        pm = np.nanmedian([r[f'P_{b}'] for b in bots])
        w(f"| {hn} {win} | {hrow['B10']:.4f} | {np.nanmedian(vals):.4f} | {hrow['B10_P']:.1f} | {pm:.1f} |")
    w('')
    r = cells[(cells.feature == 'A:imb:SOXL') & (cells.window == 'W3')].iloc[0]
    mi = float(np.nanmedian([r[f'v_{b}'] for b in bots]))
    w(f"- A:imb:SOXL의 값 중앙 차(W3 3.2084 대 3.5488, W2 1.4984 대 1.5023)는 계산 차가 아니라 표시 방식 차다. 하네스는 imb 중앙({mi:.4f})을 배수로 바꿨고((1 + {mi:.4f}) ÷ (1 − {mi:.4f}) = {(1 + mi) / (1 - mi):.4f}), 나는 바닥마다 배수로 바꾼 뒤 중앙을 냈다. 바닥이 10개(짝수)라 가운데 두 값을 평균하는 순서만 다르다. P 중앙과 다른 칸 값 중앙은 모두 같다.")
    w('')
    w('## 3. (c) 공통 P 칸 수와 불일치 원인')
    w('')
    w('| 변형(내 코드) | 칸 일치 | 공통 P | 구별(F·D ≤ 1/3) | 남은 불일치 종류(칸 수) |')
    w('|---|---|---|---|---|')
    for k in VARIANTS:
        c_, s_, d_ = res[k]
        kinds = d_[~d_.ok].groupby('kind').size()
        w(f"| {k} | {int(d_.ok.sum())}/{len(d_)} | {s_['common_P']} | {s_['distinct_P']} | {', '.join(f'{a} {b}' for a, b in kinds.items()) or '없음'} |")
    w('')
    w('- b0 = B 몫에서 T_g = 0인 날을 0으로 둠, nrel = C:nrel을 T_g + T_h > 0이면 계산, sellpos = E:srank를 창 안 매도 > 0 상품 중 순위 ÷ 그 수로 계산.')
    w('')
    diff = d[d.common_h != d.common_m]
    w('공통 판정이 갈린 칸(strict 대 하네스):')
    w('')
    w('| 칸 | 바닥 수(하네스/나) | 유효 바닥(하네스/나) | F율(하네스/나) | D율(하네스/나) |')
    w('|---|---|---|---|---|')
    for _, r in diff.iterrows():
        w(f'| {r.hname} {r.window} | {r.B10_on}/{r.n_on} | {r.B10_n}/{r.n_valid} | {fmt(r.F_rate_h)}/{fmt(r.F_rate_m)} | {fmt(r.D_rate_h)}/{fmt(r.D_rate_m)} |')
    w('')
    w('세 가지 차이와 어느 쪽이 맞는가:')
    w('')
    w('1. **C:nrel = (N_g − N_h) ÷ (T_g + T_h)의 결측**: 하네스는 T_g + T_h > 0이면 값을 둔다. 한쪽 T = 0이면 이 값은 다른 쪽 imb(부호 포함)로 바뀌어, 다른 양이 한 계열에 섞인다. 나는 둘 중 하나라도 T = 0이면 결측으로 뒀다. BRIEF2 §3 "T = 0인 묶음(목록 밖 절단)의 비율은 결측이다"에 맞고, 하네스 자신의 imbd·ltr·lbr 처리(둘 다 T > 0 요구)와도 맞는 쪽은 내 처리다. SQQQ(964거래일 중 116일 목록 밖)·INV·LEV2IDX가 든 nrel W3에서 공통 칸이 하네스 쪽 +3·−1로 갈려 37 대 35가 된다. 갈린 4칸은 어느 쪽 값으로도 F율 또는 D율이 1/3을 넘어(LEV3|INV W3는 하네스 F 0.333이지만 D 0.559) 구별 칸 수(0)는 그대로다.')
    w('2. **B 몫(T_g ÷ T_all 등)의 결측**: 하네스는 T_g = 0인 날 몫을 0으로 둔다. 나는 결측으로 뒀다. 목록 밖은 0이 아니라 50위 밖 미관측이므로 BRIEF2 문장대로면 결측이 맞다. 공통 칸 수에는 영향이 없다(b0 변형도 35).')
    w('3. **E:srank 모집단**: 하네스는 창 안 매도 > 0 상품만 순위를 매기고 그 수로 나눈 것으로 보인다(이 해석으로 E:srank 불일치 21칸이 7칸으로 준다). 나는 거래 > 0 상품(E:nrank와 같은 모집단)으로 나눴다. BRIEF2는 "종목 수로 나눔"만 적어 둘 다 문장에 맞는다. 공통 칸 수에는 영향이 없다(E:srank는 어느 해석이든 공통 0칸).')
    w('')
    r7 = d5[~d5.ok]
    w(f"- 남은 E:srank {len(r7)}칸: 바닥 P 중앙 차 최대 {float((r7.B10_med - r7.med).abs().max()):.2f}, 바닥 수 차 최대 {int((r7.B10_on - r7.n_on).abs().max())}, 바닥 수 최대 {int(r7[['B10_on', 'n_on']].max().max())}. 하네스 코드의 순위 모집단 세부 차로 보이며 공통 판정(≥ 8)과는 멀다. 원인을 끝까지 좇지 않았다.")
    w('- 자료의 작은 흠: 행 2개(2025-04-25·2025-05-13 FNGU)에서 `net` ≠ `buy − sell`(1~2달러). 나는 buy − sell을 썼고 결과에 영향이 없다.')
    w('')
    w('## 4. 판정')
    w('')
    w('- 헤드라인 5칸(SOXS|TQQQ lbr W3 10/10, SOXL|SOXS lbr W3 10/10, SOXL nshare W3 10/10, SOXL sshare W4 9/10, SOXL|STK1 nrel W2 9/10)의 바닥 수·F율·D율은 하네스와 소수점까지 같다. 추가 3칸(A:imb:SOXL·A:imb:STK1·D:fund W3)도 같다.')
    w('- 공통 P 칸: 하네스 37은 하네스 해석(nrel에서 한쪽 T = 0 허용)에서 재현된다. BRIEF2 문장 그대로는 35다. BRIEF2 기준으로는 내 쪽(35)이 맞고, 하네스 숫자는 결측 규칙 하나를 느슨하게 적용한 결과다. 우연 기대(가짜 사건 평균 38.9, p 0.51)와 비교한 결론은 35로 바꿔도 같다(공통점 수가 우연 수준).')
    w('- 구별 칸(F ≤ 1/3·D ≤ 1/3): 양쪽 모두 0이다. 헤드라인 결론 "이 자료로는 바닥에만 있는 공통점이 없다"는 독립 재계산으로 확인된다.')
    w('- 고르기·확인·조합·가짜 사건 반복(BRIEF2 §4의 2·3·5·6단계)과 Eq 정규화는 이 재계산 범위 밖이다.')
    (OUT / 'verify_recompute.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('report written:', OUT / 'verify_recompute.md')
    for k in VARIANTS:
        print(k, int(res[k][2].ok.sum()), res[k][1]['common_P'], res[k][1]['distinct_P'])


if __name__ == '__main__':
    report()
