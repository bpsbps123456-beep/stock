import sys, runpy
sys.argv = ["recheck.py", "--tag", "dbg"]
# recheck.py를 모듈처럼 실행하지 않고, 창 표 만들기까지만 재사용하려고 exec로 앞부분만 돈다
src = open("recheck.py", encoding="utf-8").read()
cut = src.index("# ---------------------------------------------------------------- 4.")
g = {"__name__": "dbg"}
exec(compile(src[:cut], "recheck_head", "exec"), g)
df = g["df"]; HARK = g["HARK"]
m = df["judge_ok"] & (df["D_str"] >= "2026-06-01")
sub = df[m]
cols = ["lnHL", "A_rz8"] + HARK
na = sub[cols].isna()
print(len(sub))
print(sub.loc[na.any(axis=1), ["win_id"] + [c for c in cols if na[c].any()]].to_string())
