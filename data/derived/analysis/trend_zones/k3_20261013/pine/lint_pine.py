"""파인 파일 줄 단위 정적 검사: 따옴표 닫힘, 문자열·주석 밖 비ASCII, 들여쓰기 4칸 배수, 괄호 균형."""
import sys
path = sys.argv[1]
bad = 0
depth = 0
for no, line in enumerate(open(path, encoding="utf-8").read().split("\n"), 1):
    code, q, i = [], None, 0
    while i < len(line):
        ch = line[i]
        if q:
            if ch == "\\":
                i += 2; continue
            if ch == q:
                q = None
            i += 1; continue
        if ch in "\"'":
            q = ch; i += 1; continue
        if line.startswith("//", i):
            break
        code.append(ch); i += 1
    if q:
        print(f"{no}: 따옴표 안 닫힘: {line}"); bad += 1
    c = "".join(code)
    na = [ch for ch in c if ord(ch) > 127]
    if na:
        print(f"{no}: 문자열·주석 밖 비ASCII {na}: {line}"); bad += 1
    ind = len(line) - len(line.lstrip(" "))
    if c.strip() and ind % 4:
        print(f"{no}: 들여쓰기 {ind}칸"); bad += 1
    if "\t" in line:
        print(f"{no}: 탭"); bad += 1
    for ch in c:
        depth += {"(": 1, ")": -1, "[": 1, "]": -1}.get(ch, 0)
    if depth != 0:
        print(f"{no}: 괄호 미균형 누적 {depth}: {line}"); bad += 1
        depth = 0
print("문제", bad, "건")
