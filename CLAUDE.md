# GBRC Claude Code 호환 진입점

이 프로젝트의 독립 운영 규칙은 이 파일에 두지 않는다.

작업을 시작하기 전에 저장소 루트의 `AGENTS.md`를 전부 읽고, 그 파일이 지정한 Path의 정본 문서를 전부 읽어 그대로 따른다. 섹터·종목 매핑은 `watchlist.py`가 정본이다.

우선순위는 다음과 같다.

1. 사용자의 현재 요청
2. `AGENTS.md`의 활성 라우팅과 금지사항
3. 해당 Path의 `docs/*_system.md` 또는 `docs/earnings_framework.md`
4. `docs/operations_changelog.md`의 과거 이력

`docs/v40_handoff.md`와 `docs/archive/`는 명시적으로 요청받지 않는 한 활성 규칙으로 사용하지 않는다. 규칙을 바꿀 때도 이 파일에 복제하지 말고 정본 한 곳만 수정한다.
