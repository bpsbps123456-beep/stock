# 과제 B 이어 하기 지시 (2026-11-11, Claude)

첫 차수는 20분 만에 '사용량 제한'으로 멈췄다. 그 시각에 과제 A 작업자는 계속 돌고 있었으므로 계정 한도는 남아 있다. `news/PROGRESS_B.md`의 '이어받는 순서' 1 → 5를 그대로 끝까지 한다.

## 멈춤 기준

- **요청이 실제로 거부되는 한도 오류가 나기 전에는 멈추지 않는다.** 일이 크다, 오래 걸린다는 이유로 인계하지 않는다. 몇 시간이 걸려도 된다.
- 실제 오류로 멈추면 그 오류 문구를 그대로 `news/PROGRESS_B.md`에 적는다.

## B1 전수 판정 (가장 먼저)

1. `pending_review_B1.csv` 4,200개를 **전부** 제목으로 판정한다. 단어 규칙만으로 한꺼번에 제외하지 않는다(앞선 PB 전수 표본에서 자동 제외 표본의 4%가 실제 해당이었다). 규칙으로 먼저 나누더라도 제외 쪽을 직접 읽어 확인한다.
2. 후보마다 한 줄씩 남긴다: `candidate_id`, `decision`(include / exclude), `reason_code`(single_stock / not_a_change / pb_flow / other_asset / other_region / duplicate / unrelated / out_of_window), `note`. 파일은 `news/review_B1_v2.csv`. Claude가 이 파일에서 무작위 표본을 뽑아 독립 재판정으로 일치율을 잰다.
3. 채택 후보는 본문(원문 → 통신사 → 재보도 순)으로 확인하고 `from_rating`·`to_rating`·목표 수치 빈칸을 채운다. 못 읽은 것은 `verification`에 그대로 적는다.
4. 같은 의견의 재보도는 `source_group`으로 묶고 최초 공개일을 쓴다. PROGRESS_B의 '추가 확인할 구체적 후보' 6건을 처리한다.
5. 반복해 나온 RBC·SocGen·Standard Chartered 같은 기관은 `protocol_B_v2.json`에 추가 사유와 함께 적고 같은 틀로 검색한다. v1 규칙은 바꾸지 않고 추가만 한다.
6. 기존 부분 동결은 그대로 둔다. 확정 표본은 `news/v2/` 아래에 만들고 `freeze_B_v2.json`을 커밋한 뒤에만 가격과 연결한다.

## 그다음

- B1 가격 분석(8칸, H-U·H-D 2026 주 검정, 가격 짝 대조, 바닥 대비 위치).
- B2 수집 → 날짜를 섞은 부호화 → 동결 커밋 → 분석. 진짜 10개, 완료 국면 첫 F 6개, 현재 국면.
- `report_B.md`의 미계산 칸을 실제 값으로 바꾼다. `py -m unittest` 전체 통과.

## 끝낼 때

`TASK_B.md` '끝낼 때' 그대로다. 경로 지정 커밋, 푸시(거부되면 rebase 없이 로컬에 두고 보고), 마지막 보고 항목 전부.
