# 사건 연구 실행 모드: 확정 사건만 (부록 14, 2026-10-23)

## 왜
- 부록 14(사용자 요청 "지금부터 바로 분석 시작해요")로 사건 연구를 부록 12 판정(10/24)보다 먼저 한다. 대상은 2026-10-22 종가까지 확정된 고점·저점뿐이고, 끝 극값은 '미확정'으로 따로 둔다.
- 기존 `run`은 날짜 가드(`guard_real_run`)가 2026-10-24 06:00 KST 전이거나 10/23 종가 재판정 전이면 멈춘다. 이 모드에서는 그 가드만 '끝 극값과 그 ±3거래일 보류 가짜를 뺀다'로 바꾼다. 분석 로직·문턱·귀무는 그대로다.

## 바꾼 줄 (`events/code/event_study.py`)
| 위치 | 바꾼 것 | 이유 |
|---|---|---|
| 맨 위 설명문 '실행' | `run --confirmed-only` 두 줄 추가 | 사용법 |
| import | `import copy` | events.json 사본을 고친다(원본 dict는 그대로) |
| 상수 | `CONFIRMED_ASOF = "2026-10-22"`, `CONFIRMED_OUT = RESULTS / "confirmed_1022"` | 기준 종가일과 산출 폴더(10/24 뒤 전체 실행 `results/`와 섞이지 않게) |
| 새 함수 `confirmed_only_events(ev, asof)` | ① 모든 자산 `data_through == asof`가 아니면 멈춤 ② 사건 = 확정 꺾임점만, `confirm_date ≤ asof`가 아니면 멈춤 ③ 끝 극값이 사건 목록에 있거나 보류 가짜가 가짜 목록에 있으면 멈춤 ④ 뺀 끝 극값·보류 가짜를 기록해 돌려줌 | 날짜 가드를 대신하는 확인. events.py가 이미 끝 극값을 사건에서, 보류 가짜를 가짜에서 뺐으므로 목록은 그대로이고 그 사실을 확인·기록만 한다 |
| `run_real(...)` | 인자 `confirmed_asof=None` 추가. `None`이면 기존 `guard_real_run`, 아니면 `confirmed_only_events` | 기본 동작은 그대로 |
| `run_real` → `write_report` | `mode=mode` 전달 | 보고서·JSON에 모드 기록 |
| `write_report(..., mode=None)` | `mode`가 있으면 `event_study.json`에 `"mode"`, `event_study.md` 머리에 '확정 사건만 모드 … 뺀 끝 극값' 한 줄 | 결과를 10/24 뒤 전체 실행과 구별 |
| `main` | `--confirmed-only`, `--out` 추가. `run --confirmed-only`면 `run_real(out_dir=results/confirmed_1022, confirmed_asof="2026-10-22")` | 실행 경로 |

## 바꾸지 않은 것
- 특징 전처리(창 종류 보정, 직전 20창 z, 직전 60창 백분위), 분석 1~4, 누출 시험, 후보 조건, 섞기 1,000번, 위약 20일, 오프셋(0, +1), 8%·12% 문턱.
- 위약 후보에서 끝 극값 ±3거래일을 빼는 규칙(`load_contrasts`의 `pivot_dates`)도 그대로 둔다. 미확정 극값 근처 날을 위약으로 쓰지 않는 기존 규칙이다.
- 기본 `run`의 날짜 가드(10/24 06:00 KST·10/23 종가 재판정)는 그대로다.

## 실행
```
cd events/code
py -B event_study.py run --confirmed-only      # → events/results/confirmed_1022/event_study.md 등
```
입력은 `events/results/features_all.parquet`(스냅샷 2026-10-23 07:59, W_KR 285창·W_U 295창), `events/results/events.json`(10/22 종가 판정), `events/results/prices_close.csv`.

## 테스트
- `py -B -m unittest test_event_study` 25개 통과(새 2개: `TestConfirmedOnly.test_checks_and_record`, `TestConfirmedOnly.test_run_real_confirmed_only_end_to_end`). `test_rows_and_markets`의 창 수는 283·294 → 285·295(`snap_paths.EXPECT_WINDOWS`)로 바꿨다.
- 같은 폴더 `test_events`·`test_views` 40개 통과.

## 10/22 종가 판정의 끝 극값(미확정, 사건에서 뺌)
| 자산 | 문턱 | 끝 극값 | 창 | 종가 | 확정에 필요한 종가 | 사건(저점/고점) | 마지막 확정일 | 뺀 보류 가짜(바닥) |
|---|---|---|---|---|---|---|---|---|
| KOSPI | 8% | 저점 2026-10-22 | K2026-10-22 | 5,533.77 | 5,976.47 이상 | 8 / 9 | 2026-10-12 | 10/19 |
| KOSPI | 12% | 저점 2026-10-22 | K2026-10-22 | 5,533.77 | 6,197.82 이상 | 4 / 5 | 2026-10-13 | 10/19 |
| SK하이닉스 | 8% | 저점 2026-10-20 | K2026-10-20 | 1,223,000 | 1,320,840 이상 | 18 / 18 | 2026-10-08 | 10/15·16·19·20·21·22 |
| SK하이닉스 | 12% | 저점 2026-10-20 | K2026-10-20 | 1,223,000 | 1,369,760 이상 | 9 / 9 | 2026-10-12 | 10/15·16·19·20·21·22 |
| SOXX | 8% | 저점 2026-10-22 | U2026-10-22 | 475.69 | 513.75 이상 | 8 / 9 | 2026-10-14 | 10/20·22 |
| SOXX | 12% | 저점 2026-10-22 | U2026-10-22 | 475.69 | 532.77 이상 | 4 / 5 | 2026-10-15 | 10/20·22 |

- 하이닉스 10/22 종가는 10/20과 같은 1,223,000원이라 동률 규칙(처음 날)으로 끝 극값이 10/20이다.
- 10/15 저점은 세 자산 모두 그 뒤 더 낮은 종가로 깨져 저점 후보가 아니다(`events.json` `focus_2026_10_15`).
- 끝 극값은 10/23 종가 뒤(`py events.py all`) 다시 판정한다. 10/23 뒤 반등은 확정에 쓰지 않는다.

## 같은 갱신에서 바꾼 경로 (계산 로직 그대로)
- 스냅샷을 2026-10-23 07:59 KST까지 늘린 `events/snapshot_1023/`(final/과 같은 꼴, 기준 시각 S 2026-10-23 09:05:26)을 만들고, events 코드가 final/results 대신 그것을 읽게 했다(`snap_paths.py`). final/·interim/ 코드와 동결 문서는 고치지 않았다.
- `views_fetch.py`·`feat_views.py`: 스냅샷·units·산출 폴더 → `snapshot_1023/(results)`. `events/results/views_*`와 `views_leak.md`는 final 스냅샷 10/22 진단판으로 남긴다.
- `features_all.py`: RES·예측 층 대응표·라벨 폴더·feat_views 경로 → snapshot_1023, 창 수 단언 283 → 285.
- `events.py`: 창 표·달력·스냅샷 기록 경로 → snapshot_1023(10/21까지 final과 같다). 그래서 KOSPI 끝 극값 10/22에 창 id K2026-10-22가 붙는다.
- 새 코드: `snap_build.py`(스냅샷·달력·동결 특징 코드·예측 층 집계), `sample_new_windows.py`(새 창 K2026-10-22·23의 예측 층 60·감정 층 40단위 표본, 기존 규칙 함수 그대로). 수치 기록은 `events/snapshot_1023/update_meta.json`.
