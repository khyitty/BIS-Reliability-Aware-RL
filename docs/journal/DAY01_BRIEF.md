# Journal Day 01 CPU brief

## 범위와 보존 경계

- 시작: 2026-09-08 21:00 KST. 원 지시의 목표 종료 00:30, 한계 01:00 KST.
- 새 저널 저장소 `BIS-Reliability-Aware-RL`의 `journal/day01-cpu` 브랜치에서만 작업한다.
- 역사 소스 import commit: `20582de4bcae12a7cb4c3e60d9a83222cfe87158` (`Align repository with seed-42 manuscript scope`). 저널 저장소 `origin`만 fetch/push remote로 남겼고 원 저장소에는 쓰지 않았다.
- 현재 모델 이름은 실행 환경에서 노출되지 않아 기록하지 못했다. prompt가 모델을 바꾼다고 간주하지 않았다.
- 기존 ICTC seed-42 결과와 별도 Phase 8G seed-43/44 결과는 수정하지 않는다. 저널 조건은 `journal_day01_synthetic_factorial_v1`로 별도 식별한다.
- raw signal, 환자/subject 레코드, episode trace, checkpoint/model은 Git에서 제외한다. 커밋 대상은 코드, 설정, 비식별 집계뿐이다.

## 실행 환경과 데이터 inventory

- Windows 11 Enterprise, AMD Ryzen 7 8840HS (8 core/16 logical), RAM 31.30 GiB(시작 시 free 21.53 GiB), C: free 870.36 GiB.
- 격리 환경: `.venv-journal`, Python 3.13.9. `requirements/phase7h_rl_lock.txt`의 고정 버전과 일치하도록 구성했다(PyTorch 2.13.0, SB3 2.8.0, Gymnasium 1.2.3, NumPy 2.4.6, SciPy 1.17.1).
- 커밋된 scaler registry와 Phase 8A split/Phase 8D config/Phase 8G aggregate는 존재한다.
- 비공개 `phase8b_train_observation_templates_v1` 및 `phase8c_train_runtime_inputs_v1`은 현재 작업 폴더와 알려진 인접 원본 경로에 없다. 따라서 실제 VitalDB 재학습으로 가장하지 않고 합성 공학 benchmark를 실행한다.

## 오늘의 결정

- 기존 P0/P1 기본 동작을 유지하면서 journal-only `ObservationRule`로 SQI gate(`off`, `>=50`)와 accepted-event age(20, 30초)를 독립화했다.
- 12 합성 subject × 2 case를 만들고 split seed `20260908`로 subject 단위 10 train / 2 validation 분할을 고정했다. profile, SQI/BIS event pattern, remifentanil schedule은 결정론적이다.
- 기존 train-only scaler와 Phase 8D PPO hyperparameter를 재사용했다. CPU, Torch thread 1, 단일 training job, 새 seed 45를 사용했다.
- 4,096-step benchmark가 4.31초(949.6 step/s)에 완료되어, 결과를 비교하기 전에 8셀 모두 32,768-step 공통 예산을 선택했다.
- 관측은 실제 PK/PD simulator와 PPO update를 통과한다. 그러나 합성 cohort이므로 결과는 파이프라인/정책 행동의 공학 증거일 뿐 VitalDB 또는 임상 증거가 아니다.

## 검증과 관측 진단

- 새 회귀 테스트와 관련 Phase 7H 테스트 9개가 통과했다. 역사 P0=off/30, P1=SQI50/20 기본값이 바뀌지 않았음을 검사했다.
- 전 학습/평가 전환에서 scaler 출력 finite invariant가 적용됐고, 8셀의 core action clipping은 모두 0이었다.
- missing event, SQI rejection, 20/30초 staleness 의미를 단위 테스트했다.
- 평가 중 BIS history를 40 대 60으로 바꾼 정책 행동 차이를 저장해, 관측 BIS 변화에 대한 행동 반응을 셀별로 진단했다.

## 상태와 재개

- 완료 상태: `outputs/journal/day01/synthetic_seed45/status.json`
- 상세 로컬 결과/모델: `outputs/journal/day01/synthetic_seed45/<condition>/` (Git-ignored)
- 집계: `reports/journal/day01_results.csv`, `reports/journal/DAY01_REPORT.md`
- 검증된 재개 명령: `.\.venv-journal\Scripts\python.exe scripts\journal\run_day01_cpu.py --resume`
- 재개는 condition/result manifest의 condition ID와 요청/실제 timestep을 확인하고 완료 셀을 건너뛴다. 오늘 실행은 이미 8/8 완료되어 위 명령은 재학습 없이 보고서와 완료 상태를 재생성했다.

## 다음 큐

1. 비공개 train stores가 읽기 전용으로 제공되면 원래 TRAIN subject만으로 새 15% validation split을 생성하고 같은 8셀을 scratch 재학습한다.
2. 데이터가 계속 없고 추가 공학 안정성 증거가 필요하면 seed 46의 8셀 전체를 동일 예산으로 실행한다.
3. 예산 심화는 결과가 좋은 셀만이 아니라 모든 셀의 다음 공통 rollout-aligned milestone으로 수행한다.
