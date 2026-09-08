# Night 02 CPU brief

## 시간과 범위

- 시작: 2026-09-09 01:17 KST
- 목표 종료: 08:47 KST, hard deadline: 09:17 KST
- 브랜치: `journal/night02-cpu`, 시작 revision `f0fe414cef213ada5961c94fa61af19a14b7ed2f`
- CPU only, `.venv-journal`, 원본 역사 checkout과 private stores는 read-only로 사용한다.

## 데이터와 주장 경계

역사 checkout의 완전한 Phase 8B/8C train stores(1,970 cases)를 loader의 checksum 및 split guard로 확인했다. 새 정책은 기록된 demographics, BIS-event availability/SQI, exogenous remifentanil tape를 보지만, BIS와 propofol-dependent concentration은 reconstructed simulator가 정책 행동으로 생성한다. 결과는 VitalDB-informed reconstructed simulation이며 실제 임상 intervention outcome이 아니다.

## 동결 개발 계획

- 원래 TRAIN subjects만 seed 20260908로 약 85/15 development train/validation 분리한다.
- 실행비용을 제한하기 위해 결과를 보지 않고 development train 96 subjects, validation 24 subjects를 deterministic hash 순서로 선택한다. 모든 case는 subject와 함께 움직인다.
- 각 episode는 실제 anesthesia start와 zero initial drug state를 유지한 첫 600초 공통 구간이다. event를 downsample하지 않고 이 구간 밖 event만 제외한다.
- normalization은 전체 development-training subjects에 역사 preprocessing-neutral recipe를 새로 fit한다. validation은 fit에 쓰지 않는다.
- 8개 gate(off/50) × age(20/30) × state(S0/S1), seeds 45/46/47, 동일 fresh initialization과 공통 rollout-aligned target을 사용한다.
- 우선 4,096 actual-input transition과 대표 validation episode를 benchmark한 뒤 25% margin과 마지막 30분을 반영해 262,144 또는 524,288 target을 결과 확인 전에 확정한다.
- checkpoint는 한 trajectory 안에서 65,536-step마다 저장한다. 중단 후 continuation은 environment mid-episode state를 완전 복원하지 못하므로 `warm_episode_restart`, bitwise exact resume가 아니다.

## baseline 계획

고정 1.5 mg/10s와 원래 P(1.5, 0.08)를 유지한다. PI는 development-training tuning subjects에서 profile별 최대 12개 고정 grid만 비교하고 validation 전에 freeze한다. missing feedback에서는 base action을 사용하고 integral을 갱신하지 않으며 saturation anti-windup을 적용한다.
