# Day 01 CPU 최종 보고서

## 실제 완료 범위

- 작업 시간: 2026-09-08 21:00–2026-09-09 00:18 KST, 약 198분.
- 합성 공학 학습: 4개 공통 milestone(32,768 / 65,536 / 131,072 / 262,144) × seed 45/46/47 × 8개 gate-age-state 셀 = 96/96 완료, 실패 0.
- 본 batch 총 11,796,480 environment timestep. 별도로 4,096-step 속도 benchmark도 완료했다.
- 실제 PK/PD simulator, 기록형 관측 event semantics, train-only historical scaler, SB3 PPO update를 사용했다. 합성 subject/profile/event이므로 VitalDB 또는 임상 성능 증거가 아니다.
- 최종 262,144 milestone의 순수 학습시간 합은 6,458.7초, 평균 974.1 timestep/s였다. seed 45/46 두 job 병렬화는 각 약 354MB에서 단일-job 셀 속도를 유지해 combined throughput을 약 2배로 높였다.

## 역사 증거 확인

커밋된 Phase 8G aggregate에서 seed를 선택하지 않고 모두 확인했다. 아래 interaction은 `(P1S1−P1S0)−(P0S1−P0S0)` MAE이다.

| seed | P0S0 | P1S0 | P0S1 | P1S1 | interaction |
|---:|---:|---:|---:|---:|---:|
| 42 | 12.781 | 5.421 | 7.673 | 6.954 | +6.641 |
| 43 | 11.717 | 6.032 | 7.726 | 10.557 | +8.516 |
| 44 | 18.694 | 8.029 | 8.428 | 7.816 | +10.053 |

세 seed의 interaction 방향은 같지만 세 seed만으로 일반 강건성을 확립하지 않는다. aggregate에는 각 최종 모델 SHA-256 provenance가 있으나 이 노트북에 historical 모델 파일은 없다.

## 최종 262,144-step 합성 결과

case metric을 먼저 계산하고 반복 case를 subject 안에서 평균한 뒤 subject 평균을 냈다.

| condition | 3-seed MAE mean | SD |
|---|---:|---:|
| Goff_A30_S0 | 5.521 | 0.739 |
| Goff_A20_S0 | 5.393 | 0.361 |
| G50_A30_S0 | 6.716 | 0.478 |
| G50_A20_S0 | 7.339 | 0.796 |
| Goff_A30_S1 | 7.204 | 1.186 |
| Goff_A20_S1 | 6.569 | 0.539 |
| G50_A30_S1 | 7.482 | 0.610 |
| G50_A20_S1 | 6.028 | 0.215 |

이 표는 순위를 확정하기 위한 것이 아니다. 최종 셀별 raw aggregate는 `day01_multiseed_t262144_results.csv`에 있다.

## 핵심 진단

- 32,768에서 seed 46의 S1 네 정책이 거의 0 행동으로 붕괴했고 MAE가 43.7–46.9였다. 65,536 이상에서는 완화되어 짧은 예산/초기화 불안정성이 확인됐다.
- gate×age interaction은 131,072에서 세 seed의 S0/S1 모두 음수였지만 262,144에서는 다시 부호가 섞였다. 최종 S0 interaction은 −0.022 / +0.219 / +2.053, S1은 +0.752 / −0.736 / −2.473이었다. 따라서 interaction 강건성은 확립되지 않았다.
- 모든 96셀의 core action clipping은 0이었다. observation/scaler finite invariant도 모든 전환에서 통과했으므로 action bound나 non-finite normalization 결함 증거는 없었다.
- 정책의 BIS-history 40↔60 counterfactual action 차이를 각 셀에 저장했다. 관측 BIS 변화에 대한 민감도는 존재하지만 seed·조건별 편차가 컸다.
- 고정 1.5 mg/10s comparator MAE는 14.399였다. 비튜닝 P-like comparator(base 1.5, gain 0.08)는 gate/age별 5.600–6.380으로 131,072-step PPO 3-seed 평균 6.186–8.568과 비슷하거나 더 좋았다. PPO 우월성 증거가 없으며 comparator는 임상 PID나 최적 baseline이 아니다.

## 구현·검증

- journal-only `ObservationRule`로 SQI gate(off/50)와 accepted-event age(20/30초)를 독립화했다. 역사 P0=off/30과 P1=SQI50/20 기본 동작은 바꾸지 않았다.
- 12 합성 subject × 2 case, split seed 20260908로 10 train / 2 validation subject를 고정했다.
- CPU, `n_envs=1`, Torch thread 1, actor/critic 128-Tanh, Phase 8D PPO hyperparameter를 사용했다.
- 관련 unit/integration/protocol 테스트 12개가 통과했다. Phase 8G runtime-only 테스트 7개는 private runtime 부재로 skip됐고 public protocol 테스트는 통과했다.
- 단일-run lock, PID/identity, atomic status/result write, 완료 manifest 검증 기반 resume를 구현했다. 모델·status·로컬 result는 Git-ignored다.

## 한계와 다음 작업

비공개 `phase8b_train_observation_templates_v1`과 `phase8c_train_runtime_inputs_v1`이 없어 실제 원본 TRAIN subject 개발 분할 재학습은 할 수 없었다. 합성 결과를 실제 VitalDB 결과로 해석하거나 최종 journal architecture를 선택해서는 안 된다.

다음 유효 작업은 두 private store를 읽기 전용 경로로 제공한 뒤 원래 TRAIN subject만으로 새 15% validation split을 만들고 동일 8셀을 scratch 재학습하는 것이다. GPU는 이번 병목의 필수 해결책이 아니었다. 작은 네트워크에서 CPU 처리량이 약 950–974 timestep/s였으며, 실제 데이터 I/O 병목을 먼저 측정해야 한다.

검증된 전체 재개 명령:

```powershell
.\.venv-journal\Scripts\python.exe scripts\journal\resume_day01.py
```

이 명령은 완료 manifest의 seed/condition/budget/claim boundary를 검증하고 완료 셀을 건너뛴 뒤 네 milestone aggregate를 재생성한다.
