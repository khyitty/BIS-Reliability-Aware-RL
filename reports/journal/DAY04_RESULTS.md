# Day 04 장기 구간 확인 결과

## 결론

P0와 P1의 S1−S0 state contrast는 모두 `moderate_suggestive`였다(P0 -2.9068, 4/5 seeds; P1 -3.6091, 4/5 seeds). 평균 방향은 S1에 유리했지만 두 기술적 bootstrap 구간 모두 0을 포함하므로 강한 내부 견고성으로 부르지 않는다. 전처리 효과와 interaction은 seed 간 이질성이 있어 서술적으로만 해석한다.

> 범위: 고정 환자 profile과 remifentanil schedule을 사용한 VitalDB-informed 재구성 simulation trajectory. 임상 효과, 외부 검증 또는 인과 효과의 증거가 아니다.

## Primary conditions

| Condition | MAE 0–1800 (mean ± seed SD) | MAE 0–600 | MAE 600–1800 | visibility | action boundary |
|---|---:|---:|---:|---:|---:|
| P0S0 | 7.3396 ± 1.5875 | 14.6989 | 3.6599 | 0.544 | 0.009 |
| P1S0 | 8.2575 ± 3.1697 | 16.0818 | 4.3453 | 0.413 | 0.003 |
| P0S1 | 4.4328 ± 3.1221 | 10.4062 | 1.4461 | 0.544 | 0.029 |
| P1S1 | 4.6484 ± 3.5379 | 10.6314 | 1.6569 | 0.413 | 0.030 |

## Fixed contrasts

| Contrast | Mean ± seed SD | Favorable seeds | Descriptive 95% hierarchical bootstrap | Language |
|---|---:|---:|---:|---|
| P0_state | -2.9068 ± 3.6016 | 4/5 | [-5.3546, +0.3341] | moderate_suggestive |
| P1_state | -3.6091 ± 5.2492 | 4/5 | [-7.7946, +0.9408] | moderate_suggestive |
| S0_preprocessing | +0.9179 ± 1.6749 | 2/5 | [-0.1297, +2.4500] | descriptive_only |
| S1_preprocessing | +0.2156 ± 0.4425 | 2/5 | [-0.0674, +0.5998] | descriptive_only |
| interaction | -0.7023 ± 1.8973 | 3/5 | [-2.3593, +0.6077] | descriptive_only |

## 구간별 state 효과

| Contrast | 0–600 s mean (favorable seeds) | 600–1,800 s mean (favorable seeds) |
|---|---:|---:|
| P0_state | -4.2927 (4/5) | -2.2139 (4/5) |
| P1_state | -5.4504 (4/5) | -2.6885 (4/5) |

평균 개선 방향은 초기와 후기 구간에 모두 나타났지만 seed 부호 일관성은 표와 같이 제한적이다. 후기 MAE가 모든 조건에서 초기보다 낮아, 1,800초 전체 차이는 초기 유도 구간의 영향도 함께 포함한다.

## 관측 가용성과 정책 진단

| Condition | visibility | SQI rejection | stale | actor Tanh saturation | diagnostic clip | core clip |
|---|---:|---:|---:|---:|---:|---:|
| P0S0 | 0.5441 | 0.0000 | 0.0010 | 0.00e+00 | 0.0092 | 0.0000 |
| P1S0 | 0.4129 | 0.1311 | 0.0012 | 0.00e+00 | 0.0028 | 0.0000 |
| P0S1 | 0.5441 | 0.0000 | 0.0010 | 8.05e-06 | 0.0294 | 0.0000 |
| P1S1 | 0.4129 | 0.1311 | 0.0012 | 1.18e-06 | 0.0298 | 0.0000 |

P1 visibility는 P0보다 약 13.1%p 낮았고, 차이는 SQI rejection과 거의 일치했다. actor Tanh saturation과 core clipping은 사실상 0이었으나 S1의 action-boundary/diagnostic-clip 비율은 약 3%로 S0보다 높았다.

## Baseline 비교

| Profile | constant | P | PI | matched S0 PPO | matched S1 PPO |
|---|---:|---:|---:|---:|---:|
| P0 | 9.4680 | 6.5085 | 6.0052 | 7.3396 | 4.4328 |
| P1 | 9.4680 | 7.0251 | 6.9496 | 8.2575 | 4.6484 |

S0 PPO 평균은 각 profile의 tuned PI보다 나빴고, S1 PPO 평균은 tuned PI보다 낮은 MAE를 보였다. 이는 동일 simulation 설정의 내부 비교이며 임상 우월성 주장이 아니다.

## 장기 Nscale과 Day 03 transfer

1,800초 train-only calibration은 1,209,600 observations를 사용했다. Day 03 대비 propofol cumulative-dose factor는 92.75→255.00 (2.75×)로 커졌고, 네 concentration factor 비율은 1.06–1.43×였다. binary/mask 좌표는 1.0을 유지했고 BIS 순서와 S0-prefix 공유 검사를 통과했다.
Frozen Day 03 P0 S1−S0의 fresh-subset 1,800초 delta는 -3.4645, -5.5104, -1.4427로 3/3 모두 S1 방향이었다. 다만 legacy seed가 3개이고 training universe와 horizon이 동시에 달라졌으므로 사전 정의된 5-seed evidence label의 대상이 아니며, 방향성 transfer 진단으로만 본다.

## Day 05에 대한 시사점

- Feature-group ablation은 S1의 후기 정보(누적 dose와 effect-site/plasma concentration)를 우선 분리하되, S1에서 약 3%였던 action-boundary/clip 진단을 guardrail로 함께 추적해야 한다.
- 더 넓은 P-profile 연구에서는 train-only audit가 만든 뚜렷한 정보 regime을 사용한다. 0–1,800초 visibility는 SQI off/A30 0.575, SQI30/A30 0.453, SQI50/A30 0.439, SQI70/A30 0.381이었다. threshold 효과를 우선 확장하고, age 효과는 높은 threshold에서 별도 factorial로 확인한다.
- 이 제안들은 다음 연구 설계 입력일 뿐 Day 04 결과로 추가 PPO를 선택하거나 시작하지 않는다.

## 한계

내부 source cohort를 재사용했지만 이번 96-subject subset은 이전 journal 개발에 사용되지 않았다. 알려진 historical test 결과는 사용하지 않았고 original-test access는 0이었다. 재구성 simulation이며 임상·인과 주장을 하지 않는다. policy seed가 5개뿐이므로 계층 bootstrap은 기술적 민감도 구간이다.

## 재현

실효 budget은 job당 524,288 steps, 총 20 jobs/10,485,760 steps이며 복구 60개와 최종 20개 checkpoint를 검증했다. Worker는 1개, Torch thread는 1개였고 최초 준비부터 최종 산출물까지 wall time은 5.99시간이었다. Resume: `.venv-journal\Scripts\python.exe scripts\journal\run_day04_confirmation.py resume`. Private identifiers, models, checkpoints, logs와 row-level 자료는 ignored Day 04 output root에만 있다.
