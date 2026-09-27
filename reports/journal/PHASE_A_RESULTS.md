# Phase A 방법론·해석 진단 결과

## 결론

Day 04--06의 핵심 수치는 저장된 5,760 development row에서 오차 없이 재계산되었다. 600--1,800초만 보아도 P0/P1의 상태 순위는 모두 `S_CONC < S_CE < S_PROP < S_CP < S0 < S_REMI`로 전체 0--1,800초 순위와 같았다. 다만 이 결과는 이미 반복 사용된 96-subject development validation의 탐색적 민감도 분석이며, 정렬 오류나 일반화를 입증하지 않는다.

가장 중요한 해석 변화는 세 가지다.

1. simulator의 `t=0`은 임상적 induction 시작이 아니라 기록의 `anesthesia_start`에 상대화한 시점이다. 0--600초 저가시성은 기록 시작과 BIS event onset/availability의 정렬에서 크게 발생하고 P1에서는 SQI gating이 추가된다. 임상적 induction 구간으로 부를 근거는 없다.
2. 보존된 historical P0S0/P1S0 final checkpoint는 원본 checkout에 없었다. 따라서 historical step 0/final 비교는 미수행이며, Day 04 journal S0 checkpoint 대체 진단에서는 gross Tanh saturation을 관찰하지 못했다. 이는 conference interaction 가설을 반증하지 않는다.
3. S_CONC는 BIS를 완전히 무시하지 않았다. value와 mask를 함께 끈 one-step 변화는 작아 보였지만 mask-only와 value-only 변화는 서로 반대 방향으로 더 컸다. 단일 ablation의 상쇄를 “BIS 불필요”로 해석하면 안 된다. 농도 입력 perturbation 반응은 BIS-off보다 대체로 컸으나, 이는 물리적으로 일관된 mismatch 실험이 아니다.

## 평가 경계와 provenance

- 시작 commit: `8b00a4ee1ed69afb92e8303d80504fca7f023465`
- frozen brief commit: `1b772ccb947c973c5e86218fd23ff4a833c612ee`
- 분석 자료: Day 04--06 ignored development outputs, Day 04 train-only calibration matrix, journal checkpoints, development train runtime bundles
- development validation: 96 subjects / 96 cases, 이미 반복 관찰되었으므로 탐색적
- A5 tuning: 미리 정한 development-train diagnostic 순서의 처음 12 cases; validation을 보며 재조정하지 않음
- future reserve 170 subjects/cases: frozen count/hash 외 bundle/outcome/statistic 접근 0
- historical sealed test 및 journal test: 접근 0
- Phase B 신규 PPO 학습: 실행 0

## A4. 시간 정렬과 구간별 성능

### 정렬 추적

| 대상 | 실제 정렬 |
|---|---|
| 실데이터 BIS/SQI | absolute timestamp에서 `anesthesia_start`를 뺀 상대시각 |
| remifentanil schedule | 같은 `anesthesia_start` 상대시각 |
| simulator `t=0` | 상대시각 0; zero-state PK/PD reset |
| 초기 Cp/Ce | 두 약물 모두 0 |
| propofol 투여 | 첫 policy action이 0--10초 simulator interval에 적용됨 |
| 임상 기록 첫 propofol | runtime 입력에 없으므로 확인 불가 |
| carry-forward | 마지막 accepted BIS를 profile별 staleness cap까지 유지 |
| BIS age | query time − event time, 0--30초 cap |
| 음수 history offset | prior event 없음으로 value 0/mask 0 |

### 가시성

full development train의 subject-mean 기준으로 0--600초 visibility는 P0(off/A30) `0.2658`, P1(SQI50/A20) `0.1220`이었다. 같은 구간 zero-visibility subject 비율은 각각 `0.5168`, `0.6857`이고, never-visible에 window endpoint를 부여한 first-visible summary는 각각 440.4초와 521.8초였다. 0--1,800초 visibility는 `0.5758`, `0.4362`였다.

P0에서도 초기 visibility가 낮고 staleness rejection은 0.01% 수준이므로, 주된 원인은 carry-forward cap이 아니라 anesthesia-start 이후 BIS event가 아직 없거나 available하지 않은 기록 정렬이다. P1에서는 0--600초 BIS/SQI event 중 SQI rejection fraction `0.3004`가 추가로 visibility를 낮춘다. 이 수치는 센서 연결 시각이나 임상 induction 시작을 직접 측정하지 않는다.

### 600--1,800초 성능

| Profile | State | MAE 0--600 | MAE 600--1,800 | TIR 0--1,800 | Dose mg |
|---|---|---:|---:|---:|---:|
| P0 | S_CONC | 7.3678 | 0.6617 | 0.9274 | 214.2 |
| P0 | S_CE | 8.5303 | 1.1666 | 0.9163 | 215.2 |
| P0 | S_PROP | 9.3082 | 1.3246 | 0.9097 | 212.4 |
| P0 | S0 | 14.6989 | 3.6599 | 0.8096 | 216.5 |
| P1 | S_CONC | 7.2151 | 0.7590 | 0.9298 | 213.5 |
| P1 | S_CE | 7.9176 | 1.2769 | 0.9225 | 221.3 |
| P1 | S_PROP | 8.8041 | 1.4665 | 0.9135 | 214.0 |
| P1 | S0 | 16.0818 | 4.3453 | 0.7539 | 204.7 |

모든 six-state 순위는 full-window와 late-window에서 같았다. Private subject aggregation과 public Day 06 CSV의 최대 절대 불일치는 `0.0`이었다. 이는 후반부에서도 concentration representation과 성능의 연관이 남는다는 관찰이지, 앞 600초 정렬이 잘못됐다는 증거는 아니다.

## A3. Numerical conditioning

### 수행 범위

요청된 historical P0S0/P1S0 final checkpoint는 원본 checkout에 존재하지 않았다. P0S1/P1S1 일부만 있었으므로 다른 상태를 대신 historical S0라고 부르지 않았다. 대체 진단은 Day 04 journal-trained P0S0/P1S0, seeds 48--52에 대해 수행했다.

| Profile/window | input |preact| p95 | first Tanh |h|>0.95 | |h|>0.99 | BIS-off Δaction mean abs |
|---|---:|---:|---:|---:|
| P0 0--600 | 1.000 | 0.1568 | 1.61e-4 | 3.62e-5 | 0.0529 |
| P0 600--1,800 | 1.000 | 0.1211 | 2.67e-6 | 0 | 0.1460 |
| P1 0--600 | 1.000 | 0.0611 | 1.03e-4 | 2.02e-5 | 0.0376 |
| P1 600--1,800 | 1.000 | 0.0674 | 7.63e-7 | 0 | 0.1322 |

Day 04 final S0 actor에는 gross saturation이 없었다. 후기 BIS-off `Δpreactivation`과 `Δaction`이 초기보다 커서 작은 포화율이 곧 작은 BIS 민감도를 뜻하지도 않았다. 같은 34→128 구조를 12개 새 seed로 무학습 초기화한 별도 probe의 BIS-off `Δpreactivation L2` 평균은 `0.9187`이었다. 당시 초기 가중치는 보존되지 않았으므로 이 값은 historical step 0 복원이 아니다.

결론 강도는 제한적이다. final saturation이 학습 실패의 원인/결과라는 인과 판정은 불가하고, historical conference-like full-length 정책 자체도 검사하지 못했다. 향후 replication에서는 checkpoint 보존, paired P0/P1 observation replay, per-layer activation뿐 아니라 input-output Jacobian/gradient와 action distribution magnitude를 함께 기록해야 한다.

## A1. 기존 S_CONC 정책의 관측 반사실

동일 model input에서 BIS coordinate만 바꾼 one-step paired counterfactual이다. 닫힌루프 환경의 visibility process를 변경한 실험이 아니다.

| Profile/window | BIS value+mask off | mask-only off | value-only zero |
|---|---:|---:|---:|
| P0 0--600 | 0.0452 | 0.1237 | 0.1301 |
| P0 600--1,800 | 0.1087 | 0.4237 | 0.3093 |
| P1 0--600 | 0.0224 | 0.1323 | 0.0980 |
| P1 600--1,800 | 0.0507 | 0.4455 | 0.2478 |

표는 seeds 48--52의 mean absolute `Δaction`(mg/10 s) 평균이다. value+mask 동시 제거가 작은 이유는 mask-only가 action을 높이고 value-only가 낮추는 반대 반응이 상쇄되기 때문이다. 따라서 S_CONC의 BIS-off 반응이 작다는 한 사실로 BIS가 불필요하거나 reliability fusion이 검증됐다고 말할 수 없다. 또한 off input은 학습 분포 밖일 수 있다.

## A2. Concentration/PD 입력 민감도

| Perturbation | P0 0--600 / late mean | P1 0--600 / late mean |
|---|---:|---:|
| concentration ×0.8 | 0.2831 / 0.3887 | 0.3125 / 0.4195 |
| concentration ×1.2 | 0.2689 / 0.2940 | 0.2948 / 0.3082 |
| −0.5 model-unit bias | 1.6253 / 1.7419 | 1.8232 / 1.9230 |
| +0.5 model-unit bias | 1.4637 / 0.6305 | 1.5463 / 0.6056 |

값은 mean absolute `Δaction`(mg/10 s)이다. 20% concentration scaling의 방향은 일관되어 낮춘 입력이 action을 높이고 높인 입력이 action을 낮췄다. additive bias는 크고 비대칭적이었으며 최대 개별 변화는 약 2.36 mg/10 s였다.

이 probe는 policy input만 바꿔 true patient dynamics와 observation을 그대로 둔 물리적으로 불일치한 실험이다. PD parameter, true PK/PD mismatch, concentration delay는 수행하지 않았다. 따라서 임상 robustness 근거가 아니며, 다음 단계에서는 true patient와 nominal observer를 분리하고 profile/seed별 sign reversal, action saturation, state-estimation error와 BIS artifact를 함께 기록해야 한다.

## A5. 정보량을 맞춘 최소 비학습 비교군

development train 12 cases에서 6개 nominal Ce-target 후보를 비교해 `target Ce=2.5 mg/L`, maintenance `1.5 mg/10 s`, proportional gain `3.0`을 고정했다. 이어 P0/P1을 함께 사용해 6개 BIS trim 후보에서 `Kp=0.05`, `Ki=0.002`를 고정했다. 적분기는 ±300 error·seconds에서 제한하고 saturation 시 적분하지 않는다. action은 0--27.7 mg/10 s이며 현재 시점의 simulator-derived nominal Ce와 causal BIS만 사용한다.

| Controller/profile | MAE | late MAE | TIR | BIS>60 | BIS<40 | Dose mg | requested clip/boundary | core clip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ce target / P0 | 4.4006 | 4.0011 | 0.9591 | 0.0333 | 0.0075 | 250.6 | 0.0427 | 0 |
| Ce target + BIS PI / P0 | 3.6647 | 2.9390 | 0.9652 | 0.0329 | 0.0019 | 241.8 | 0.0447 | 0 |
| Ce target / P1 | 4.4006 | 4.0011 | 0.9591 | 0.0333 | 0.0075 | 250.6 | 0.0427 | 0 |
| Ce target + BIS PI / P1 | 3.7670 | 3.0878 | 0.9637 | 0.0332 | 0.0031 | 241.9 | 0.0437 | 0 |

PI trim은 train tuning MAE를 5.0456에서 4.1912로 낮췄고 development validation에서도 두 profile 모두 개선됐다. 그러나 S_CONC PPO의 MAE 2.8971/2.9110보다 높았으며, 반대로 TIR은 baseline이 더 높았다. 이는 MAE 하나로 controller를 정렬하면 다른 임상형 지표를 놓칠 수 있음을 보여준다.

이 비교군은 작동하는 최소 feasibility 구현이다. simulator-derived Ce를 nominal observer 값으로 사용해 true dynamics와 observer가 분리되지 않았고, 임상 TCI pump 알고리즘을 재현하지 않았다. 따라서 TCI를 oracle이라 부르지 않으며 exact-state PPO도 수학적 oracle이라 부르지 않는다.

## 실패 사례와 미수행 항목

- historical P0S0/P1S0 checkpoint 부재: A3 historical final/step-0 비교 미수행
- paired P0/P1 raw observation replay 부재: 동일 latent state의 실제 P0/P1 새 초기화 비교 미수행
- 임상 propofol track 부재: 기록 첫 propofol과 simulator first action 정렬 미확인
- trajectory에는 10초별 latent/action row가 공개 보존되지 않음: A4는 이미 저장된 subject/case window metric 재계산
- A2 PD/PK dynamics mismatch와 delay 미수행
- A5 nominal observer/true patient 분리와 정식 TCI 미수행
- Phase C patient--observer mismatch, SQI-dependent artifact, recurrent observer-policy는 설계 후보로만 유지

## Phase B 제안

현재 근거만으로 곧바로 대규모 PPO를 재학습하지 않는 편이 타당하다. 먼저 다음 세 조건을 충족하는 짧은 protocol을 동결하는 것을 제안한다.

1. historical 또는 conference-like P0S0/P1S0 checkpoint와 paired replay observations를 보존해 A3의 직접 replication을 가능하게 한다.
2. true patient PK/PD와 nominal observer를 분리하고, concentration bias/lag 및 SQI-dependent BIS artifact를 독립 요인으로 만든다.
3. S_CONC PPO와 동일 정보 경계의 Ce-target/BIS-trim baseline을 정식화하고 MAE·TIR·high/low BIS·dose·clipping을 공동 endpoint로 둔다.

그 뒤 Phase B는 짧은 smoke/scale gate를 먼저 통과하고, mechanism 차이가 실제로 관찰될 때만 full-length paired-seed 학습으로 확장한다. 이번 Phase A만으로 학회 interaction의 원인을 밝혀냈다고 결론내릴 수 없다.

## 재현과 검증

- 최종 config 진단 실행 wall time: 133.5초
- worker/Torch thread: 1/1
- 공개 결과 파일: 5개; checksum 검증 통과
- Phase A + Day 04--06 관련 tests: 26 passed
- 전체 suite: 393 passed, 19 skipped, 41 failed, 27 errors. 실패/오류는 기존 protected artifact checksum 불일치와 Git-ignored raw/private 자료 부재에서 발생했으며 이번 변경으로 수정하지 않았다.
- 재개: `.venv-journal\Scripts\python.exe scripts\journal\run_phase_a_diagnostics.py resume --historical-root "C:\Users\admin\Desktop\김중헌 교수님 학부인턴\VitalDB-State-Selection-Reconstructed"`
- 완료 marker가 존재하면 `resume`은 계산을 중복 실행하지 않고 공개 산출물을 검증한다.
