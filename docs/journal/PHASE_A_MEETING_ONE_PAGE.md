# Phase A 미팅 1페이지 요약

## 현재 확실한 것

- Day 04--06 development 수치는 5,760 private rows에서 subject 단위로 재계산했으며 public aggregate와 최대 절대 차이 `0.0`이었다.
- 600--1,800초만 보아도 P0/P1 순위는 모두 `S_CONC < S_CE < S_PROP < S_CP < S0 < S_REMI`였다. S_CONC late MAE는 P0/P1 `0.662/0.759`였다.
- simulator `t=0`은 임상 induction이 아니라 `anesthesia_start` 상대 0초다. 0--600초 visibility는 P0 `0.266`, P1 `0.122`; zero-visibility subject 비율은 `51.7%/68.6%`였다. P0에서도 낮으므로 기록/BIS event onset 정렬이 크고, P1에서는 SQI rejection이 추가된다.
- Day 04 journal S0 actor에는 gross Tanh saturation이 없었다. 단, 요청한 historical P0S0/P1S0 checkpoint가 없어 historical mechanism replication은 못 했다.
- S_CONC의 BIS value+mask 동시 off 반응은 작지만 mask-only와 value-only 반응은 더 크고 방향이 반대였다. “BIS를 무시한다”는 결론은 지지되지 않는다.
- concentration input ±20%는 mean absolute action을 약 `0.27--0.42 mg/10 s` 바꿨다. additive ±0.5 model-unit bias는 더 컸다. 이는 policy-input-only stress test다.
- 최소 Ce-target baseline은 MAE `4.401`, BIS PI trim은 `3.665/3.767`이었다. S_CONC PPO의 MAE `2.897/2.911`보다 높지만 baseline TIR은 `0.964--0.965`로 더 높아 multi-metric 비교가 필요하다.
- reserve 170 bundle/outcome/statistic과 sealed test 접근은 모두 0이고 Phase B 재학습도 0이다.

## 아직 미확정인 것

- historical conference interaction이 input scale, gradient, saturation, horizon 중 무엇에서 생겼는지
- 실제 임상 첫 propofol/센서 연결 시각과 simulator `t=0`의 관계
- Cp/Ce 추정 bias·delay와 patient PK/PD mismatch에서 S_CONC 이점이 유지되는지
- SQI가 실제 measurement error를 얼마나 나타내는지와 artifact의 ground truth
- nominal observer를 분리한 정식 TCI/BIS-trim과 PPO의 동일 정보량 비교
- recurrent observer-policy가 필요한지 여부; 현재는 Phase C 설계 후보일 뿐임

## 석사분께 결정받을 질문 3개

1. Phase B 전에 historical P0S0/P1S0 checkpoint와 paired observation replay를 다시 확보·보존하는 replication을 필수 gate로 둘 것인가?
2. 다음 핵심 실험을 `PK/PD patient--observer mismatch × SQI-dependent BIS artifact` factorial로 정하고 S_CONC와 동일 정보량 baseline을 함께 넣을 것인가?
3. primary endpoint를 MAE 단독이 아니라 MAE + TIR + BIS>60/BIS<40 + dose/clipping의 다중 guardrail로 동결할 것인가?

제안: 세 질문을 먼저 확정한 뒤 짧은 simulator mismatch smoke를 수행하고, 구별되는 mechanism 신호가 있을 때만 Phase B full-length PPO로 진행한다.
