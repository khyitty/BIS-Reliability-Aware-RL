# Phase A Frozen Brief

## 목적과 판정 경계

Phase A는 Day 04--06의 수치를 더 좋게 만들기 위한 추가 학습이 아니라, 시간 정렬·수치 conditioning·관측 반사실·농도 입력 민감도·정보량을 맞춘 비학습 비교군을 점검하는 탐색적 진단이다. 개발 validation 96 subjects는 이미 반복 관찰되었으므로 모든 성능 결과를 탐색적으로만 해석한다. Historical 정책은 약 1,970 train cases에 노출된 in-sample 진단 대상이며 reserve 또는 일반화 근거로 사용하지 않는다.

금지 경계는 다음과 같다.

- future reserve 170 subjects/cases의 bundle, outcome, row-level statistic을 읽지 않는다. 기존 frozen count/hash만 확인한다.
- historical sealed test와 journal test 자료를 읽거나 평가하지 않는다.
- Phase B 장시간 PPO 재학습, Phase C patient--observer mismatch/SQI artifact/recurrent policy 구현은 하지 않는다.
- case/subject identifier, private row, model/checkpoint, 원자료는 공개 산출물이나 Git에 기록하지 않는다.

## 확인된 사실

- 시작 HEAD는 요청된 Day 06 커밋 `8b00a4ee1ed69afb92e8303d80504fca7f023465`이며 시작 작업 트리는 clean이었다.
- 새 작업 브랜치는 `journal/phase-a-diagnostics`이다.
- Day 04--06 ignored output에는 final/checkpoint 모델과 development/train-diagnostic 집계 및 private row-level 평가가 존재한다. 기존 private row에는 0--600 s와 600--1,800 s의 MAE, TIR, high/low BIS, dose, boundary/clipping 지표가 이미 포함되어 있다.
- simulator `t=0`은 anesthesia start에 상대화된 template/schedule의 0초이며 reset 시 Cp/Ce와 투여 history는 0이다. BIS/SQI와 remifentanil source timestamps는 anesthesia start를 빼서 상대화된다.
- 관측값은 마지막 accepted BIS를 staleness cap까지 carry-forward하며 age는 simulator query time과 event timestamp의 차이다. state history의 음수 offset은 prior event가 없으면 mask 0/value 0으로 채운다.
- action은 매 10초 구간 시작에 결정되어 그 구간의 propofol 투여율로 적용된다. action bounds는 0--27.7 mg/10 s이다.
- public Day 04 visibility audit와 private development outputs는 존재하지만, 실제 propofol raw administration track은 runtime bundle에 없다. simulator의 첫 propofol 투여는 policy의 첫 action이며 임상 기록의 첫 propofol 투여 시각이 아니다.

## 아직 확인하지 않은 추정

- 0--600초 낮은 visibility가 주로 anesthesia-start 이전/이후 센서 기록 범위와 SQI gating 중 어느 쪽에서 생기는지는 구간별 event audit로 계산해야 한다.
- checkpoint actor의 첫 층 preactivation과 BIS-only paired counterfactual 반응은 모델을 직접 계측해야 한다.
- BIS-off 및 mask/visibility 반사실은 동일 latent trajectory에서 one-step action 변화로만 해석 가능하며, 닫힌루프 환경 관측과 동등하지 않을 것으로 예상한다.
- Cp/Ce 입력 perturbation은 policy-input-only stress test로 수행 가능하다. true dynamics와 observer를 분리한 일관된 mismatch simulator는 이번 범위 밖이다.
- nominal TCI와 BIS PI trim은 현재 deterministic PK/PD simulator 위에서 최소 비교군으로 구현 가능해 보이나, 목표 Ce와 gain은 development train 진단 subset에서만 동결해야 한다.

## 실행 계획과 대략 비용

| 순서 | 진단 | 계획 | 예상 CPU wall time |
|---|---|---|---:|
| A4 | 시간 정렬/구간 성능 | 코드 provenance + development aggregate 재계산 + visibility event audit | 1--10분 |
| A3 | numerical conditioning | historical P0S0/P1S0 final actor 계측, BIS-only paired delta, 동일 구조 새 초기화 probe | 10--40분 |
| A1 | 관측 반사실 | journal S_CONC final 정책의 BIS-off/mask/value perturbation one-step delta action | 10--40분 |
| A2 | 농도/PD probe | Cp/Ce bias/scale/lag 대용 입력 perturbation의 paired delta action 및 실패 패턴 | 10--40분 |
| A5 | 비학습 비교군 | development-train에서 고정한 최소 Ce-target TCI와 nominal TCI+BIS PI trim을 development-validation에서 평가 | 20--90분 |

모든 실행은 1 worker/1 Torch thread를 사용한다. 출력에는 단계별 completion marker와 source checksum을 넣고, 기존 출력이 동일하면 재사용하며 불일치하면 중단한다. A5가 simulator 제약으로 완전하지 않으면 작동하는 최소 비교군과 부족한 observer/실측 투여 경계를 별도로 기록한다.

## 완료 조건

- 각 A1--A5에 질문, 사용 자료, 수치, 실패 사례, 주장 강도, 미수행 범위를 남긴다.
- 공개 aggregate에는 identifier가 없어야 하고 reserve/test access count가 0이어야 한다.
- 관련 단위 테스트, 기존 전체 테스트, 접근 경계 감사를 통과한다.
- 한국어 최종 보고서와 미팅용 1페이지 요약을 작성하고 Phase B 실행 여부는 제안으로만 적는다.
