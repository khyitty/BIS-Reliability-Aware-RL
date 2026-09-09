# Day 03 Scale Mechanism Results

## 결론

Nscale은 S1의 BIS MAE를 모든 paired seed에서 N0보다 낮췄고, Nscale 조건의 S1도 모든 seed에서 S0보다 낮았다. 수치 conditioning 문제와 scaling 후 유용한 S1 정보라는 설명을 이 제한된 설정에서 지지한다. S0도 Nscale에서 모든 seed가 소폭 개선되어 효과를 S1에만 고유하다고 볼 수는 없다.

이 결과는 고정된 24명 내부 검증셋을 재사용한 탐색적 개발 근거이며 임상 중재 효과, 독립 검증, 광범위한 PK/PD 우월성 또는 새로운 방법론 기여를 입증하지 않는다.

## 완성된 비교

| 변환 | 상태 | BIS MAE ± SD | 40–60 비율 | <40 비율 | >60 비율 | 총 propofol (mg) | action 경계 비율 | 출처 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| N0 | S0 | 12.2267 ± 0.1974 | 0.641 | 0.015 | 0.344 | 96.86 | 0.000 | 재사용 |
| N0 | S1 | 25.5805 ± 14.9629 | 0.300 | 0.000 | 0.700 | 47.96 | 0.065 | 재사용 |
| Nclip | S0 | 12.2236 ± 0.1556 | 0.646 | 0.003 | 0.352 | 94.29 | 0.000 | 신규 |
| Nclip | S1 | 6.5218 ± 0.2523 | 0.825 | 0.000 | 0.175 | 99.45 | 0.017 | 재사용 |
| Nscale | S0 | 11.7340 ± 0.2935 | 0.649 | 0.005 | 0.346 | 95.72 | 0.004 | 신규 |
| Nscale | S1 | 5.9965 ± 0.5946 | 0.820 | 0.000 | 0.180 | 98.41 | 0.052 | 신규 |

## Paired seed 차이 (BIS MAE)

음수는 표의 비교식에서 앞 변환/상태가 더 낮은 MAE임을 뜻한다.

| 비교 | 조건 | seed 45 / 46 / 47 |
|---|---|---|
| S1_minus_S0 | N0 | +23.3369 / +20.8075 / -4.0828 |
| S1_minus_S0 | Nclip | -5.4364 / -5.8467 / -5.8223 |
| S1_minus_S0 | Nscale | -5.5785 / -6.1007 / -5.5331 |
| Nclip_minus_N0 | S0 | +0.0308 / +0.0147 / -0.0547 |
| Nscale_minus_N0 | S0 | -0.4100 / -0.6417 / -0.4264 |
| Nclip_minus_N0 | S1 | -28.7425 / -26.6395 / -1.7942 |
| Nscale_minus_N0 | S1 | -29.3254 / -27.5499 / -1.8767 |

고정 PI baseline의 validation BIS MAE는 약 11.897이다. paired 차이는 세 seed를 그대로 제시하며 seed 강건성을 subject-only bootstrap으로 과장하지 않았다.

## 학습 업데이트 회계

Night 02 재사용 모델은 262,144개 전이를 수집했지만 callback 경계 때문에 260,096개(127 rollout)만 업데이트에 사용했다. 저장 모델의 `_n_updates=1,270`은 PPO epoch 수이다. Day 03 신규 모델은 260,096개를 수집하고 마지막 rollout 업데이트 뒤 저장해 같은 127 rollout/1,270 epoch에 맞췄다.

## 해석 한계

Nclip은 모든 좌표를 ±10으로 잘라 현재 scaler에서 BIS 40/50/60을 모두 10으로 만든다. Nscale은 훈련 보정분포의 좌표별 q95로 나누어 순서와 구분을 보존한다. 두 개입의 차이 때문에 성능 변화만으로 단일 원인을 확정할 수 없다. 세 seed 결과는 seed 강건성의 제한적 점검이며 subject-only bootstrap으로 이를 대체하지 않았다.

## 실행 시간

전체 sprint wall time은 82.0분이다.

## 프로젝트 진행

| 항목 | 상태 |
|---|---|
| 데이터 연결/실행 인프라 | 구축 완료 |
| 96-case / 10분 pilot | 완료 |
| scale mechanism | 완료 |
| 장시간/관측 풍부 평가 | 대기 |
| 더 큰 독립 검증 및 최종 저널 방법 기여 | 대기 |

표준 정규화 개선은 필요한 baseline 교정이며, 그 자체만으로 Neural Networks 저널의 새로운 방법론 기여를 뜻하지 않는다.

## 검증

두-rollout 경계 점검은 4,096개 전이/20 epoch와 저장 전후 정책의 정확한 일치를 확인했다. Day 03 검증기는 신규 post-update 모델 9개, 재사용 모델 9개, 18개 matrix cell, factor 불변식과 공개 산출물 privacy scan을 확인한다.

## 재현 및 개인정보

사례/대상자 식별자, calibration 배열, 모델, checkpoint와 로그는 git-ignored 출력 경로에만 남겼다. test set 접근 횟수는 0이다. 재개 명령은 `.venv-journal\Scripts\python.exe scripts\journal\run_day03_scale.py resume --minutes 145 --workers 1`이며 새 실행마다 명시적 시간 예산을 요구한다.
