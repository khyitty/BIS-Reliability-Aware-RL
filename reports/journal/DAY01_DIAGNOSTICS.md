# Day 01 controller 진단

고정 1.5 mg/10s와 비튜닝 proportional comparator(`base=1.5`, `gain=0.08`)를 동일한 합성 validation subject에 적용했다. 임상 PID나 공정한 최적 baseline으로 주장하지 않는다.

| condition | constant MAE | P-like MAE | constant BIS 40–60 | P-like BIS 40–60 |
|---|---:|---:|---:|---:|
| Goff_A30_S0 | 14.399 | 5.600 | 0.542 | 0.837 |
| Goff_A20_S0 | 14.399 | 5.706 | 0.542 | 0.833 |
| G50_A30_S0 | 14.399 | 6.264 | 0.542 | 0.825 |
| G50_A20_S0 | 14.399 | 6.380 | 0.542 | 0.817 |
| Goff_A30_S1 | 14.399 | 5.600 | 0.542 | 0.837 |
| Goff_A20_S1 | 14.399 | 5.706 | 0.542 | 0.833 |
| G50_A30_S1 | 14.399 | 6.264 | 0.542 | 0.825 |
| G50_A20_S1 | 14.399 | 6.380 | 0.542 | 0.817 |

고정-rate 결과가 모든 관측 조건에서 같아야 한다는 불변식과 core clip fraction 0을 확인했다. P-like 차이는 오직 각 gate/age 규칙이 제공한 현재 BIS/mask에서 생긴다. 이 진단은 PPO 비교의 우월성 검정이 아니다.
