# 판정기 단독 비교 · AMI dev

착용자 직후 구간 중 라벨 y/n(single 정의) 922개 · 시나리오 24개
Whisper large-v3-turbo 를 같은 GPU에 올린 상태에서 측정(Whisper 전 (3986, 8188), 후 (5189, 8188) MB 사용/전체).

| 판정기 | AUC | 임계 0.5 정확도 | 최적 임계값 | 최적 정확도 | 지연 중앙/p95 | 모델 VRAM | 적재 | 동시 탑재(Whisper+LLM) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| qwen3:4b · P1c (v1, confidence 매핑) | 0.541 | 69% | 0.50 | 69% | – | – | GPU | – |
| qwen3:8b · P1 | 0.613 | 62% | 1.00 | 83% | 165/174ms | 5.58GB | GPU | 7489/8188MB · 동시 탑재 OK |
| qwen3:8b · P2 | 0.631 | 50% | 1.00 | 83% | 164/174ms | 5.58GB | GPU | 7478/8188MB · 동시 탑재 OK |
| qwen3:8b · P3 | 0.582 | 45% | 1.00 | 84% | 174/184ms | 5.58GB | GPU | 7492/8188MB · 동시 탑재 OK |

연속 점수: Ollama logprobs 사용 가능(true/false 토큰 확률).
![PR](llm_v2/dev/pr_curves.png)
