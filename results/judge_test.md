# 판정기 단독 비교 · AMI test

착용자 직후 구간 중 라벨 y/n(single 정의) 489개 · 시나리오 48개
Whisper large-v3-turbo 를 같은 GPU에 올린 상태에서 측정(Whisper 전 (4083, 8188), 후 (5181, 8188) MB 사용/전체).

| 판정기 | AUC | 임계 0.5 정확도 | 최적 임계값 | 최적 정확도 | 지연 중앙/p95 | 모델 VRAM | 적재 | 동시 탑재(Whisper+LLM) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| qwen3:4b · P1c (v1, confidence 매핑) | 0.552 | 61% | 0.50 | 61% | – | – | GPU | – |
| qwen3:4b · P1 | 0.615 | 58% | 0.94 | 69% | 156/168ms | 3.18GB | GPU | 5181/8188MB · 동시 탑재 OK |

연속 점수: Ollama logprobs 사용 가능(true/false 토큰 확률).
![PR](llm_v2/test/pr_curves.png)
