# 판정기 단독 비교 · AMI dev

착용자 직후 구간 중 라벨 y/n(single 정의) 2516개 · 시나리오 72개
Whisper large-v3-turbo 를 같은 GPU에 올린 상태에서 측정(Whisper 전 (3989, 8188), 후 (5180, 8188) MB 사용/전체).

| 판정기 | AUC | 임계 0.5 정확도 | 최적 임계값 | 최적 정확도 | 지연 중앙/p95 | 모델 VRAM | 적재 | 동시 탑재(Whisper+LLM) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| qwen3:4b · P1c (v1, confidence 매핑) | 0.540 | 68% | 0.50 | 68% | – | – | GPU | – |
| qwen3:4b · P1 | 0.586 | 62% | 1.00 | 84% | 168/181ms | 3.18GB | GPU | 5186/8188MB · 동시 탑재 OK |
| qwen3:4b · P2 | 0.584 | 28% | 1.00 | 84% | 178/186ms | 3.18GB | GPU | 5170/8188MB · 동시 탑재 OK |
| qwen3:4b · P3 | 0.585 | 32% | 1.00 | 84% | 182/192ms | 3.18GB | GPU | 5170/8188MB · 동시 탑재 OK |

연속 점수: Ollama logprobs 사용 가능(true/false 토큰 확률).
![PR](llm_v2/dev/pr_curves.png)
