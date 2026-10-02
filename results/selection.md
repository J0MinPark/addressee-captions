# 최종 구성 선택 (개발 세트만 사용)

**규칙(사전 고정)**: dev(3개 조건 합산) single 정의 F0.5 점추정 최대(후보: 모든 판정기·융합·플래그 조합과 LLM 없는 timing·timing_speaker). 학습된 융합은 착용자 단위 6겹 교차검증 추정으로 비교한다. 동률이면 단순한 구성(LLM 없음 → 손 가중치 → 플래그 없음) 우선.

**선택: `P1-qwen3:4b-learned`** — P1-qwen3:4b | learned | none

- dev F0.5 = 0.221 (timing 0.185, v1 구성 0.167)
- 정밀도 20.5%, 재현율 32.9%, 자연 함정 오표시율 40.7%
- timing 대비 짝지은 차이 +0.037 [+0.006, +0.064]

상위 5개:

1. P1-qwen3:4b | learned | none — F0.5 0.221
2. P2-qwen3:4b | learned | none — F0.5 0.220
3. P1-qwen3:4b | learned | rejudge — F0.5 0.216
4. P2-qwen3:4b | learned | rejudge — F0.5 0.208
5. P3-qwen3:4b | learned | none — F0.5 0.203

판정기 단독 비교는 `results/judge_dev.md`, 전체 표는 `results/dev_results.md`.
