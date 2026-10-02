# 한국어 판정 프롬프트 점검과 P1 버그 수정 (2026-10-02, 서버)

- 모델 qwen3:4b (Ollama 0.32.3, 127.0.0.1:11435, GPU 2), temperature 0, `think: false`, JSON 스키마 강제, logprobs top 10.
- 영어 프롬프트는 바꾸지 않았다(`tests/test_prompts.py`가 영어 4개 변형의 버전 해시를 고정).
- AI Hub 데이터는 쓰지 않았다. 점검에는 직접 작성한 한국어 쌍 48개와 그 영어 역번역(`data/judge_pairs_ko_check.csv`)을 썼다.
- 재현: `source scripts/server_env.sh && python tools/ko_prompt_check.py` → `results/ko_prompt_check.json`

## 1. 한국어판 ↔ 영어판 문장 대조 (한국어판의 영어 역번역)

| 위치 | 한국어판 | 한국어판 역번역 | 영어판 | 의미 차이 |
|---|---|---|---|---|
| 공통 1 | 너는 대화 분석기다. | You are a conversation analyzer. | You are a conversation analyzer. | 없음 |
| 공통 2 | A가 방금 말했고, 그 직후 B가 말했다. | A just spoke, and B spoke right after that. | A has just spoken, and B spoke right after. | 없음 |
| P1c·P1 3 | B의 말이 A의 말에 대한 응답으로서 인접쌍(질문→대답, 인사→인사, 요청→수락/거절, 제안→응답, 평가→반응)을 이루는지 판단하라. | Judge whether B's utterance, as a response to A's utterance, forms an adjacency pair (question→answer, greeting→greeting, request→accept/decline, proposal→response, assessment→reaction). | Decide whether B's utterance forms an adjacency pair as a response to A's utterance (question→answer, …). | 없음 |
| P1c·P1 4 | B가 A가 아닌 다른 사람에게 말하거나 A의 말과 무관한 주제를 말하면 pair는 false다. | If B speaks to someone other than A, or speaks about a topic unrelated to A's utterance, pair is false. | If B is talking to someone other than A, or about a topic unrelated to what A said, pair is false. | 없음 |
| P1c 5 | JSON만 출력하라. | Output JSON only. | Output JSON only. type must be one of: 질문-대답 (question-answer), … 없음 (none). | 영어판만 type 값 목록을 문장으로 설명한다(값이 한국어라 영어 모델에 풀이를 붙인 것). 한국어판은 스키마 enum으로만 강제. 의미 차이 없음 |
| P1·P2·P3 끝 | JSON {"pair": true\|false}만 출력하라. | Output only JSON {"pair": true\|false}. | Output only JSON {"pair": true\|false}. | 없음 |
| P2 3 | B의 말이 A의 말에 대한 반응(대답, 동의·반대, 이어받기, 맞장구, 되묻기)인가, 아니면 다른 사람이나 다른 화제를 향한 말인가? | Is B's utterance a reaction to A's utterance (an answer, agreement/disagreement, carrying on, a backchannel, asking back), or is it directed at another person or another topic? | Is B's utterance a reaction to A's utterance (an answer, agreement or disagreement, taking up the point, a backchannel, or a clarification question), or is it directed at someone else or at a different topic? | 근사 일치("이어받기"≈taking up the point, "되묻기"≈clarification question) |
| P2 4 | A의 말에 대한 반응이면 pair는 true, 다른 사람이나 다른 화제를 향한 말이면 false다. | If it is a reaction to A's utterance, pair is true; if directed at another person or topic, false. | If it is a reaction to A, pair is true; if it is directed at someone else or a different topic, pair is false. | 없음 |
| P3 추가 | 이전 대화의 각 줄 앞에는 화자가 표시된다: A = 방금 말한 사람(착용자), B = 판정 대상 화자, X1·X2… = 그 밖의 사람. 이전 대화를 참고해 B가 누구에게, 무엇에 대해 말하는지 판단하라. | Each line of the previous conversation is prefixed with its speaker: A = the person who just spoke (the wearer), B = the speaker being judged, X1, X2… = others. Using the previous conversation, judge whom B is talking to and about what. | (같은 내용) | 없음 |
| 사용자 메시지 | 이전 대화: / A(방금): … / B(직후): … | Previous conversation: / A (just now): … / B (right after): … | Previous conversation: / A (just said): … / B (right after): … | 없음 |

**few-shot 대응**

| 변형 | 한국어판 few-shot | 영어판 few-shot | 1:1 번역인가 |
|---|---|---|---|
| P1c (v1) | 일상·카페 6개(질문-대답, 인사, 요청 / 다른 사람에게 ×2, 다른 화제) | 회의 6개(질문-대답, 제안, 요청 / 다른 사람에게 ×2, 다른 화제) | 아니다 — 의도된 차이(v1 한국어가 원본, 영어판은 AMI용으로 새로 씀) |
| **P1** | **v1 한국어 few-shot을 그대로 재사용** | 영어 P1 = 영어 v1 few-shot | **아니다 — 버그.** README는 "영어·한국어판이 1:1로 대응한다"고 적었지만 P1만 대응하지 않았다 |
| P2 | 회의 6개(대답·되묻기·반대 / 다른 사람에게·다른 화제·통화) | 같은 6개의 영어 | 1:1 |
| P3 | P2 + 화자 표시 이전 대화 | 같은 것의 영어 | 1:1 |

## 2. 원인: "혹시 여기 자리 있어요? → 아니요, 비어 있어요" 가 낮게 나오는 이유

이전 PC에서는 0.08이었고, 이 서버에서 수정 전 코드로 재현하면 0.18이다. 낮은 쪽이라는 점은 같다.
차이는 환경 차이로 본다(이 서버 Ollama 0.32.3, README에 적힌 이전 PC Ollama는 0.35).

**(a) logprob 추출 — 원인 아님.** 원시 응답을 확인했다.
- 출력은 `{"pair": false}`이고, `pair` 다음 토큰은 ` false`(−0.154)이며 상위 후보에 ` true`(−1.97)가 있다.
- `p_pair_from_logprobs`는 공백을 떼고 비교해 이 두 토큰을 정확히 잡는다(p = 1/(1+e^(−0.154+1.97)) = 0.14).
- 출력 JSON과 logprob 판단은 항상 일치했다. 첫 토큰 `{`는 스키마 강제라 모델 선호("Okay", 사고 모드 습관)와 다르지만, 영어판도 같다.

**(b) 번역 — 원인 아님.**
- 시스템 프롬프트의 한국어판과 영어판은 문장 단위로 의미가 같다(1절).
- 같은 문장을 영어 P1에 그대로 넣으면 0.75, 역번역 영어 문장을 넣으면 1.00이다.

**(c) few-shot 구성 — 원인.** 같은 시스템 프롬프트에서 few-shot만 바꿨다(문제 쌍 + 시연 쌍 5개):

| 조건 | 자리→아니요, 비어 있어요 | 자리→네, 비어 있어요 | 자리→아니요, 앉으셔도 돼요 | 와이파이→카운터 옆에 | 커피→라떼가 괜찮아요 | 와이파이→야 지훈아(함정) |
|---|---:|---:|---:|---:|---:|---:|
| 한국어 P1 수정 전(v1 한국어 few-shot) | 0.18 | 1.00 | 0.86 | 0.66 | 0.84 | 0.03 |
| 한국어 P1, few-shot 없음 | 0.96 | 1.00 | 0.96 | 0.99 | 0.99 | 0.00 |
| 한국어 P1, '자리' 짝 아님 예시만 교체 | 0.33 | 1.00 | 0.60 | 0.82 | 0.87 | 0.03 |
| 한국어 P1, 영어 few-shot 1:1 번역 | 0.39 | 0.99 | 0.49 | 0.26 | 0.70 | 0.03 |
| 영어 P1 프롬프트 + 한국어 문장 | 0.75 | 0.95 | 0.68 | 0.52 | 0.94 | 0.01 |
| 영어 P1 + 역번역 영어 문장 | 1.00 | 0.99 | 1.00 | 0.97 | 0.99 | 0.01 |

- 한국어 few-shot이 붙으면 짝 점수가 전반적으로 눌린다. 특히 "아니요"로 시작하는 대답이 크게 떨어진다("네, 비어 있어요"는 1.00인데 "아니요, 비어 있어요"는 0.18).
- 수정 전 few-shot의 짝 아님 예시 "여기 앉아도 돼요? → 엄마, 나 오늘 좀 늦을 것 같아"는 시연의 '자리' 질문과 같은 질문이다. 이 예시만 바꿔도 0.18에서 0.33으로 오른다(일부 기여).
- few-shot을 빼면 바로 해결된다. 영어 few-shot을 한국어로 옮겨도 해결되지 않는다(4b 모델에서 한국어 대화 예시 자체가 짝 점수를 낮추는 경향).

**점검 48쌍**(짝 24: 네/아니요 대답·인사·요청 수락/거절·제안·평가 / 짝 아님 24: 다른 사람에게·통화·다른 화제·같은 화제어 함정):

| 변형 | 언어 | AUC | 정확도@0.5 | 짝 평균 | 짝 아님 평균 | 영어판과 MAD |
|---|---|---:|---:|---:|---:|---:|
| P1c | 영어(역번역) | 0.938 | 0.938 | 0.950 | 0.163 | – |
| P1c | 한국어 | 0.917 | 0.917 | 0.912 | 0.163 | 0.094 |
| P1 | 영어(역번역) | 1.000 | 1.000 | 0.972 | 0.037 | – |
| **P1** | **한국어 수정 전** | 0.983 | **0.792** | **0.582** | 0.020 | **0.207** |
| **P1** | **한국어 수정 후** | 1.000 | 1.000 | 0.950 | 0.071 | **0.050** |
| P2 | 영어(역번역) | 1.000 | 0.979 | 0.984 | 0.140 | – |
| P2 | 한국어 | 0.993 | 0.938 | 0.888 | 0.141 | 0.083 |
| P3 | 영어(역번역) | 1.000 | 0.958 | 0.981 | 0.186 | – |
| P3 | 한국어 | 0.993 | 0.896 | 0.749 | 0.101 | 0.164 |

수정 전 한국어 P1은 순위 정보(AUC 0.983)는 있었지만, 점수가 아래로 밀려 있었다(짝 평균 0.58).
- 이 상태에서 영어 회의로 학습한 융합 가중치·임계값과 겹치면 표시가 거의 사라진다. README 6-2절에 적힌 "한국어 합성 시연에서 아무것도 표시하지 않음"과 맞는다.

## 3. 수정

- **선택 기준**(결과를 보기 전에 `tools/ko_prompt_check.py`에 고정): 한국어판은 영어판의 번역이어야 하므로, 영어 P1(바꾸지 않음)이 역번역 문장에 내는 점수와의 평균 절대 차이(MAD)가 가장 작은 후보를 고른다. 동률(0.01 이내)이면 AUC.
- 후보 4개(고정): K0 수정 전 / K1 few-shot 없음 / K2 영어 few-shot 1:1 번역 / K3 '자리' 예시만 교체.

| 후보 | AUC | 정확도@0.5 | 짝 평균 | 짝 아님 평균 | 영어판과 MAD | 자리→아니요 |
|---|---:|---:|---:|---:|---:|---:|
| K0 수정 전 | 0.983 | 0.792 | 0.582 | 0.020 | 0.207 | 0.176 |
| **K1 few-shot 없음 (선택)** | 1.000 | 1.000 | 0.950 | 0.071 | **0.050** | 0.945 |
| K2 영어 few-shot 1:1 번역 | 0.995 | 0.792 | 0.621 | 0.034 | 0.187 | 0.399 |
| K3 자리 예시만 교체 | 0.993 | 0.792 | 0.650 | 0.032 | 0.172 | 0.342 |

**수정 전**(`app/llm_judge.py`):
```python
_SHOT_P1_KO = [(ex_in, {"pair": ex_out["pair"]}) for ex_in, ex_out in FEW_SHOT]   # v1 한국어 few-shot 6개 재사용
```
**수정 후**:
```python
_SHOT_P1_KO_OLD = [...]   # 기록용(쓰지 않음)
_SHOT_P1_KO: list = []     # 한국어 P1은 few-shot 없이: 시스템 프롬프트 + 사용자 메시지(이전 대화·A·B)만
```
- 시스템 프롬프트(`_P1_KO`)는 바꾸지 않았다(1절에서 영어판과 의미가 같음을 확인).
- **프롬프트 버전 해시**: 한국어 P1 `243b6e2a → 437056c3`. 캐시 키가 (모델, 프롬프트 버전, 입력)이라 이전 캐시는 쓰이지 않는다. 결과 파일 메타(`llm_prompt_version`)에도 새 해시가 찍힌다.
- 바뀌지 않은 것: 영어 P1c/P1/P2/P3(e389a86c / b627183d / 0cc65e8f / 2faa83e5), 한국어 P1c(29f29731, v1 = 시연 기본), 한국어 P2(e8bd838f), 한국어 P3.

**바꾸지 않은 것과 이유**
- **한국어 P3**: 짝 점수가 낮은 쪽으로 밀린다(MAD 0.164). 그러나 few-shot이 영어판의 1:1 번역이고 문장 의미도 같다. 번역 버그가 아니라 모델 특성이므로 기록만 한다.
  P3는 few-shot이 화자 표시 형식을 보여 주는 역할도 하므로 빼지 않았다. AI Hub dev에서 후보로 그대로 비교된다(학습된 융합은 점수 이동을 흡수할 수 있다).
- **한국어 P2**: MAD 0.083로 영어판과 가깝다.
- **v1(P1c)**: 시연 기본 구성이고 의도된 원본이라 바꾸지 않는다.

**한계**: 점검 쌍은 직접 작성한 짧은 문장 48개다. 실제 구어(ASR 오류, 맞장구, 긴 발화)에서의 효과는 AI Hub dev(P2 단계)에서 판정기 후보 비교로 확인한다.
