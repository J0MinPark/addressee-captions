# 나에게 온 말만 보여주는 자막

시끄러운 곳에서 주변 말을 전부 받아 적는 대신, **착용자와 대화 중인 사람의 말만** 크게 보여 주는 실시간 자막 시스템.
"이 발화는 착용자에게 하는 말인가?"를 세 가지 증거로 판정한다.

| 증거 | 무엇 | 경로 |
|---|---|---|
| **T 타이밍** | 착용자 말이 끝난 직후(0~1초) 시작했는가 | 빠름 |
| **S 화자** | 이미 대화 상대로 등록된 목소리인가 (ECAPA) | 빠름 |
| **L 의미** | 착용자의 마지막 말과 질문→대답 같은 인접쌍인가 (로컬 LLM) | 느림(0.3~0.9초) |

`z = b + w_t·T + w_s·S + w_l·(2L−1)`, `prob = sigmoid(z)`. LLM 결과 전에는 T·S만으로 임시 표시("판정 중")하고, 결과가 오면 갱신한다.
위험 소리(사이렌·화재경보·경적)와 착용자 이름 호명은 판정과 무관하게 항상 알린다. 출력은 화면 자막(노트북 대시보드 + 폰)과 폰 진동뿐이다.
모든 모델은 로컬에서 돈다. 모델을 받은 뒤에는 인터넷이 필요 없다.

---

## 1. 설치 (Windows 11 / Ubuntu 22.04, Python 3.11)

```bash
python -m venv .venv            # ⚠ OneDrive 폴더 안이면 venv와 모델은 OneDrive 밖에 두세요(아래 참고)
.venv\Scripts\activate          # Ubuntu: source .venv/bin/activate
pip install -r requirements.txt # torch cu124 휠 포함(약 3GB). GPU 없는 노트북에서도 그대로 동작
```

**Ollama** 설치: <https://ollama.com/download> (Windows: `winget install Ollama.Ollama`).
시연 때는 **모델을 한 번에 하나만** 올리도록 띄운다(VRAM 8GB에서 Whisper·AST와 공존, 다른 모델이 VRAM을 잡아 판정이 멈추는 사고 방지):

```bash
scripts\start_ollama.bat      # Windows: 트레이 앱·고아 러너(llama-server.exe)까지 종료 후 OLLAMA_MAX_LOADED_MODELS=1 로 serve
./scripts/start_ollama.sh     # Ubuntu (systemd 서비스면 Environment="OLLAMA_MAX_LOADED_MODELS=1")
scripts\run_demo.bat          # Ollama(없으면 위 방식으로) → preflight → 서버, 한 번에
```
영구 설정: `setx OLLAMA_MAX_LOADED_MODELS 1` 후 트레이 앱 재시작.
서버는 시작할 때 config 모델이 아닌 로드된 모델을 내리고(`keep_alive: 0`), 실제 판정 1회로 지연을 재서
`llm=qwen3:4b ok 540ms (GPU)` 또는 `llm=FAIL <원인>` 을 크게 출력한다. 이후 10초마다(실패 중엔 2초마다) `/api/ps`로
모델 이름·GPU 적재를 확인하고, 대시보드 상단 칩(`LLM: qwen3:4b · GPU · 0.5s`)과 빨간 배너("LLM 끊김 — 의미 판정 꺼짐")로 보여 준다.

**모델 미리 받기 (행사장 네트워크를 믿지 않는다):**

```bash
python scripts/download_models.py          # Whisper large-v3-turbo + small, ECAPA, AST, Ollama qwen3:4b + qwen3:1.7b
python scripts/download_models.py --check  # 오프라인으로 전부 로드 + 워밍업 시간 출력 (M0 완료 기준)
```

정상 출력 예 (RTX 4060):
```
[load] 워밍업 시간(s): vad_a=1.13, vad_b=0.06, speaker=0.85, asr=3.19, sound=1.95, llm=0.4
[load] 사용 모델: vad=silero, speaker=ecapa, asr=large-v3-turbo/cuda/int8_float16, sound=ast/cuda, llm=qwen3:4b
```
받은 뒤에는 `models/.downloaded` 가 생기고 이후 실행은 HF 오프라인 모드로 동작한다.

> **OneDrive/Dropbox 폴더라면**: 모델(~3GB)이 동기화되지 않게 환경변수로 다른 곳을 지정하세요.
> `setx HEARME_MODELS_DIR C:\hearme_models` (새 터미널부터 적용). venv도 OneDrive 밖에 만드는 게 좋다.

## 2. 장치 연결

1. 마이크 두 개: **A = 착용자 핀마이크**(입 가까이), **B = 주변 마이크**(가슴/테이블, 상대 쪽을 향하게).
2. `python -m app.devices` 로 이름 확인 → `app/config.yaml`:
   ```yaml
   audio:
     device_wearer: "Lavalier"   # 이름 일부(대소문자 무시)
     device_ambient: "USB Audio"
   ```
   Windows에서 같은 장치가 MME/DirectSound/WASAPI로 여러 번 보이면 WASAPI가 자동 우선된다. 장치가 16kHz를 지원하지 않으면 자동 리샘플.
3. 마이크가 하나뿐이면 `--single-mic`: 시작할 때 착용자가 10초 말해서 목소리를 등록하고, ECAPA 유사도로 본인 발화를 가린다(정확도 낮음, 폴백).

## 3. 폰 연결 — 휴대폰 핫스팟을 쓰세요

학교/행사장 와이파이는 **기기 간 통신(클라이언트 격리)을 막는 경우가 많다.** 휴대폰 핫스팟 하나에 노트북과 폰을 함께 연결한다.
서버를 켜면 터미널에 LAN 주소와 QR 코드가 나온다. 폰 카메라로 QR → `http://<IP>:8000/phone` → 화면을 한 번 눌러 진동/화면 켜짐 허용.
Windows 방화벽 창이 뜨면 **개인 네트워크 허용**. 그래도 안 되면 `netsh advfirewall firewall add rule name=hearme dir=in action=allow protocol=TCP localport=8000`.

## 4. 실행

```bash
python -m app.preflight --profile gpu_4060   # 사전 점검: 마이크·dB 차·GPU/VRAM·ASR·LLM·사이렌·LAN/QR → PASS/WARN/FAIL 표
python -m app.server --profile gpu_4060      # 실시간 시연 (RTX 4060)
python -m app.server --profile cpu_light     # GPU 없는 노트북: Whisper small/CPU, AST CPU, qwen3:1.7b
python tools/replay.py data/demo --realtime  # 마이크 없이 녹음 재생 + 같은 화면 (라이브 실패 시 백업)
python -m app.server --replay data/demo --realtime --loop   # 무한 반복(부스 시연)
```
- 대시보드: `http://<IP>:8000/` (프로젝터) · 폰: `http://<IP>:8000/phone`
- 옵션: `--mode full|timing|...`, `--no-llm`, `--no-llm-cache`, `--single-mic`, `--port 8001`
- 모든 이벤트는 `logs/run_<timestamp>.jsonl` 에 기록된다.

**대시보드**: 왼쪽 자막(대화 상대 = 화자 색 큰 글씨, 다른 대화 = 회색 한 줄·클릭하면 펼침, 착용자 = 작게 오른쪽), 오른쪽 화자 목록(**클릭 = 수동 등록/해제**), 소거 모드 5개 토글, 지연 지표, 판정 로그, 초기화 버튼. 위험 소리는 화면 전체 빨간 깜박임.

## 5. 임계값 보정 순서 (현장 도착 후 10분)

**데이터 분할 규칙**: 같은 시나리오를 역할을 바꿔 두 번 녹음한다. **1회차 `NAME_take1` = 보정용**, **2회차 `NAME_take2` = 평가용**.
`calibrate.py`는 `_take1`만 받고(쓴 녹음은 `results/calibration_used.json`에 기록), `evaluate.py`·`diff_modes.py`는 `_take2`만 기본으로 쓴다.
보정용 녹음이 평가 집합에 들어가면 오류로 멈춘다.

1. **본인 발화 마진**: 2채널로 30초 녹음(착용자 말 + 상대 말 섞어서) → 측정
   ```bash
   python tools/record.py --scenario cafe_take1 --seconds 30
   python tools/calibrate.py --ownvoice data/cafe_take1     # → ownvoice.own_margin_db 제안
   ```
   녹음 중 화면의 `A-B dB` 값이 착용자 말 때 +10dB 이상, 상대 말 때 음수여야 정상. 아니면 마이크 위치부터 고친다.
2. **화자 임계값**: 팀원 각자 20초씩 혼자 녹음(개인 등록 녹음은 분할 표시가 없어도 되지만 `_take2`는 거부) →
   ```bash
   python tools/calibrate.py --speaker 민수=data/spk1_B.wav --speaker 지영=data/spk2_B.wav --speaker 현우=data/spk3_B.wav
   ```
   → `speaker.spk_threshold` 제안(1초 구간 기준). 같은 사람이 여러 화자 번호로 쪼개지면 낮추고, 다른 사람이 합쳐지면 올린다.
3. **이름**: `wearer.name`, `wearer.name_variants` 를 착용자 이름으로.

## 6. 녹음 · 재생 · 라벨 · 평가 (발표용 표)

```bash
python tools/record.py --scenario cafe_trap_take2        # 평가용: data/cafe_trap_take2_A.wav, _B.wav, .json (Ctrl+C 종료)
python tools/replay.py data/cafe_trap_take2              # results/cafe_trap_take2.segments.jsonl (+ .emb.npz 특징 캐시)
python tools/label.py data/cafe_trap_take2               # y=나에게 n=아님 w=본인 s=건너뜀 p=듣기 b=이전 q=종료
python tools/evaluate.py                                 # _take2 전부 → results/ablation.md, ablation.csv
python tools/diff_modes.py --a full --b timing_speaker   # 판정이 갈린 구간 목록 → results/diff_full_vs_timing_speaker.md
```
- replay는 LLM을 착용자 직후 구간 **전부**에 호출해 결과를 segments.jsonl에 캐시한다(모델명·프롬프트 버전도 기록).
  evaluate·diff_modes는 모델 없이 5개 모드의 정책만 다시 돌린다. 시나리오마다 LLM 모델/프롬프트가 다르면 표에 경고가 붙는다.
  LLM 지연을 새로 재려면 `--no-llm-cache`.
- `ablation.md` 구성: **표 1 전체**(정밀도·재현율·F1·자막 오염도·등록 지연), **표 2 함정 시나리오만**(이름에 `trap`: 오등록률·함정 구간 오표시율),
  **표 3 LLM 혼동행렬**(착용자 직후 구간의 라벨 y/n × LLM 짝/짝 아님/결과 없음).
- **합성 데이터**(`data/NAME.json`의 `"synthetic": true`, 예: `data/demo`, `demo_trap`)로 만든 결과는 파일 이름과 표 제목에 `[SYNTHETIC]`이 붙고
  (`ablation_[SYNTHETIC].md`, `diff_..._[SYNTHETIC].md`) 실제 녹음 결과와 섞이지 않는다. **발표 자료에는 `[SYNTHETIC]` 없는 파일만.**
- 라벨러는 착용자 구간에 자동으로 `w`를 붙이고 건너뛴다. 키 하나로 넘어가며 소리도 자동 재생 → 200구간 10분 이내.
- 실제 녹음 전 개발용: `python tools/make_test_scenario.py --name demo_trap` (Windows 내장 한국어 음성으로 합성한 2채널 대화 + 정답) → `python tools/label.py demo_trap --auto`.

예시 결과 **[SYNTHETIC]** (합성 시나리오 `demo_trap`, 72초, 실제 모델 전부 사용 · 라벨 y 9 / n 6 — 발표용 아님, 경향만):

| 모드 | 정밀도 | 재현율 | F1 | 자막 오염도 ↓ |
|---|---:|---:|---:|---:|
| 전부 표시 | 60% | 100% | 0.75 | 38% |
| 타이밍 | 83% | 56% | 0.67 | 22% |
| 타이밍+화자 | 90% | 100% | 0.95 | 16% |
| 전체 융합 | 89% | 89% | 0.89 | 17% |
| 의미만 | 75% | 33% | 0.46 | 35% |

## 7. 시연 런북 (90초)

**발표 전 체크리스트** (순서대로, 발표 15분 전)

1. [ ] GPU 쓰는 프로그램(게임·영상 편집 등) 종료. 핫스팟 켜고 노트북·폰을 같은 핫스팟에 연결.
2. [ ] `scripts\start_ollama.bat` (트레이 Ollama·고아 러너 종료 후 모델 1개 모드로 실행. 켜지는 데 ~15초)
3. [ ] `python -m app.preflight --profile gpu_4060` → 안내에 따라 착용자/상대가 5초씩 말하기 → **모든 항목 PASS** 확인
   (FAIL이면 상세 칸의 조치대로. 마이크 녹음이 FAIL이면 5절 1번 보정)
4. [ ] 서버를 **새로** 시작(메모리·화자 상태 초기화): 떠 있던 서버는 Ctrl+C 후 `python -m app.server --profile gpu_4060`
   → 터미널 배너가 `llm=qwen3:4b ok …ms (GPU)` 인지 확인(`!!` 배너면 원인 해결 후 재시작)
5. [ ] 대시보드 상단 **상태 칩이 초록**(`LLM: qwen3:4b · GPU · 0.xs`)이고 빨간 "LLM 끊김" 배너가 없는지 확인
6. [ ] 폰으로 QR 접속 → 화면 한 번 터치(진동·화면 켜짐 허용) → 우상단 "연결됨" 확인
7. [ ] 대시보드 **초기화** → 모드 **전체 융합** → 프로젝터 연결
8. [ ] 백업 터미널에 `python tools/replay.py data/demo --realtime --port 8001` 준비(포트 다름, `data/demo`는 실제 시연 녹음으로 교체해 둘 것)
사이렌 소리는 다른 폰에 영상(유튜브 "ambulance siren" 등, **미리 오프라인 저장**)으로 준비.

| 시간 | 행동 | 화면에서 보여줄 것 |
|---|---|---|
| 0:00–0:10 | "카페에서 자막 앱을 켜면 모든 말이 다 뜹니다." 모드 **전부 표시** 클릭, 옆 팀원 둘이 잡담 | 잡담까지 큰 자막으로 다 뜸 |
| 0:10–0:15 | 모드 **전체 융합** 클릭, **초기화** | 화면 정리 |
| 0:15–0:35 | 착용자: "안녕하세요, 여기 자리 있어요?" → 상대(A): "네, 비어 있어요." | 상대 말 "판정 중" → 큰 색 자막, 배너 **대화 상대 #1 추가**, 근거 칩 "응답 0.3초 · 질문→대답 · 화자 #1" |
| 0:35–0:50 | **함정**: 착용자: "와이파이 비번 아세요?" → 바로 옆 사람(B)이 팀원에게: "야, 어제 경기 봤어?" | B의 말은 회색 한 줄로 **접힘**(칩: 짝 아님). 이어서 A가 대답하면 크게 표시 |
| 0:50–1:00 | 모드 **타이밍** 클릭 후 같은 함정 반복 | B가 대화 상대로 **잘못 등록**됨 → "타이밍만으로는 안 된다" |
| 1:00–1:10 | 사이렌 영상 재생 | 대시보드 빨간 깜박임 + **폰 진동** |
| 1:10–1:20 | 뒤에서 "민수야!" | 노란 호명 배너 + 폰 진동 (판정과 무관하게 항상) |
| 1:20–1:30 | 소거 표(`results/ablation.md`) 슬라이드 | 오염도·오등록률 비교 |

**라이브가 이상하면**: 즉시 백업 탭(`:8001`)으로 전환해 녹음 재생. 화자가 꼬이면 대시보드에서 화자를 **클릭해 수동 등록/해제**. LLM이 느리면 모드 **타이밍+화자**로(LLM 없이도 충분히 동작).

## 8. 자주 나는 오류와 해결

| 증상 | 원인 / 해결 |
|---|---|
| `Could not load symbol cudnnGetLibConfig` 후 종료 | cuDNN이 여러 벌(torch 번들, ctranslate2, pip nvidia) 섞임. `app.winsetup.setup()`이 torch를 먼저 import해 통일한다. 직접 스크립트를 짤 때도 **가장 먼저** `from app import winsetup; winsetup.setup()` |
| `cublas64_12.dll not found` / `cudnn_ops64_9.dll` | `pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"`. 그래도 안 되면 `--profile cpu_light` |
| ASR이 `small/cpu`로 떠 있음 | GPU 로딩 실패 시 자동 폴백. 터미널의 `[asr] ... 실패:` 메시지 확인 |
| `[llm] Ollama 연결 실패` | Ollama 실행 확인(`ollama list`). 없으면 LLM 없이 동작(타이밍 2회 교대 규칙으로 등록) |
| `설치된 모델 없음` | `ollama pull qwen3:4b` (cpu_light는 `qwen3:1.7b`) |
| 대시보드에 빨간 "LLM 끊김" 배너 | 원인이 배너 괄호에 나온다. 연결 실패 → Ollama 실행. 모델 불일치/CPU로 밀려남 → `scripts\start_ollama.bat`로 재시작. Ollama는 켜는 데 이 PC에서 ~14초 걸린다(GPU 탐색) — 켜지면 서버가 2초 안에 감지해 자동 복구 |
| Ollama를 껐다 켰더니 판정이 멈춤(GPU 100%) | `ollama.exe`만 종료하면 `llama-server.exe` 러너가 고아로 남아 VRAM을 쥔다 → 새 Ollama가 모델을 한 번 더 올려 VRAM이 넘치고 멈춤. `taskkill /IM llama-server.exe /F` (start_ollama.bat가 처리). **게임 등 GPU를 쓰는 프로그램은 시연 전 종료** |
| 시작 로그에 `llm=None` | LLM 없이 도는 중(full 모드가 타이밍 규칙으로만 등록). 다른 Ollama 모델이 VRAM을 잡고 있으면 실패할 수 있다 → `ollama ps` 확인, `ollama stop qwen3:1.7b` 후 재시작. **발표 전 시작 로그에서 `llm=qwen3:4b` 확인 필수** |
| LLM 타임아웃이 잦음 | 대시보드 "LLM 타임아웃" 증가 → `llm.timeout_s` 늘리거나 cpu_light 모델로. 첫 호출은 모델 로딩으로 느림(워밍업이 처리) |
| 폰이 접속 안 됨 | 같은 핫스팟인지, 방화벽 개인 네트워크 허용, 주소가 `http://`(https 아님)인지. 터미널에 IP가 여러 개면 핫스팟 대역(예: 172.20.x / 192.168.43.x) 것을 쓴다 |
| 폰 진동이 안 옴 | iOS Safari는 진동 API 미지원(화면 깜박임만). 안드로이드 크롬은 화면을 한 번 터치해야 허용 |
| `입력 장치 '...'를 찾을 수 없음` | `python -m app.devices`의 이름 일부로 config 수정. 블루투스 마이크는 통화 모드로 바뀌면서 이름이 달라질 수 있음 |
| 착용자 말이 "다른 대화"로 뜸 / 상대 말이 착용자로 잡힘 | `own_margin_db` 보정(5절 1). 녹음 화면의 A-B dB 확인 |
| 같은 사람이 화자 #1, #3으로 쪼개짐 | `speaker.spk_threshold` 낮추기(5절 2) 또는 대시보드에서 둘 다 수동 등록 |
| 자막에 "시청해 주셔서 감사합니다" 류 | `asr.blacklist`에 문구 추가 |
| 느린 사이렌(wail)을 못 잡음 | AST 1초 창에서는 느린 스윕이 단음처럼 보여 점수가 낮다. `sound.window_s: 2.0` 또는 `score_threshold` 조정 |
| 포트 사용 중 | `--port 8001` |
| SpeechBrain 심볼릭 링크 오류(Windows) | `LocalStrategy.COPY`로 받는다. 그래도 나면 `models/spkrec-ecapa-voxceleb` 지우고 다시 다운로드 |

## 9. 구조

```
app/config.yaml      모든 파라미터 + 프로필(gpu_4060, cpu_light). 임계값은 코드에 없다
app/audio_source.py  LiveSource(마이크 2개) / ReplaySource(WAV 2개) → 같은 Block → 같은 파이프라인
app/ownvoice.py      A·B dB 차 + VAD → 본인 발화(hangover 300ms)
app/segmenter.py     Silero VAD 스트리밍 → 주변 구간(0.3s~8s). 착용자 발화 중엔 B 분할을 가린다
app/speaker.py       ECAPA 임베딩 + 온라인 군집(코사인, EMA 0.8)
app/asr.py           faster-whisper 작업 스레드, 환각 필터, 핫워드 편향 재확인
app/sound_events.py  AST 위험 소리(최신 요청만 처리), 2회 연속 + 5초 쿨다운
app/namecall.py      한글 자모 분해 + 편집거리 호명
app/llm_judge.py     Ollama 인접쌍 판정(think:false, JSON 스키마, 2.5초, 캐시)
app/policy.py        정책 엔진(순수 로직, 시간은 인자) — 단위 테스트 대상
app/pipeline.py      스레드 연결: audio → control(정책은 한 스레드에서 순서대로) / asr / sound / llm
app/server.py        FastAPI + WebSocket 방송, LAN 주소·QR
app/preflight.py     발표 전 사전 점검(PASS/WARN/FAIL 표)
tools/               record, replay, label, evaluate, diff_modes, calibrate, datasplit(분할 규칙), make_test_scenario(개발용)
scripts/             download_models, start_ollama(.bat/.sh), run_demo.bat
assets/              preflight용 3초 음성·사이렌 샘플
tests/               policy (a)~(f), namecall, ownvoice, asr 필터, 합성 재생 통합 테스트
```
테스트: `python -m pytest -q tests` (모델 없이 돈다).

## 10. ASSUMPTIONS (애매한 부분에 정한 기본값)

1. **시간 기준**: 정책·평가는 "스트림 시간"(첫 샘플부터의 샘플 수/16000)을 쓴다. 장치 두 개의 클럭 차이는 버퍼 차가 0.5초를 넘으면 앞선 쪽을 버려 맞춘다. 지연 측정만 `time.monotonic()`.
2. **착용자 발화 중 B 분할**: B 마이크도 착용자 목소리를 듣기 때문에, 빠른 대답(<0.4초)이 착용자 말과 한 구간으로 합쳐졌다. 그래서 본인 발화가 진행되는 동안 B 분할을 가리고, 착용자가 말을 시작하면 열린 B 구간을 자르고, 끝나면 실제 종료+0.1초(잔향 가드)부터 B 프레임을 다시 넣는다. 50% 겹침 규칙은 그 뒤에도 남는 경우에 적용.
3. **구간 시간**: 타이밍 증거는 패딩 없는 말소리 시작 시각을 쓴다(오디오만 앞뒤 0.1초 패딩해 ASR에 넣음).
4. **화자 0.8~1.0초 구간**: 명세에 없는 구간이다. 임베딩을 계산하되 기존 화자와 맞을 때만 배정하고, 중심 갱신이나 새 화자 생성은 하지 않는다. 화자가 미상인 1.0초 미만 구간에는 상속 규칙(직전이 대화 상대이고 간격 ≤1초)을 적용한다. 상속된 구간의 유사도는 직전 구간 값을 쓴다.
5. **등록 규칙**: LLM을 쓰는 모드(full, semantic)는 LLM 결과 도착 뒤 `prob ≥ register_threshold` 이고 응답 타이밍(T≥0.5)이 있으면 등록한다. LLM을 쓰지 않는 모드(timing, timing_speaker)는 음향 점수가 곧 최종이라 즉시 같은 규칙을 적용한다. 그래서 timing 모드에서는 함정 한 번에 등록되고, 이것이 소거 실험의 핵심 비교다. LLM이 None이면(타임아웃·미설치) 서로 다른 착용자 발화 2개에 연속으로 응답해야 등록한다.
6. **all 모드**: 기존 자막 앱 흉내라서 모든 비착용자 발화를 partner로 표시하고 화자도 등록한다(오등록률 기준선). 다른 모드로 돌아갈 때는 **초기화**를 누를 것.
7. **교대(만료 기준)**: 상대가 착용자 말 직후(T≥0.5) 말하거나, 착용자가 상대 말 직후 1.5초 안에 말하면 교대로 친다. 상대가 혼자 계속 말하는 독백은 교대가 아니다(명세대로). 60초 넘게 독백하면 만료될 수 있다. 수동 등록한 화자는 만료되지 않는다. candidate는 60초 뒤 unknown으로 돌아간다.
8. **role**: `prob ≥ show_threshold` → partner. 아니면 화자가 있으면 other, 없으면 unknown(둘 다 회색으로 표시).
9. **ASR 빈 결과**(환각 필터, 큐 넘침으로 버림)인 구간은 표시하지 않고, 정책에도 넣지 않는다. 실시간에서 ASR 큐가 12개를 넘으면 가장 오래된 비착용자 작업부터 버린다. 최대 속도 재생에서는 버리지 않고 기다린다(평가 결정성).
10. **핫워드 편향**: `hotwords=이름`을 쓰면 Whisper가 "지훈아"를 "민수"로 바꾸거나 무음에서 "민수 씨 민수 씨…"를 만들었다. 그래서 (a) 이름만 반복된 출력은 버리고, (b) 결과에 이름이 나오면 핫워드 없이 다시 받아써서 이름이 여전히 있을 때만 믿는다.
11. **짧은 이름 호명**: 자모 5개 이하 변형(예: "민수")은 편집거리 2를 허용하면 "만세"까지 잡혀서 1로 제한한다(`namecall.short_max_edit_distance`). 긴 변형("민수야")은 명세대로 2.
12. **LLM 출력 필드 순서**: 스키마 필드 순서를 `type → pair → confidence`로 바꿨다(필드·값은 명세와 동일). qwen3:4b 점검 문장 18개에서 `pair`를 먼저 쓰게 하면 11/18, `type`을 먼저 쓰게 하면 13/18이었다. 작은 모델은 "B가 질문을 하면 대답"으로 보는 편향이 남아 있어 함정 일부를 놓친다. 그래서 융합에서 화자 증거와 함께 쓴다.
13. **LLM 맥락**: 착용자 말이 1.5초 이하 쉼으로 쪼개졌으면 하나로 합쳐 A로 넣는다. 이전 대화는 최근 30초, 같은 화자 연속 발화를 합친 2턴, 각 60자까지. 대화 상대로 최종 판정된 발화만 맥락에 넣는다.
14. **위험 소리 cpu_light**: YAMNet(tensorflow)은 설치가 무거워 기본 requirements에서 뺐다. cpu_light는 `backend: auto`(AST를 CPU로, 실패하면 YAMNet, 그것도 없으면 비활성), 분류 간격은 1초다. YAMNet을 쓰려면 `pip install tensorflow tensorflow-hub` 후 `download_models.py --yamnet`.
15. **단일 마이크 모드**: 시작할 때 첫 10초(또는 `audio.enroll_wav`)로 착용자를 등록하고, 이 10초는 처리하지 않는다. 본인 판정 기준은 등록 음성과의 ECAPA 유사도 ≥ 0.6이고, 본인 구간은 B 오디오로 받아쓴다.
16. **모델 경로**: `HEARME_MODELS_DIR` 환경변수가 있으면 그 경로를 쓰고, 없으면 `models/`를 쓴다. HF·Torch·TF Hub 캐시는 모두 이 아래에 둔다.
17. **Windows cuDNN**: 명세대로 pip `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`를 받아 `os.add_dll_directory`로 등록한다. 다만 torch cu124 휠이 cuDNN 9.1을 함께 넣어 오므로, 버전이 섞이지 않게 torch를 먼저 import한다(8절 첫 줄).
18. **합성 테스트 데이터**: `tools/make_test_scenario.py`는 시스템 TTS로 **테스트 녹음만** 만드는 개발 도구다. 제품은 음성을 출력하지 않는다. 피치를 바꾼 같은 목소리라 실제 사람보다 화자 구분이 어렵고 ASR은 쉽다. 발표 수치는 반드시 실제 녹음으로 다시 뽑을 것.
19. **LLM 상태 판정**: `ok` = 설치된 config 모델이 실제 판정 1회에 응답하고, `/api/ps`상 그 모델만 로드돼 있고, GPU에 올라가 있음(`size_vram`이 `size`의 95% 이상이면 GPU, 0보다 크면 GPU+CPU로 노랑 칩, 0이면 FAIL). 타임아웃 2회 연속이면 FAIL이고, 다음 점검에서 판정 1회가 성공하면 복구된다. FAIL인 동안은 LLM을 부르지 않는다(정책은 LLM None 규칙). `--no-llm`은 FAIL이 아니라 OFF(회색 칩, 배너 없음)로 표시한다.
20. **다른 모델 정리**: 명세는 "config 모델이 아닌 모델"을 내리라고 했지만, VRAM 확보를 위해 **선택된 모델 외에는 모두** 내린다. config 폴백 목록에 있는 모델도 포함한다.
21. **복구 시간**: 실패 중에는 2초마다 점검한다. 이 PC에서 `ollama serve` 자체가 API 응답까지 ~14초(Windows GPU 탐색), 모델 로딩이 ~3.5초 걸려, 프로세스 시작부터 정상 표시까지는 ~19초다. API가 뜬 뒤 감지·로딩·점검까지는 ~5초다.
22. **기존 LLM 캐시**: 디스크에 따로 저장되던 LLM 캐시 파일은 없었다(프로세스 메모리 캐시). 재생 결과(`results/*.segments.jsonl`)에 들어 있던 LLM 결과가 모델·프롬프트 버전 기록이 없는 캐시라서, 이것을 `cache/legacy/`로 옮기고 다시 생성했다.
23. **함정 구간**: 착용자 발화 직후(LLM 호출 창 −0.3~1.5초 안)에 시작했는데 라벨이 n인 구간이다. 표 2의 오표시율은 그중 큰 자막으로 표시된 비율이다.
24. **분할 미지정 이름**: `_take1`/`_take2` 표시가 없는 시나리오는 평가 기본 집합에서 빠진다. 이름을 직접 지정하면 경고와 함께 평가한다. 보정에는 `--allow-untagged`가 있어야 쓰이고, 쓰인 뒤에는 기록돼서 평가에서 거부된다.
