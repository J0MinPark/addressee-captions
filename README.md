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

## 0. 연구실 서버 + 원격 시연 (2026-10, 현재 기본 구성)

이전 시연 PC(RTX 4060 노트북)는 더 이상 쓰지 않는다. **모든 처리는 연구실 리눅스 서버(RTX PRO 6000 Blackwell, 공용)**에서 하고,
시연장의 **Galaxy Book5(Windows, NVIDIA GPU 없음)는 마이크 2개 입력과 대시보드 브라우저만** 맡는다.

```
Book5: tools/client_capture.py (마이크 A·B, 20ms 블록) ──┐  학교 VPN + SSH 터널
       브라우저 http://localhost:8000/ ◀──────────────────┤  -L 8000:localhost:8000 -L 8765:localhost:8765
서버:  127.0.0.1:8765 NetworkSource(지터 버퍼 100ms) → 기존 파이프라인 → 127.0.0.1:8000 대시보드
       Ollama 127.0.0.1:11435 (우리 인스턴스, GPU 2) · Whisper·AST·ECAPA도 GPU 2
```

**공용 서버 규칙**: 우리 프로세스는 전부 GPU 2(`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=2`)에서만 돈다.
시스템 Ollama(11434, `ollama` 계정, GPU 1, 다른 사람의 모델)는 건드리지 않는다. 이름으로 프로세스를 죽이지 않고(pkill/killall 금지),
우리가 띄운 프로세스는 `logs/run/*.pid`로만 관리한다. 대시보드·음성 포트는 **localhost에만** 연다.

**서버 환경** (Ubuntu, Python 3.11 conda 환경 `.venv`):
- `requirements.txt`는 노트북용(torch 2.6.0+cu124)으로 그대로 둔다. **서버는 torch 2.7.1+cu128이 필요하다**(Blackwell sm_120 커널은 cu124 빌드에 없다:
  `no kernel image is available` → AST가 GPU에 못 올라간다). 실제 설치 목록: `requirements-server.lock.txt`(pip freeze).
  ```bash
  pip install -r requirements.txt && pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  pip install nvidia-ml-py      # 대시보드 GPU 사용률
  ```
- `scripts/server_env.sh`: GPU 2 고정, `OLLAMA_URL=http://127.0.0.1:11435`, `HEARME_DATA=~/jm/hearme_data`(저장소 밖 데이터 루트). 서버 스크립트가 source 한다.
  - `OLLAMA_URL`이 있으면 `config.yaml`의 `llm.url`(기본 11434, 노트북 호환)을 덮어쓴다.
  - `HEARME_DATA`가 있으면 시나리오 폴더(`paths.data_dir`)가 `$HEARME_DATA/data`가 된다. AI Hub 원본·파생 파일, DEMAND 소음도 여기에 둔다.
- `scripts/start_ollama.sh start|stop|status|restart|run [--port 11435] [--gpu 2]`: 우리 Ollama만 pid 파일로 관리한다.
  Vulkan을 끈다(`OLLAMA_VULKAN=0`: 켜 두면 `CUDA_VISIBLE_DEVICES`와 무관하게 GPU 0·1을 잡는 것을 확인했다). 컨텍스트 4096(판정 프롬프트는 1k 토큰 미만,
  기본값이면 4b 모델이 VRAM 41GB를 잡았다). Windows 노트북은 `scripts\start_ollama.bat` 그대로.
- `scripts/run_server.sh start|stop|status|run [인자]`: 앱 서버(기본 `--profile server --network`). `server` 프로필 = GPU ASR·AST + 원격 음성 + localhost 바인딩.
- **자동 재시작**: systemd 사용자 서비스(`scripts/install_services.sh`, linger 켬). `systemctl --user status|restart hearme-ollama hearme-server`.
  로그: `logs/ollama_11435.log`, `logs/server.log`. 수동 실행과 동시에 켜지 않는다(포트가 겹치면 다음 빈 포트로 바뀐다).
- **점검**: `source scripts/server_env.sh && python tools/env_check.py [--manifest]` → GPU 고정·CUDA·Whisper/AST/Ollama가 GPU 2에 있는지·모델 다이제스트·
  logprobs·DEMAND·단위 테스트 PASS/FAIL 표. `--manifest`는 `results/env_manifest_server.json`과 `requirements-server.lock.txt`를 다시 쓴다.
- **커밋 전 테스트**: `scripts/install_hooks.sh`(한 번) → pre-commit 훅이 전체 테스트 + 통합 테스트 5회 반복, 실패하면 커밋을 막는다.

**시연 구성**: 시연 프로필(`server`, `gpu_4060`, `cpu_light`)은 `app/demo_config.yaml`(AI Hub dev 규칙으로 고른 구성, `tools/apply_selection.py`가 씀)을 쓴다.
없으면 **v1 구성(P1c · 손 가중치, `--no-selected`와 같음)**이다. AMI dev 선택(`app/selected_config.yaml`, P1-qwen3:4b-learned)은 `ami` 프로필에만 적용된다(보고·재현용).

**재현성**: AMI 결과(6-1, 6-2절)는 이전 PC에서 만들었고, AMI 데이터는 이 서버에 없다(이전 PC와의 수치 비교는 하지 않았다). AI Hub 결과는 dev·test 모두 이 서버에서 만든다.

### 0-1. 원격 시연 접속 (Book5)

```powershell
# Book5 (Windows). 한 번만: 클라이언트 패키지만 설치(GPU 패키지 불필요)
python -m venv .venv-client ; .venv-client\Scripts\activate ; pip install -r requirements-client.txt
# 학교 VPN 연결 후 SSH 터널(창을 열어 둔다)
ssh -L 8000:localhost:8000 -L 8765:localhost:8765 <사용자>@<서버>
# 다른 창: 마이크 확인 → 음성 보내기
python tools/client_capture.py --list
python tools/client_capture.py --wearer "핀마이크 이름 일부" --ambient "주변 마이크 이름 일부"
# 브라우저: http://localhost:8000/
```
- 서버 포트가 8000/8765가 아니면(사용 중이라 바뀐 경우) 서버 시작 로그·`logs/run/ports.json`의 포트로 터널을 맞춘다.
- **폰 접속**: 터널을 `ssh -L 0.0.0.0:8000:localhost:8000 -L 8765:localhost:8765 <사용자>@<서버>`로 열면 같은 네트워크(핫스팟)의 폰이
  `http://<Book5 IP>:8000/phone`으로 접속한다. Book5 IP는 `ipconfig`의 핫스팟 어댑터 IPv4. Windows 방화벽에서 8000 인바운드 허용:
  관리자 PowerShell `New-NetFirewallRule -DisplayName hearme8000 -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private,Public`
  (또는 `netsh advfirewall firewall add rule name=hearme8000 dir=in action=allow protocol=TCP localport=8000`). 끝나면 규칙을 지운다.
- 마이크 문제 시 백업: `python tools/client_capture.py --replay data\demo --loop`(녹음 파일을 마이크 대신 보냄).
- 마이크가 하나면: 서버를 `--single-mic`으로(`systemctl --user stop hearme-server` 후 `scripts/run_server.sh start --single-mic`), 클라이언트도 `--single-mic`.
- 끊기면 클라이언트가 2초마다 다시 연결하고, 끊긴 동안의 음성은 버린다. 대시보드에는 "음성 클라이언트 연결 끊김"(서버가 음성을 못 받음)과
  "서버 연결 끊김"(브라우저↔서버 터널 끊김) 배너가 따로 뜬다. RTT 300ms 초과, 블록 누락 1% 초과(최근 30초), LLM 이상도 배너로 뜬다.
- 대시보드 지표: 네트워크 RTT·지터·누락 블록·수신량, 서버 GPU 2 사용률·메모리, LLM 상태, 판정·LLM·ASR 지연.
- 사전 점검(행사장): `python tools/net_check.py`(30초, RTT·지터·처리량 표, 자막에 영향 없음),
  `python tools/latency_bench.py --remote --replay data\demo_trap`(녹음을 실시간으로 보내 자막 지연 평균·p95를 `results/latency_*.csv`로. **시작 시 대시보드를 초기화**하므로 시연 중에는 돌리지 말 것).

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

## 6-1. 평가 데이터: AMI Meeting Corpus (사람 라벨링 없이)

공개 회의 코퍼스 [AMI](https://groups.inf.ed.ac.uk/ami/corpus/)의 **수동 대화행위 주석에 있는 addressee(수신자) 속성**을 정답으로 쓴다.
녹음·라벨링 없이 소거 표가 나온다.

```bash
set HEARME_DATA_DIR=C:\hearme_data\data              # 시나리오 WAV ~2GB — OneDrive 밖에 두기(선택)
python tools/import_ami.py --list                    # addressee 주석이 있는 회의 22개(그중 실제 주석 비율 50%+ 18개)
python tools/import_ami.py                           # 기본: 회의 3개 · 앞 15분 · clean/snr10/snr5 (다운로드 ~1.2GB + DEMAND 107MB)
python tools/run_ami.py --meeting ES2008b --wearer A --conds clean   # 빠른 확인(약 1분)
python tools/run_ami.py --yes                        # 전체(평가 24개, RTX 4060 약 20분) → results/ami_ablation.md/.csv
python tools/evaluate.py --ami --to-me-definition single+group       # 캐시만으로 재평가(모델 호출 없음)
```

- **시나리오 구성**: 회의마다 참가자 4명이 차례로 "착용자"가 된다. 채널 A는 그 사람의 개별 헤드셋, 채널 B는 원거리 마이크 Array1-01이고, 16kHz로 앞 15분만 쓴다.
  참가자 문자(A–D)와 헤드셋 채널의 대응은 `corpusResources/meetings.xml`의 `speaker@channel`에서 읽어 로그에 남긴다. 선택된 3개 회의는 모두 A→0, B→1, C→2, D→3이었다.
  두 채널 모두 정답 발화 구간의 RMS를 −26 dBFS로 맞춘다. 장비마다 다른 녹음 이득을 없애기 위해서다.
- **소음 조건**: B 채널에 DEMAND `PCAFETER`(카페테리아) 소음을 SNR 10 dB, 5 dB로 섞는다. SNR은 정답 발화 구간의 신호 전력을 기준으로 계산한다.
- **분할**: 주석 비율이 높은 회의부터 고르되, 같은 그룹(같은 참가자, 예: IS1008a/b)은 하나만 고른다. 그중 주석 비율이 가장 낮은 회의를 **보정용(take1)**, 나머지를 **평가용(take2)**으로 둔다.
  `run_ami.py`는 take1의 clean 녹음으로만 `own_margin_db`를 측정해 평가에 쓰고, 그 녹음을 `results/calibration_used.json`에 기록한다. 기록된 녹음은 평가에서 거부된다.
- **정답 라벨**(`tools/ami_labels.py`, `label.py`와 같은 CSV 형식):
  - 같은 화자의 연속 대화행위 중 addressee가 같고 사이 간격이 0.5초 이하인 것은 한 발화로 합친다.
  - 시스템이 만든 각 구간은 시간 겹침이 가장 큰 발화에 연결한다. 겹침이 구간 길이의 50% 미만이면 unmatched(`s`)로 두고 평가에서 뺀다.
  - 라벨 값은 아래 네 가지다.
    - `w`: 착용자 본인 발화
    - `y`: addressee가 착용자 한 명
    - `g`: addressee에 착용자를 포함해 2명 이상(그룹)
    - `n`: 다른 사람에게 한 말이거나 addressee 없음
- **'나에게 한 말' 정의**(`--to-me-definition`): `single`은 `y`만 양성이다. `single+group`은 `g`도 양성이다. 보고서에는 두 정의 표가 모두 나온다.
- **자연 함정**: 타이밍 증거가 성립하는데(T ≥ 0.5, 착용자 발화 직후 −0.3~1.5초 안에 시작) 라벨이 `n`인 구간이다. 대본 녹음의 trap 시나리오를 대신한다.
  보고서에는 모드별 **오표시율**과, 그런 구간을 만든 (진짜 대화 상대가 아닌) 화자의 **오등록률**이 나온다.
- **실행 시간**: 같은 회의·같은 조건의 B 채널 결과(VAD 확률, ASR, 화자 임베딩)는 착용자 4명이 공유한다(`results/cache/`). 두 번째 착용자부터는 15분 오디오가 약 12초에 처리된다.
- **LLM**: 영어 프롬프트와 영어 few-shot 6개를 쓴다(`llm.prompt_lang: en`, 프롬프트 버전이 바뀌어 캐시 키도 바뀐다). ASR은 영어이고, hotwords와 호명 감지는 끈다(`--profile ami`).

**v2 평가 규칙 (결과를 보기 전에 고정·커밋)**

- **분할**(`splits.json`, `tools/splits.py`가 강제): 시험 test = ES2008b, IS1008b, IS1000a, ES2009d / 개발 dev = IS1001b, IS1001c, IS1003b, IS1003d, IS1006b, IS1006d / 보정 calib = TS3005b.
  같은 참가자 그룹은 같은 집합에만 있고, 그 그룹의 나머지 회의는 쓰지 않는다.
  튜닝·학습·선택 스크립트는 test 회의를 읽으면 오류로 멈춘다. 최종 시험은 `results/final_test.lock`으로 한 번만 돈다.
- **주 지표: `single` 정의 기준 F0.5**(정밀도 가중)이고, dev 세트 3개 조건(clean·snr10·snr5)을 합산해 계산한다.
  보조 지표: 자연 함정 오표시율, 자연 함정 화자 오등록률, 재현율, 정밀도.
  - 근거: 놓친 말은 회색 한 줄로 남아 다시 볼 수 있지만, 오표시는 이 시스템이 없애려는 "자막 과부하"를 그대로 재현한다. 그래서 정밀도를 재현율보다 2배 중시한다(β=0.5).
- **통계**: (회의, 착용자) 단위 부트스트랩 1000회로 95% 신뢰구간을 낸다. timing과 최종 방식 비교는 같은 재표본을 쓰는 짝지은 부트스트랩으로 차이의 구간을 낸다.

**해석할 때 주의할 점**
1. **영어 회의다.** ASR, LLM 프롬프트, 대화 관습이 모두 한국어 시연 환경과 다르다. 수치는 "방법의 상대 비교"로만 보고, 한국어 시연 성능으로 옮겨 말하지 말 것.
2. **4인 회의라 대부분이 그룹 발화다.** 착용자 관점 정답 발화 중 `g`가 26%, 착용자 한 명에게 한 `y`는 5% 안팎이다.
   `single` 정의에서는 양성이 극히 적어 정밀도가 낮게 나오기 쉽고 분산이 크다. `single+group` 정의에서는 '전부 표시'가 이미 높은 정밀도를 갖는다.
   1:1 대화를 가정한 이 시스템의 목표 상황(카페에서 한 사람과 대화 + 주변 잡음 대화)과는 다르다.
3. **"주변의 다른 대화"가 없다.** 회의 참가자는 모두 같은 대화 안에 있다. 시스템이 걸러야 할 '옆 테이블 대화'는 소음 조건(카페 소음, 내용 없는 웅성거림)으로만 흉내 냈다.
4. **원거리 마이크 구간은 여러 화자 발화를 묶는다.** 짧은 `n`(맞장구 등)이 긴 그룹 발화 구간에 흡수되어, 구간 단위 `n`이 발화 단위 `n`보다 훨씬 적다.
5. **본인 발화 마진은 보정값이 하한(2.0 dB)에 걸렸다.** 헤드셋 대 원거리 마이크의 dB 차이 분포가 두 봉우리로 깔끔하게 갈리지 않았다는 뜻이다. 착용자 구간 판정 오류가 일부 섞인다(`labelstats.json`의 `wearer_seg:g` 등).
7. **등록 지연이 음수**로 나올 수 있다: 그 화자의 첫 `y` 라벨 발화보다 먼저 partner로 등록된 경우다(특히 `all` 모드, 그룹 발화로 등록). AMI에서는 '등록 지연'보다 '미등록 n/m'과 자연 함정 지표를 볼 것.
8. 회의 2개, 착용자 8명, 15분 분량이다. 표본이 작으니 모드 간 차이가 몇 %p 이내면 같은 수준으로 볼 것.

## 6-2. v2: 함정 거르기를 유지하며 재현율 올리기 (AMI dev → test)

**절차** (분할·주 지표·8b 확장 기준은 모두 결과를 보기 전에 커밋):
```bash
python tools/import_ami.py --ids IS1001b IS1001c IS1003b IS1003d IS1006b IS1006d IS1000a ES2009d --calib TS3005b
python tools/run_ami.py --split dev --no-report --yes        # dev 재생(특징 추출: v1 판정기·손 가중치 고정)
python tools/judge_offline.py --models qwen3:4b --variants P1 P2 P3              # 판정기 단독 비교
python tools/judge_offline.py --models qwen3:8b --variants P1 P2 P3 --conds clean --subdir dev_clean8b
python tools/compare_8b.py                                    # 8b 확장 여부(사전 기준)
python tools/tune_dev.py                                      # 융합·플래그 비교 + 선택 → results/selection.md
python tools/apply_selection.py                               # 모든 프로필에 같은 구성 적용 + 확인
python tools/final_test.py                                    # 시험 세트 한 번 → results/final_test.md
```
- **판정기 v2**(`llm.variant`): P1 = 인접쌍 정의, P2 = 반응 정의 확장(대답·동의/반대·이어받기·맞장구·되묻기), P3 = P2 + 화자 표시 최근 4턴. 영어·한국어판이 1:1로 대응한다.
  출력은 `{"pair": bool}` 하나이고, Ollama `logprobs`의 true/false 토큰 확률로 **연속 점수 p_pair**를 만든다(이 Ollama 0.35에서 동작 확인).
  짝 아님 few-shot에는 "질문 직후 제3자에게 하는 다른 질문"을 넣었다.
- **학습된 융합**(`policy.fusion.type: logistic`): 특징은 T, 간격 g, 화자 유사도, 등록 상태, p_pair, LLM 결과 유무, 구간 길이, 착용자 직전 말이 의문문인지(`policy.fusion_features`).
  L2 로지스틱이고, 정규화 강도는 (회의, 착용자) 단위 6겹 교차검증 log-loss로, 임계값은 dev F0.5 최대로 정한다.
  **선택 비교에는 겹 밖(교차검증) 추정을 쓴다**(in-sample 값은 낙관적이라 따로 표시).
- **플래그**(기본 꺼짐):
  - `policy.candidate_rejudge`: T=1인데 LLM이 '짝 아님'이면 바로 접지 않고 보류한다. 같은 화자가 다음 교대에서 다시 응답하면 확정하고, 다른 사람이 응답하거나 30초가 지나면 접는다.
  - `policy.short_skip_llm`: 2음절 이하는 LLM을 생략한다.
- **통계**: (회의, 착용자) 부트스트랩 1000회, 핵심 비교는 짝지은 부트스트랩(`tools/stats_boot.py`).

**개발 세트 결과**(dev 6회의 × 착용자 4 × 3조건 = 72개, single 정의, 3조건 합산; 전체 표는 `results/dev_results.md`):

| 구성 | F0.5 [95% CI] | 정밀도 | 재현율 | 자연 함정 오표시율 |
|---|---:|---:|---:|---:|
| timing | 0.185 [0.146, 0.229] | 16.0% | 49.6% | 86.7% |
| v1 (P1c · 손 가중치) | 0.167 [0.114, 0.224] | 15.6% | 23.6% | 31.8% |
| P1 · 손 가중치 | 0.161 | 14.4% | 30.9% | 42.0% |
| **P1 · 학습된 융합 (선택)** | **0.221 [0.180, 0.271]** | 20.5% | 32.9% | 40.7% |
| P2 · 학습된 융합 | 0.220 | 20.5% | 31.7% | 39.3% |
| P3 · 학습된 융합 | 0.203 | 19.6% | 23.1% | 30.1% |

- **판정기 단독**(착용자 직후 구간 2,516개, `results/judge_dev.md`):
  - AUC는 v1(confidence 매핑) 0.540, 4b 연속 점수 P1 0.586, P2 0.584, P3 0.585다. 연속 점수가 순위 정보를 조금 더 주지만, 변형 간 차이는 없다.
  - 지연 중앙값은 4b 170ms 안팎, VRAM 3.18GB이고, Whisper와 같이 올리면 5.2/8.2GB다.
- **qwen3:8b**(dev clean만, `results/compare_8b.md`):
  - AUC는 최고 0.631(P2)이고, 4b 최고 0.609 대비 차이는 +0.022 [−0.025, +0.075]다. 사전 기준(≥+0.02이고 CI 하한 >0)에 **미달해 후보에서 제외**했다.
  - VRAM 5.58GB이고, Whisper와 같이 올리면 7.49/8.19GB라 AST까지 올리면 빠듯하다.
- **플래그**: dev에서 F0.5를 올리지 못했다. 선택 구성 기준 rejudge 0.216, shortskip 0.182(없음 0.221). 기본값 꺼짐을 유지한다.
- **선택**: `P1-qwen3:4b-learned`. timing 대비 짝지은 차이는 F0.5 +0.037 [+0.006, +0.064]다.
  자연 함정 오표시율을 86.7%에서 40.7%로 낮추면서 재현율은 v1의 23.6%에서 32.9%로 올렸다.
  v1보다 함정은 덜 거르지만(31.8%→40.7%) 정밀도·재현율이 모두 올랐다.
- **학습된 계수 읽기**: p_pair(+0.81)·의문문 직후(+0.74)·화자 유사도(+0.32)는 양(+)이다. 간격 g(−2.84)·구간 길이(−1.56)는 음(−)이다: 짧고 바로 붙은 응답일수록 나에게 한 말이다.
  T가 음수(−1.71)인 것은 g와 강하게 겹치는(공선성) 탓으로 보고, 둘을 함께 해석할 것.
  임계값이 0.2로 낮은 것은 양성(착용자 한 명에게 한 말)이 7%뿐이라 확률이 전반적으로 낮게 나오기 때문이다.
- **시연 프로필에도 같은 구성** *(2026-10 변경: 지금은 `ami` 프로필에만 적용, 시연은 0절·6-3절 참고)*: `app/selected_config.yaml`(자동 생성)이 모든 프로필에 같은 판정기 변형·융합 가중치·플래그를 적용한다(`python tools/apply_selection.py --check`).
  시작 로그에 `[구성] P1-qwen3:4b-learned`가 찍힌다. 한국어 시연은 같은 P1의 한국어판 프롬프트를 쓴다.
  ⚠ 융합 가중치는 **영어 4인 회의에서 학습**됐다. 1:1 한국어 대화에서는 양성 비율이 훨씬 높아 임계값 0.2가 관대하게 작동할 수 있다.
  시연 전 `tools/judge_text_eval.py`와 대본 녹음으로 확인하고, 필요하면 `--mode timing_speaker`·수동 등록으로 대응한다.

**시험 세트 결과** (한 번만 실행, `results/final_test.md`; test 4회의 × 착용자 4 × 3조건 = 48개, single 정의, 3조건 합산):

| 방식 | F0.5 [95% CI] | 정밀도 | 재현율 | 자연 함정 오표시율 | 함정 화자 오등록률 |
|---|---:|---:|---:|---:|---:|
| baseline-v1 (P1c · 손 가중치) | 0.265 [0.161, 0.383] | 48.0% | 9.5% | 15.5% | 23.5% |
| timing | 0.348 [0.269, 0.407] | 35.9% | 31.2% | 87.4% | 89.5% |
| **최종 P1-qwen3:4b-learned** | **0.354 [0.270, 0.452]** | 53.2% | 15.1% | 20.2% | 26.8% |

짝지은 부트스트랩:
- **최종 − baseline-v1**: F0.5 +0.089 [+0.039, +0.151], 재현율 +5.6%p [+3.1, +8.1].
  자연 함정 오표시율은 +4.7%p [−0.4, +9.8]로 약간 늘었지만, 구간이 0을 포함한다.
- **최종 − timing**: F0.5 +0.005 [−0.088, +0.116]로 차이 없음. 정밀도 +17.3%p [+6.5, +31.5], 자연 함정 오표시율 −67.2%p [−76.4, −57.8], 재현율 −16.1%p [−23.6, −8.3].

해석:
- v1 대비 재현율을 올리면서 함정 거르기는 거의 유지했다(목표 달성).
- timing과 F0.5는 같지만 성격이 다르다. 최종 구성은 정밀도와 함정 거르기를 얻고 재현율을 내준다.
- test의 양성 비율·분포가 dev와 달라 절대값이 dev보다 높다. dev→test로 순위(최종 ≥ timing > v1)는 유지됐다.

**⚠ 한국어 시연에 그대로 쓰면 안 되는 문제 (시험 후 발견)**
- **한국어판 P1 판정기가 "짝"을 거의 내지 않는다.** 합성 한국어 대화(`demo_trap`)에서 "혹시 여기 자리 있어요 → 아니요 비어 있어요"의 p_pair가 0.08이었다.
  예시 5쌍(`data/judge_pairs_ko.example.csv`) 정확도는 P1 60%, P1c 100%, P2 100%였다.
- 이 점수에 영어 회의에서 학습된 융합 가중치가 겹쳐, 한국어 합성 시연에서는 **전체 융합 모드가 아무것도 표시하지 않았다**(대화 상대 등록 0).
- 선택은 dev 원칙대로 유지했다(5쌍·합성 데이터로 바꾸면 그 자체가 새 선택이 된다). 대신 시연용 즉시 대체 경로를 만들었다:
  **`python -m app.server --profile gpu_4060 --no-selected`** → v1 구성(P1c · 손 가중치). 합성 한국어 시연에서 정상 동작한다(F1 0.89, 대화 상대 등록 정상).
- 권장: 사람이 `data/judge_pairs_ko.csv`를 30쌍 이상 채워 `tools/judge_text_eval.py`로 P1/P1c/P2를 비교한 뒤 시연 구성을 정할 것.
- **2026-10 수정(버그)**: 원인은 한국어 P1의 few-shot이었다(영어 P1과 1:1 대응이 아닌 v1 한국어 예시를 재사용, 짝 점수가 전반적으로 눌림).
  번역·logprob 추출은 원인이 아니었다. 한국어 P1을 few-shot 없이 쓰도록 고쳤다(점검 48쌍 정확도 0.79→1.00, 위 쌍 0.18→0.95, 프롬프트 해시 243b6e2a→437056c3).
  영어 프롬프트는 그대로다. 자세한 내용: `results/ko_prompt_fix.md`. 시연 구성은 AI Hub dev 규칙으로 다시 고른다(6-3절).


## 7. 시연 런북 (90초)

### 7-0. 원격 시연 런북 (발표 당일 · 서버 처리 + Book5) — 현재 기본

**서버(발표 1시간 전, SSH로 접속해서)**
1. [ ] `systemctl --user status hearme-ollama hearme-server` → 둘 다 `active (running)`. 아니면 `systemctl --user restart hearme-ollama hearme-server`
2. [ ] `cd ~/jm/addressee-captions && source scripts/server_env.sh && python tools/env_check.py` → **모두 PASS**
   (GPU 2 사용률이 다른 사람 작업으로 높으면 표의 GPU 행과 대시보드 GPU 지표를 보고, 지연이 크면 아래 4번 결과로 판단)
3. [ ] `grep "\[구성\]\|llm=" logs/server.log | tail -2` → 구성 이름(`P1c-qwen3:4b-hand` = v1, 또는 `demo_config.yaml`의 구성)과 `llm=qwen3:4b ok … (GPU)` 확인

**Book5(발표 30분 전, 시연 자리에서)**
4. [ ] 학교 VPN 연결 → 터널 창: `ssh -L 8000:localhost:8000 -L 8765:localhost:8765 <사용자>@<서버>` (폰도 쓰면 `-L 0.0.0.0:8000:...`, 0-1절 방화벽)
5. [ ] `python tools/net_check.py` → **모두 통과**(RTT p95 ≤ 300ms, 처리량 ≥ 필요량 95%). FAIL이면 핫스팟/유선으로 바꾸고 다시
6. [ ] `python tools/latency_bench.py --remote --replay data\demo_trap` → 자막 지연 p95를 적어 둔다(2초 이하여야 함, 시연 구성 선택의 제약 2).
   **client_capture를 켜기 전에** 돌린다(새 음성 연결이 기존 연결을 끊는다). 결과 CSV(`results/latency_demo_trap_remote_*.csv`)를 서버 담당에게 전달
7. [ ] `python tools/client_capture.py --list` → `python tools/client_capture.py --wearer "<핀마이크>" --ambient "<주변 마이크>"` (창을 열어 둔다)
8. [ ] 브라우저 `http://localhost:8000/` → 상단 칩 초록(`LLM: qwen3:4b · GPU`), 배너 없음, 오른쪽 지표 "음성 클라이언트 연결됨", RTT 수십 ms
9. [ ] 착용자·상대가 한 마디씩 → 자막이 뜨는지, 착용자 말이 오른쪽 작은 글씨인지(아니면 마이크 위치·`own_margin_db`, 5절)
10. [ ] 대시보드 **초기화** → 모드 **전체 융합** → 프로젝터(필요하면 폰 QR 대신 `http://<Book5 IP>:8000/phone`)
11. [ ] 백업 준비: 다른 창에 `python tools/client_capture.py --replay data\demo --loop` (마이크가 안 되면 7번 창을 닫고 이것을 실행)

**라이브가 이상하면**: "음성 클라이언트 연결 끊김" → 7번 창 확인(자동 재연결). "서버 연결 끊김" → 4번 터널 창 확인 후 새로고침.
지연 배너(RTT 300ms 초과) → 핫스팟 전환. LLM 배너 → 서버에서 `systemctl --user restart hearme-ollama`(10초 안에 자동 복구). 화자가 꼬이면 화자를 클릭해 수동 등록/해제.
시연 시나리오(아래 표)는 같다. 시연 구성이 timing으로 정해지면(6-3절 규칙) 함정 장면(0:35–1:00)은 뺀다.

### 7-1. (이전) 노트북 단독 시연 체크리스트

**발표 전 체크리스트** (순서대로, 발표 15분 전)

1. [ ] GPU 쓰는 프로그램(게임·영상 편집 등) 종료. 핫스팟 켜고 노트북·폰을 같은 핫스팟에 연결.
2. [ ] `scripts\start_ollama.bat` (트레이 Ollama·고아 러너 종료 후 모델 1개 모드로 실행. 켜지는 데 ~15초)
3. [ ] **시연 구성 결정**: 한국어 판정기 문제(6-2절 ⚠)로 현재는 `--no-selected`(v1 구성)를 권장. 5번 서버 명령에 붙인다
4. [ ] `python -m app.preflight --profile gpu_4060` → 안내에 따라 착용자/상대가 5초씩 말하기 → **모든 항목 PASS** 확인
   (FAIL이면 상세 칸의 조치대로. 마이크 녹음이 FAIL이면 5절 1번 보정)
5. [ ] 서버를 **새로** 시작(메모리·화자 상태 초기화): 떠 있던 서버는 Ctrl+C 후 `python -m app.server --profile gpu_4060 --no-selected`
   → 터미널 배너가 `llm=qwen3:4b ok …ms (GPU)` 인지 확인(`!!` 배너면 원인 해결 후 재시작)
6. [ ] 대시보드 상단 **상태 칩이 초록**(`LLM: qwen3:4b · GPU · 0.xs`)이고 빨간 "LLM 끊김" 배너가 없는지 확인
7. [ ] 폰으로 QR 접속 → 화면 한 번 터치(진동·화면 켜짐 허용) → 우상단 "연결됨" 확인
8. [ ] 대시보드 **초기화** → 모드 **전체 융합** → 프로젝터 연결
9. [ ] 백업 터미널에 `python tools/replay.py data/demo --realtime --port 8001` 준비(포트 다름, `data/demo`는 실제 시연 녹음으로 교체해 둘 것)
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
tools/               record, replay, label, evaluate, diff_modes, calibrate, datasplit(분할 규칙), make_test_scenario(개발용),
                     import_ami · ami_labels · run_ami · ami_report (AMI 평가)
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
25. **원격 음성 블록**: 클라이언트는 20ms(320샘플) 2채널 int16 블록에 seq와 클라이언트 단조 시간을 붙여 WebSocket으로 보낸다(약 520kbps).
    서버 지터 버퍼는 다음 seq가 없을 때 그 뒤 블록이 도착한 지 100ms까지 기다렸다가 무음으로 채우고 누락으로 센다(재정렬은 이 100ms 안에서만).
    WebSocket은 TCP라 실제 재정렬·손실은 드물고, 누락은 주로 클라이언트 쪽 장치 오류나 재연결에서 생긴다.
26. **재연결**: 새 음성 연결이 기존 연결을 대체한다(마지막 연결이 이긴다). 재연결하면 새 세션이고 끊긴 동안의 음성은 버린다. 서버 스트림 시간은 끊김 없이 이어진다
    (정책 시간에는 끊긴 시간이 없는 셈이라 끊김 직전·직후 발화가 붙어 보일 수 있다. 짧은 끊김만 가정).
27. **원격 지연 정의**: `latency_bench.py`의 지연 = 대시보드 이벤트를 받은 클라이언트 시각 − 그 구간 마지막 음성 블록을 보낸 클라이언트 시각(같은 시계).
    "처음 표시"는 caption 이벤트, "최종"은 LLM 판정이 반영된 caption_update. 마이크 버퍼링(~20ms)은 포함하지 않는다. 시연 구성 제약 2(p95 ≤ 2초)에는 "최종"을 쓴다.
28. **누락률 배너**: 누적이 아니라 최근 30초 기준으로 1%를 넘으면 띄운다(초반 한 번의 끊김이 계속 배너로 남지 않게). 누적값은 지표 표에 같이 보인다.
29. **최대 속도 재생의 순서 보장**: 재생(평가)에서는 ASR에 넣은 구간의 결과를 제어 스레드가 처리할 때까지 다음 오디오를 넣지 않는다(최대 30초).
    이전에는 제어 스레드가 밀리면 뒤 구간이 앞 구간의 LLM 결과(등록)보다 먼저 판정되는 경쟁이 있었다(부하에서 통합 테스트 약 0.5% 실패, 이 서버에서 재현·수정).
    AMI 결과는 수정 전 코드로 이전 PC에서 만든 것이고 다시 돌리지 않았다. 실시간 경로는 바뀌지 않는다.
30. **리눅스 합성 음성**: `make_test_scenario.py`는 리눅스에서 edge-tts(온라인, 한국어 뉴럴 음성 InJoon·SunHi·Hyunsu, D는 SunHi 음높이 +25Hz)를 쓴다.
    Windows SAPI 한 목소리를 음높이로 바꾼 이전 합성보다 화자 구분이 쉽다. 같은 대본이다. `[SYNTHETIC]` 규칙은 같다.
32. **한국어 P1 수정 기준**: "한국어판은 영어판의 번역"이라는 원칙으로, 영어 P1이 역번역 문장에 내는 점수와 가장 가까운(MAD 최소) 후보를 골랐다. 라벨 정확도로 고르지 않았다(점검 세트에 맞춘 튜닝을 피하기 위해). 점검 쌍은 AI Hub가 아닌 직접 작성 문장이다.
31. **calibration_used.json**: gitignore 대상이라 이전 PC에만 있었다. 커밋된 `results/ami_calibration.json`의 `from` 목록(TS3005b 착용자 4명 take1)으로 이 서버에서 다시 만들었다.

### 편차 기록 (사전 계획·이전 PC 대비)

| # | 내용 | 영향 |
|---|---|---|
| D1 | AMI 결과(dev·test)는 이전 PC(RTX 4060, Windows, torch 2.6.0+cu124)에서 만들었고, AMI 데이터가 이 서버에 없어 서버에서 재현·비교하지 않았다 | AMI 수치는 그대로 보고. 서버와의 수치 동일성은 확인 안 됨 |
| D2 | 서버 torch는 2.7.1+cu128(Blackwell 필수). `requirements.txt`(노트북)와 다르다 | AI Hub 결과는 전부 서버 환경(`results/env_manifest_server.json`) |
| D3 | 이전 PC에서 하려던 정리(잠금 파일 커밋 확인, 경로 정리, 환경·데이터 매니페스트)는 이 서버에서 했다. 잠금 파일 `results/final_test.lock`은 이미 커밋돼 있었다(42e2d23) | 없음 |
| D4 | 최대 속도 재생의 순서 경쟁 수정(가정 29). AMI 결과는 수정 전 코드 | AI Hub 재생만 수정 후 코드 |
| D5 | 시연 프로필에서 AMI 선택 구성을 뗐다(시연 기본 = v1). AMI 선택은 `ami` 프로필에만 | 6-2절의 "시연 프로필에도 같은 구성" 문장은 더 이상 유효하지 않음 |
| D6 | 사전 등록 개정 1: AI Hub 원천 음성 대신 라벨 기반 대화 시뮬레이션(정답 분할·전사·화자 ID)으로 P2/P3를 대체(일정) | 판정 계층만 평가. ASR·화자 임베딩·분할·소음 영향 미포함 |
