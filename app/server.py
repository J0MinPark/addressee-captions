"""FastAPI 서버: 정적 페이지 + WebSocket 이벤트 방송.

    python -m app.server --profile gpu_4060
    python -m app.server --profile cpu_light
    python -m app.server --replay data/demo --realtime     (마이크 없이 녹음 재생)

대시보드: http://<LAN IP>:8000/   폰: http://<LAN IP>:8000/phone
"""
# (FastAPI가 지역 import 타입 힌트를 해석해야 하므로 from __future__ import annotations 를 쓰지 않는다)

import argparse
import asyncio
import json
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from app import winsetup

winsetup.setup()

WEB = Path(__file__).resolve().parent / "web"


def lan_addresses() -> list[str]:
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("10.255.255.255", 1))   # 패킷을 보내지 않는다. 기본 경로의 IP만 얻음
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    ips = [ip for ip in ips if not ip.startswith("127.") and not ip.startswith("169.254.")]
    return ips or ["127.0.0.1"]


def print_qr(url: str) -> None:
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        try:
            qr.print_ascii(invert=True)
        except Exception:
            qr.print_tty()
    except Exception as e:
        print(f"(QR 출력 실패: {e})")


def build_app(pipeline):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse

    clients: set = set()
    state = {"loop": None}

    def on_event(ev: dict):
        loop = state["loop"]
        if loop is None:
            return
        data = json.dumps(ev, ensure_ascii=False, default=str)
        for q in list(clients):
            try:
                loop.call_soon_threadsafe(_put_nowait_drop, q, data)
            except RuntimeError:
                pass

    @asynccontextmanager
    async def lifespan(app):
        state["loop"] = asyncio.get_running_loop()
        pipeline.add_listener(on_event)
        pipeline.start()
        yield
        pipeline.stop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    async def dashboard():
        return FileResponse(WEB / "dashboard.html", headers={"Cache-Control": "no-store"})

    @app.get("/phone")
    async def phone():
        return FileResponse(WEB / "phone.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/snapshot")
    async def snapshot():
        return JSONResponse(json.loads(json.dumps(pipeline.snapshot(), default=str)))

    @app.post("/api/cmd")
    async def cmd(body: dict):
        pipeline.command(body)
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        clients.add(q)

        async def sender():
            await sock.send_text(json.dumps(pipeline.snapshot(), ensure_ascii=False, default=str))
            while True:
                data = await q.get()
                await sock.send_text(data)

        task = asyncio.create_task(sender())
        try:
            while True:
                msg = await sock.receive_text()
                try:
                    body = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                if body.get("cmd") == "ping":
                    continue
                pipeline.command(body)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            clients.discard(q)
            task.cancel()

    return app


def _put_nowait_drop(q: asyncio.Queue, data: str):
    if q.full():   # 느린 클라이언트: 오래된 것 버림
        try:
            q.get_nowait()
        except Exception:
            pass
    q.put_nowait(data)


def make_source(cfg, replay: str | None, realtime: bool, loop: bool = False):
    from app.audio_source import LiveSource, ReplaySource
    if replay:
        return ReplaySource(replay, cfg, realtime=realtime, loop=loop)
    return LiveSource(cfg)


def serve(cfg, source, models=None, mode=None, port=None, on_finished=None, run_name="",
          record_segments=False):
    import uvicorn
    from app.pipeline import Pipeline, load_models
    if models is None:
        models = load_models(cfg)
    pipe = Pipeline(cfg, source, models, mode=mode, record_segments=record_segments, run_name=run_name)
    app = build_app(pipe)
    port = port or cfg["server"]["port"]
    ips = lan_addresses()
    print("\n" + "=" * 60)
    for ip in ips:
        print(f"  대시보드:  http://{ip}:{port}/")
        print(f"  폰:        http://{ip}:{port}/phone")
    print("=" * 60)
    print_qr(f"http://{ips[0]}:{port}/phone")
    print("폰이 접속 안 되면: 노트북과 폰을 같은 휴대폰 핫스팟에 연결하세요(README 참고).")
    print(f"이벤트 로그: {pipe.log_path}")
    if models.judge is not None:
        from app.pipeline import log_llm_banner
        log_llm_banner(models.judge)
    from app.pipeline import config_name
    print(f"[구성] {config_name(cfg)}")
    print()

    if on_finished is not None:
        def watcher():
            pipe.wait_finished()
            on_finished(pipe)
        threading.Thread(target=watcher, daemon=True).start()
    uvicorn.run(app, host=cfg["server"]["host"], port=port, log_level="warning")
    pipe.close()
    return pipe


def main(argv=None):
    from app.config import load_config
    ap = argparse.ArgumentParser(description="나에게 온 말만 보여주는 자막 서버")
    ap.add_argument("--profile", default=None, help="gpu_4060 | cpu_light")
    ap.add_argument("--config", default=None)
    ap.add_argument("--replay", default=None, help="녹음 접두사(예: data/demo) — 마이크 대신 재생")
    ap.add_argument("--realtime", action="store_true", help="재생을 실시간 속도로")
    ap.add_argument("--loop", action="store_true", help="재생을 무한 반복(부스 시연·장시간 안정성 시험)")
    ap.add_argument("--mode", default=None, help="all|timing|timing_speaker|full|semantic")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--single-mic", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-llm-cache", action="store_true", help="LLM 판정 캐시 끄기(매번 새로 호출)")
    ap.add_argument("--no-selected", action="store_true",
                    help="개발 세트 선택 구성(app/selected_config.yaml) 대신 v1 구성(P1c·손 가중치)으로 실행")
    args = ap.parse_args(argv)
    over = {}
    if args.single_mic:
        over.setdefault("audio", {})["single_mic"] = True
    if args.no_llm:
        over.setdefault("llm", {})["enabled"] = False
    if args.no_llm_cache:
        over.setdefault("llm", {})["cache"] = False
    if args.no_selected:
        import os
        os.environ["HEARME_NO_SELECTED"] = "1"
    cfg = load_config(args.profile, args.config, over)
    print(f"[server] 프로필: {cfg['_profile']}")
    try:
        source = make_source(cfg, args.replay, args.realtime or not args.replay, args.loop)
    except Exception as e:
        print(f"[server] 오디오 소스 열기 실패: {e}")
        print("  → `python -m app.devices`로 장치 이름을 확인하고 config.yaml audio.device_* 를 고치세요.")
        print("  → 또는 녹음 재생: python -m app.server --replay data/demo --realtime")
        sys.exit(1)
    serve(cfg, source, mode=args.mode, port=args.port)


if __name__ == "__main__":
    main()
