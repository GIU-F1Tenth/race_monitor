"""
Race Monitor Dashboard — FastAPI backend.

Bridges rclpy state to the browser via WebSocket at 10 Hz.
Exposes REST endpoints for race-control service calls.
"""

import asyncio
import json
import os
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ros_bridge import RaceBridge

# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Race Monitor Dashboard", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bridge = RaceBridge()
bridge.start()

# ── WebSocket manager ─────────────────────────────────────────────────────────

class _WsManager:
    def __init__(self):
        self._conns: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._conns.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._conns:
            self._conns.remove(ws)

    async def broadcast(self, payload: str) -> None:
        dead = []
        for ws in self._conns:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


_mgr = _WsManager()


# ── background broadcast loop (10 Hz) ─────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(_broadcast_loop())


async def _broadcast_loop() -> None:
    while True:
        state = bridge.get_state()
        await _mgr.broadcast(json.dumps(state))
        await asyncio.sleep(0.1)


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await _mgr.connect(ws)
    try:
        while True:
            # Keep the connection alive; client may send pings
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        _mgr.disconnect(ws)
    except Exception:
        _mgr.disconnect(ws)


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    return bridge.get_state()


# These are sync (not async) so FastAPI runs them in a thread pool,
# preventing the subprocess.run() call from blocking the event loop.

@app.post("/api/race/reset")
def reset_race():
    return bridge.reset_race()

@app.post("/api/race/force_complete")
def force_complete():
    return bridge.force_race_complete()

@app.post("/api/race/pause")
def pause_race():
    return bridge.pause_race()

@app.post("/api/race/resume")
def resume_race():
    return bridge.resume_race()

@app.post("/api/race/reset_lap")
def reset_lap():
    return bridge.reset_lap_time()

@app.get("/api/health")
async def health():
    return {"ok": True}


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("BACKEND_PORT", "8082"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
