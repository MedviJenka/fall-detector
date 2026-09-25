import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware

from settings import Settings, get_settings
from src.ai.agents.alert_agent.fall_reviewer import CrewAIFallReviewer
from src.ai.config import AgentConfig
from src.models import EventStatus, FallEvent, FallReview, MonitorSnapshot
from src.services.alerts import WebhookNotifier
from src.services.camera import CameraEvent, CameraWorker
from src.services.events import EventStore
from src.services.monitor import (
    EventNotFoundError,
    InvalidTransitionError,
    MonitorService,
)

_STATIC_DIR = Path(__file__).with_name("static")
_ALLOWED_ORIGINS = {
    "http://127.0.0.1:8000",
    "http://localhost:8000",
}


class HealthResponse(BaseModel):
    status: str


class PublicFallEvent(BaseModel):
    event_id: UUID
    detected_at: str
    review: FallReview
    status: EventStatus
    deadline: str
    delivery_error: str | None = None


def _public_event(event: FallEvent) -> PublicFallEvent:
    return PublicFallEvent(
        event_id=event.event_id,
        detected_at=event.detected_at.isoformat(),
        review=event.review,
        status=event.status,
        deadline=event.deadline.isoformat(),
        delivery_error=event.delivery_error,
    )


def create_app(
    settings: Settings | None = None,
    monitor_factory: Callable[[Settings], MonitorService] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        active_settings = settings or get_settings()
        if monitor_factory is not None:
            monitor = monitor_factory(active_settings)
        else:
            event_queue: asyncio.Queue[CameraEvent] = asyncio.Queue(maxsize=1)
            camera = CameraWorker(active_settings.CAMERA_INDEX, event_queue)
            monitor = MonitorService(
                camera=camera,
                reviewer=CrewAIFallReviewer(
                    config=AgentConfig(
                        model=active_settings.OPENAI_MODEL,
                        api_key=active_settings.OPENAI_API_KEY,
                    )
                ),
                event_store=EventStore(active_settings.EVENT_DATA_DIR),
                notifier=WebhookNotifier(active_settings.ALERT_WEBHOOK_URL),
                event_queue=event_queue,
                countdown_seconds=active_settings.ALERT_COUNTDOWN_SECONDS,
            )
        app.state.monitor = monitor
        try:
            yield
        finally:
            await monitor.shutdown()

    app = FastAPI(
        title="Local Fall Detection Prototype",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @app.middleware("http")
    async def enforce_local_origin(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if (
            request.method in {"POST", "PUT", "PATCH", "DELETE"}
            and request.headers.get("origin") not in _ALLOWED_ORIGINS
        ):
            return JSONResponse(
                status_code=403,
                content={"detail": "Mutating requests require a local origin."},
            )
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/api/state", response_model=MonitorSnapshot)
    async def state(request: Request) -> MonitorSnapshot:
        return await request.app.state.monitor.snapshot()

    @app.get("/api/events", response_model=list[PublicFallEvent])
    async def events(request: Request) -> list[PublicFallEvent]:
        return [_public_event(event) for event in request.app.state.monitor.events()]

    @app.post("/api/monitor/start", response_model=MonitorSnapshot)
    async def start_monitor(request: Request) -> MonitorSnapshot:
        try:
            await request.app.state.monitor.start()
        except InvalidTransitionError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return await request.app.state.monitor.snapshot()

    @app.post("/api/monitor/stop", response_model=MonitorSnapshot)
    async def stop_monitor(request: Request) -> MonitorSnapshot:
        try:
            await request.app.state.monitor.stop()
        except InvalidTransitionError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return await request.app.state.monitor.snapshot()

    @app.post(
        "/api/alerts/{event_id}/cancel",
        response_model=PublicFallEvent,
    )
    async def cancel_alert(event_id: UUID, request: Request) -> PublicFallEvent:
        try:
            event = await request.app.state.monitor.cancel_alert(event_id)
        except EventNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except InvalidTransitionError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return _public_event(event)

    @app.get("/video.mjpg", include_in_schema=False)
    async def video(request: Request) -> StreamingResponse:
        async def stream() -> AsyncIterator[bytes]:
            while True:
                monitor = request.app.state.monitor
                snapshot = await monitor.snapshot()
                jpeg = monitor.latest_frame()
                if jpeg is None:
                    jpeg = _placeholder_jpeg(snapshot.state)
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    b"Cache-Control: no-store\r\n\r\n" + jpeg + b"\r\n"
                )
                if snapshot.state in {"stopped", "camera_error"}:
                    return
                await asyncio.sleep(0.1)

        return StreamingResponse(
            stream(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    return app


@lru_cache(maxsize=8)
def _placeholder_jpeg(state: str) -> bytes:

    image = np.full((480, 640, 3), (24, 30, 36), dtype=np.uint8)
    cv2.putText(image, "Camera feed unavailable", (126, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (224, 229, 233), 2, cv2.LINE_AA)
    cv2.putText(image, f"Status: {state.replace('_', ' ')}", (205, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 174), 1, cv2.LINE_AA,)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    if not ok:
        raise RuntimeError("Could not create camera placeholder")
    return encoded.tobytes()
