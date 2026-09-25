from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np
from fastapi.testclient import TestClient

from settings import Settings
from src.api.app import create_app
from src.models import FallEvent, FallReview, MonitorSnapshot
from src.services.monitor import EventNotFoundError, InvalidTransitionError


class FakeMonitor:
    def __init__(self) -> None:
        self.current = MonitorSnapshot(state="stopped")
        self.event_id = uuid4()
        self.event = FallEvent(
            event_id=self.event_id,
            detected_at=datetime(2026, 1, 1, tzinfo=UTC),
            review=FallReview(
                classification="fall",
                confidence=0.91,
                person_visible=True,
                reason="A visible person fell and remained down.",
            ),
            snapshot_path=Path("secret/local/snapshot.jpg"),
            status="pending",
            deadline=datetime.now(UTC) + timedelta(seconds=15),
        )
        self.closed = False

    async def start(self) -> None:
        self.current = MonitorSnapshot(
            state="monitoring",
            camera_status="running",
        )

    async def stop(self) -> None:
        if self.current.state in {"countdown", "sending"}:
            raise InvalidTransitionError("alert active")
        self.current = MonitorSnapshot(state="stopped")

    async def shutdown(self) -> None:
        self.closed = True

    async def snapshot(self) -> MonitorSnapshot:
        return self.current

    def events(self) -> list[FallEvent]:
        return [self.event]

    async def cancel_alert(self, event_id: UUID) -> FallEvent:
        if event_id != self.event_id:
            raise EventNotFoundError("missing")
        if self.event.status != "pending":
            raise InvalidTransitionError("already final")
        self.event = self.event.model_copy(update={"status": "cancelled"})
        self.current = MonitorSnapshot(state="monitoring", camera_status="running")
        return self.event

    def latest_frame(self) -> bytes | None:
        return None


def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="vision-model",
        EVENT_DATA_DIR=tmp_path,
    )


def client_for(tmp_path: Path) -> tuple[TestClient, FakeMonitor]:
    monitor = FakeMonitor()
    app = create_app(
        settings(tmp_path),
        monitor_factory=lambda _: monitor,
    )
    return TestClient(app, base_url="http://localhost:8000"), monitor


def test_health_state_and_events_are_typed_and_secret_safe(tmp_path: Path) -> None:
    client, _ = client_for(tmp_path)

    with client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/api/state").json()["state"] == "stopped"
        events = client.get("/api/events").json()

    assert events[0]["status"] == "pending"
    assert "snapshot_path" not in events[0]
    assert "test-key" not in str(events)


def test_start_requires_exact_local_origin(tmp_path: Path) -> None:
    client, _ = client_for(tmp_path)

    with client:
        assert client.post("/api/monitor/start").status_code == 403
        assert (
            client.post(
                "/api/monitor/start",
                headers={"Origin": "http://localhost.evil:8000"},
            ).status_code
            == 403
        )
        response = client.post(
            "/api/monitor/start",
            headers={"Origin": "http://localhost:8000"},
        )

    assert response.status_code == 200
    assert response.json()["state"] == "monitoring"


def test_countdown_cancel_and_unknown_event_mapping(tmp_path: Path) -> None:
    client, monitor = client_for(tmp_path)
    monitor.current = MonitorSnapshot(
        state="countdown",
        active_event_id=monitor.event_id,
        alert_deadline=monitor.event.deadline,
        camera_status="running",
        webhook_status="pending",
    )
    headers = {"Origin": "http://127.0.0.1:8000"}

    with client:
        cancelled = client.post(
            f"/api/alerts/{monitor.event_id}/cancel",
            headers=headers,
        )
        missing = client.post(
            f"/api/alerts/{uuid4()}/cancel",
            headers=headers,
        )

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert missing.status_code == 404


def test_stop_conflict_maps_to_409(tmp_path: Path) -> None:
    client, monitor = client_for(tmp_path)
    monitor.current = MonitorSnapshot(state="sending", webhook_status="sending")

    with client:
        response = client.post(
            "/api/monitor/stop",
            headers={"Origin": "http://localhost:8000"},
        )

    assert response.status_code == 409


def test_video_endpoint_returns_finite_mjpeg_placeholder_when_stopped(
    tmp_path: Path,
) -> None:
    client, _ = client_for(tmp_path)

    with client:
        response = client.get("/video.mjpg")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "multipart/x-mixed-replace; boundary=frame"
    )
    assert b"Content-Type: image/jpeg" in response.content
    jpeg = response.content.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
    assert (
        cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR) is not None
    )


def test_dashboard_contains_required_warning_and_privacy_copy(tmp_path: Path) -> None:
    client, _ = client_for(tmp_path)

    with client:
        page = client.get("/").text

    assert (
        "Prototype only — do not rely on this as the sole way to detect a fall or "
        "contact emergency services."
    ) in page
    assert "Only locally screened candidate frames are sent" in page
