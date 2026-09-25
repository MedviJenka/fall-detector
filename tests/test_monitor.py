import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from src.models import DeliveryResult, FallReview
from src.services.alerts import WebhookNotifier
from src.services.camera import CandidateBundle
from src.services.events import EventStore, PersistenceError
from src.services.monitor import InvalidTransitionError, MonitorService


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value


class FakeCamera:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def latest_jpeg(self) -> bytes | None:
        return b"latest"


class FakeReviewer:
    def __init__(self, review: FallReview, gate: asyncio.Event | None = None) -> None:
        self.review_result = review
        self.gate = gate
        self.started = asyncio.Event()
        self.calls = 0

    async def review(self, frames: tuple[bytes, ...]) -> FallReview:
        self.calls += 1
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        return self.review_result


class FakeNotifier:
    def __init__(
        self,
        result: DeliveryResult | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.result = result or DeliveryResult(status="sent")
        self.gate = gate
        self.started = asyncio.Event()
        self.calls: list[tuple[Any, bytes]] = []

    async def send(self, event: Any, snapshot: bytes) -> DeliveryResult:
        self.calls.append((event, snapshot))
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        return self.result


async def sleep_forever(_: float) -> None:
    await asyncio.Event().wait()


def confirmed_review(confidence: float = 0.9) -> FallReview:
    return FallReview(
        classification="fall",
        confidence=confidence,
        person_visible=True,
        reason="A visible person fell and remained down.",
    )


def candidate() -> CandidateBundle:
    return CandidateBundle(
        candidate_timestamp=10.0,
        frames=(b"one", b"two", b"three", b"snapshot"),
    )


def build_monitor(
    tmp_path: Path,
    *,
    review: FallReview | None = None,
    reviewer: FakeReviewer | None = None,
    notifier: FakeNotifier | None = None,
) -> tuple[MonitorService, MutableClock, FakeNotifier]:
    clock = MutableClock()
    selected_notifier = notifier or FakeNotifier()
    service = MonitorService(
        camera=FakeCamera(),
        reviewer=reviewer or FakeReviewer(review or confirmed_review()),
        event_store=EventStore(tmp_path),
        notifier=selected_notifier,
        event_queue=asyncio.Queue(maxsize=1),
        countdown_seconds=15,
        clock=clock,
        sleep=sleep_forever,
    )
    return service, clock, selected_notifier


@pytest.mark.asyncio
async def test_confirmed_fall_creates_one_pending_event(tmp_path: Path) -> None:
    service, clock, _ = build_monitor(tmp_path)
    await service.start()

    assert await service.handle_camera_event(candidate()) is True

    events = service.events()
    assert len(events) == 1
    assert events[0].status == "pending"
    assert events[0].deadline == clock.value + timedelta(seconds=15)
    assert events[0].snapshot_path.read_bytes() == b"snapshot"
    assert (await service.snapshot()).state == "countdown"
    await service.shutdown()


@pytest.mark.asyncio
async def test_low_confidence_fall_does_not_create_event(tmp_path: Path) -> None:
    service, _, _ = build_monitor(tmp_path, review=confirmed_review(0.69))
    await service.start()

    assert await service.handle_camera_event(candidate()) is True

    assert service.events() == []
    assert (await service.snapshot()).state == "monitoring"
    await service.shutdown()


@pytest.mark.asyncio
async def test_cancellation_prevents_delivery(tmp_path: Path) -> None:
    service, clock, notifier = build_monitor(tmp_path)
    await service.start()
    await service.handle_camera_event(candidate())
    event = service.events()[0]

    cancelled = await service.cancel_alert(event.event_id)
    clock.value = event.deadline
    await service.deliver_due_event(event.event_id)

    assert cancelled.status == "cancelled"
    assert notifier.calls == []
    await service.shutdown()


@pytest.mark.asyncio
async def test_deadline_sends_exactly_once(tmp_path: Path) -> None:
    service, clock, notifier = build_monitor(tmp_path)
    await service.start()
    await service.handle_camera_event(candidate())
    event = service.events()[0]
    clock.value = event.deadline

    await service.deliver_due_event(event.event_id)
    await service.deliver_due_event(event.event_id)

    assert len(notifier.calls) == 1
    assert service.events()[0].status == "sent"
    await service.shutdown()


@pytest.mark.asyncio
async def test_simultaneous_cancel_and_deadline_have_one_atomic_winner(
    tmp_path: Path,
) -> None:
    service, clock, notifier = build_monitor(tmp_path)
    await service.start()
    await service.handle_camera_event(candidate())
    event = service.events()[0]
    clock.value = event.deadline

    results = await asyncio.gather(
        service.cancel_alert(event.event_id),
        service.deliver_due_event(event.event_id),
        return_exceptions=True,
    )

    assert sum(isinstance(result, InvalidTransitionError) for result in results) == 1
    assert len(notifier.calls) == 1
    assert service.events()[0].status == "sent"
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ["No caregiver webhook is configured.", "Webhook failed."]
)
async def test_missing_or_failed_webhook_becomes_delivery_failed(
    tmp_path: Path, reason: str
) -> None:
    notifier = FakeNotifier(DeliveryResult(status="delivery_failed", error=reason))
    service, clock, _ = build_monitor(tmp_path, notifier=notifier)
    await service.start()
    await service.handle_camera_event(candidate())
    event = service.events()[0]
    clock.value = event.deadline

    await service.deliver_due_event(event.event_id)

    failed = service.events()[0]
    assert failed.status == "delivery_failed"
    assert failed.delivery_error == reason
    assert (await service.snapshot()).webhook_status == "failed"
    await service.shutdown()


@pytest.mark.asyncio
async def test_concurrent_candidate_is_dropped_while_reviewing(tmp_path: Path) -> None:
    gate = asyncio.Event()
    reviewer = FakeReviewer(confirmed_review(), gate=gate)
    service, _, _ = build_monitor(tmp_path, reviewer=reviewer)
    await service.start()

    first = asyncio.create_task(service.handle_camera_event(candidate()))
    await reviewer.started.wait()
    second_result = await service.handle_camera_event(candidate())
    gate.set()
    await first

    assert second_result is False
    assert reviewer.calls == 1
    assert len(service.events()) == 1
    await service.shutdown()


@pytest.mark.asyncio
async def test_start_drops_camera_events_left_from_previous_run(tmp_path: Path) -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    queue.put_nowait(candidate())
    reviewer = FakeReviewer(confirmed_review())
    service = MonitorService(
        camera=FakeCamera(),
        reviewer=reviewer,
        event_store=EventStore(tmp_path),
        notifier=FakeNotifier(),
        event_queue=queue,
        countdown_seconds=15,
        clock=MutableClock(),
        sleep=sleep_forever,
    )

    await service.start()
    await asyncio.sleep(0)

    assert reviewer.calls == 0
    await service.shutdown()


@pytest.mark.asyncio
async def test_stopping_during_countdown_or_send_conflicts(tmp_path: Path) -> None:
    send_gate = asyncio.Event()
    notifier = FakeNotifier(gate=send_gate)
    service, clock, _ = build_monitor(tmp_path, notifier=notifier)
    await service.start()
    await service.handle_camera_event(candidate())
    event = service.events()[0]

    with pytest.raises(InvalidTransitionError):
        await service.stop()

    clock.value = event.deadline
    delivery = asyncio.create_task(service.deliver_due_event(event.event_id))
    await notifier.started.wait()
    with pytest.raises(InvalidTransitionError):
        await service.stop()
    send_gate.set()
    await delivery
    await service.shutdown()


def test_malformed_final_jsonl_record_is_reported_and_skipped(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event = store.create_event(
        confirmed_review(),
        b"snapshot",
        detected_at=datetime(2026, 1, 1, tzinfo=UTC),
        countdown_seconds=15,
    )
    with (tmp_path / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"event_id":')

    reloaded = EventStore(tmp_path)

    assert reloaded.persistence_warning is not None
    assert reloaded.get(event.event_id).status == "pending"


def test_malformed_earlier_jsonl_record_is_startup_error(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "events.jsonl").write_text(
        '{"event_id":\n{"also":"bad"}\n', encoding="utf-8"
    )

    with pytest.raises(PersistenceError):
        EventStore(tmp_path)


def test_jsonl_reload_recovers_latest_status_per_event(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event = store.create_event(
        confirmed_review(),
        b"snapshot",
        detected_at=datetime(2026, 1, 1, tzinfo=UTC),
        countdown_seconds=15,
    )
    store.transition(event.event_id, "sending")
    store.transition(event.event_id, "sent")

    reloaded = EventStore(tmp_path)

    assert len(reloaded.list_events()) == 1
    assert reloaded.get(event.event_id).status == "sent"


class FakeHttpResponse:
    status_code = 202


class FakeHttpClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> FakeHttpResponse:
        self.calls.append({"url": url, **kwargs})
        return FakeHttpResponse()


@pytest.mark.asyncio
async def test_webhook_payload_and_idempotency_header(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    event = store.create_event(
        confirmed_review(),
        b"snapshot",
        detected_at=datetime(2026, 1, 1, tzinfo=UTC),
        countdown_seconds=15,
    )
    client = FakeHttpClient()
    notifier = WebhookNotifier("http://127.0.0.1:9000/falls", client=client)

    result = await notifier.send(event, b"snapshot")

    assert result.status == "sent"
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["timeout"] == 5.0
    assert call["headers"]["X-Fall-Detector-Event-ID"] == str(event.event_id)
    assert call["json"] == {
        "event_id": str(event.event_id),
        "detected_at": event.detected_at.isoformat(),
        "classification": "fall",
        "confidence": 0.9,
        "reason": "A visible person fell and remained down.",
        "needs_help": True,
        "source": "local_webcam",
        "snapshot_data_url": "data:image/jpeg;base64,c25hcHNob3Q=",
    }
