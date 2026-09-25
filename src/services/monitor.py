import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from src.ai.agents.alert_agent.fall_reviewer import FallReviewError
from src.models import DeliveryResult, FallEvent, MonitorSnapshot
from src.services.camera import CameraError, CameraEvent
from src.services.events import EventStore


class InvalidTransitionError(RuntimeError):
    pass


class EventNotFoundError(LookupError):
    pass


def utc_now() -> datetime:
    return datetime.now(UTC)


class MonitorService:
    def __init__(
        self,
        *,
        camera: Any,
        reviewer: Any,
        event_store: EventStore,
        notifier: Any,
        event_queue: asyncio.Queue[CameraEvent],
        countdown_seconds: int,
        clock: Callable[[], datetime] = utc_now,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._camera = camera
        self._reviewer = reviewer
        self._event_store = event_store
        self._notifier = notifier
        self._event_queue = event_queue
        self._countdown_seconds = countdown_seconds
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._state = "stopped"
        self._camera_status = "off"
        self._webhook_status = "idle"
        self._latest_review = None
        self._active_event_id: UUID | None = None
        self._error: str | None = None
        self._cooldown_until: datetime | None = None
        self._consumer_task: asyncio.Task[None] | None = None
        self._resume_task: asyncio.Task[None] | None = None
        self._deadline_tasks: dict[UUID, asyncio.Task[None]] = {}

    def _drain_camera_events(self) -> None:
        while True:
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    async def start(self) -> None:
        self._drain_camera_events()
        async with self._lock:
            if self._state not in {"stopped", "camera_error"}:
                raise InvalidTransitionError("Monitoring is already active.")
            self._state = "monitoring"
            self._camera_status = "starting"
            self._error = None
        await self._camera.start()
        async with self._lock:
            self._camera_status = "running"
        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(self._consume_camera_events())

    async def stop(self) -> None:
        async with self._lock:
            if self._state == "stopped":
                raise InvalidTransitionError("Monitoring is already stopped.")
            active = self._active_event()
            if active is not None and active.status in {"pending", "sending"}:
                raise InvalidTransitionError(
                    "Monitoring cannot stop while an alert is pending or sending."
                )
            self._state = "stopped"
            self._camera_status = "off"
            self._error = None
        await self._cancel_consumer()
        await self._camera.stop()
        self._drain_camera_events()

    async def shutdown(self) -> None:
        await self._cancel_consumer()
        tasks = [*self._deadline_tasks.values()]
        if self._resume_task is not None:
            tasks.append(self._resume_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._deadline_tasks.clear()
        self._resume_task = None
        await self._camera.stop()
        self._drain_camera_events()
        async with self._lock:
            self._state = "stopped"
            self._camera_status = "off"

    async def handle_camera_event(self, event: CameraEvent) -> bool:
        if isinstance(event, CameraError):
            async with self._lock:
                self._state = "camera_error"
                self._camera_status = "error"
                self._error = event.message
            return False

        async with self._lock:
            now = self._clock()
            if self._state != "monitoring":
                return False
            if self._cooldown_until is not None and now < self._cooldown_until:
                return False
            self._state = "reviewing"
            self._error = None

        try:
            review = await self._reviewer.review(event.frames)
        except FallReviewError as error:
            await self._record_review_failure(str(error))
            return True
        except Exception:  # noqa: BLE001 - sanitize reviewer boundary failures
            await self._record_review_failure("AI review failed.")
            return True

        async with self._lock:
            now = self._clock()
            self._latest_review = review
            self._cooldown_until = now + timedelta(seconds=20)
            if (
                review.classification == "fall"
                and review.person_visible
                and review.confidence >= 0.70
            ):
                created = self._event_store.create_event(
                    review,
                    event.frames[-1],
                    detected_at=now,
                    countdown_seconds=self._countdown_seconds,
                )
                self._active_event_id = created.event_id
                self._state = "countdown"
                self._webhook_status = "pending"
                self._schedule_deadline(created)
            else:
                self._state = "monitoring"
            return True

    async def cancel_alert(self, event_id: UUID) -> FallEvent:
        async with self._lock:
            event = self._get_event(event_id)
            if event.status != "pending" or self._clock() >= event.deadline:
                raise InvalidTransitionError("This alert can no longer be cancelled.")
            cancelled = self._event_store.transition(event_id, "cancelled")
            if self._active_event_id == event_id:
                self._active_event_id = None
                self._state = "monitoring"
                self._webhook_status = "idle"
            task = self._deadline_tasks.pop(event_id, None)
            if task is not None:
                task.cancel()
            return cancelled

    async def deliver_due_event(self, event_id: UUID) -> bool:
        async with self._lock:
            event = self._get_event(event_id)
            if event.status != "pending":
                return False
            if self._clock() < event.deadline:
                return False
            sending = self._event_store.transition(event_id, "sending")
            self._state = "sending"
            self._webhook_status = "sending"

        try:
            result = await self._notifier.send(
                sending,
                self._event_store.read_snapshot(event_id),
            )
        except Exception:  # noqa: BLE001 - preserve alert state on notifier failure
            result = DeliveryResult(
                status="delivery_failed",
                error="Caregiver notification failed.",
            )

        async with self._lock:
            self._event_store.transition(
                event_id,
                result.status,
                error=result.error,
            )
            if self._active_event_id == event_id:
                self._active_event_id = None
            self._state = "monitoring"
            self._webhook_status = "sent" if result.status == "sent" else "failed"
            self._error = result.error
            self._deadline_tasks.pop(event_id, None)
        return True

    async def snapshot(self) -> MonitorSnapshot:
        async with self._lock:
            active = self._active_event()
            return MonitorSnapshot(
                state=self._state,
                latest_review=self._latest_review,
                active_event_id=self._active_event_id,
                alert_deadline=active.deadline if active is not None else None,
                camera_status=self._camera_status,
                webhook_status=self._webhook_status,
                error=self._error,
                persistence_warning=self._event_store.persistence_warning,
            )

    def events(self) -> list[FallEvent]:
        return self._event_store.list_events()

    def latest_frame(self) -> bytes | None:
        return self._camera.latest_jpeg()

    async def _consume_camera_events(self) -> None:
        while True:
            event = await self._event_queue.get()
            await self.handle_camera_event(event)

    async def _record_review_failure(self, message: str) -> None:
        async with self._lock:
            self._state = "ai_error"
            self._error = message
            self._cooldown_until = self._clock() + timedelta(seconds=20)
            if self._resume_task is not None:
                self._resume_task.cancel()
            self._resume_task = asyncio.create_task(self._resume_after_ai_error())

    async def _resume_after_ai_error(self) -> None:
        await self._sleep(20.0)
        async with self._lock:
            if self._state == "ai_error":
                self._state = "monitoring"
                self._error = None

    def _schedule_deadline(self, event: FallEvent) -> None:
        task = asyncio.create_task(self._wait_for_deadline(event))
        self._deadline_tasks[event.event_id] = task

    async def _wait_for_deadline(self, event: FallEvent) -> None:
        delay = max(0.0, (event.deadline - self._clock()).total_seconds())
        await self._sleep(delay)
        await self.deliver_due_event(event.event_id)

    def _get_event(self, event_id: UUID) -> FallEvent:
        try:
            return self._event_store.get(event_id)
        except KeyError as error:
            raise EventNotFoundError("Fall event was not found.") from error

    def _active_event(self) -> FallEvent | None:
        if self._active_event_id is None:
            return None
        return self._event_store.get(self._active_event_id)

    async def _cancel_consumer(self) -> None:
        task = self._consumer_task
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._consumer_task = None
