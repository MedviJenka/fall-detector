import json
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import ValidationError

from src.models import EventStatus, FallEvent, FallReview


class PersistenceError(RuntimeError):
    pass


class EventStore:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir.resolve()
        self._events_path = self._data_dir / "events.jsonl"
        self._snapshots_dir = self._data_dir / "snapshots"
        self._events: dict[UUID, FallEvent] = {}
        self.persistence_warning: str | None = None
        self._load()

    def create_event(
        self,
        review: FallReview,
        snapshot: bytes,
        *,
        detected_at: datetime,
        countdown_seconds: int,
    ) -> FallEvent:
        event_id = uuid4()
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = self._snapshots_dir / f"{event_id}.jpg"
        snapshot_path.write_bytes(snapshot)
        event = FallEvent(
            event_id=event_id,
            detected_at=detected_at,
            review=review,
            snapshot_path=snapshot_path,
            status="pending",
            deadline=detected_at + timedelta(seconds=countdown_seconds),
        )
        self._events[event_id] = event
        self._append(event)
        return event

    def transition(
        self,
        event_id: UUID,
        status: EventStatus,
        *,
        error: str | None = None,
    ) -> FallEvent:
        current = self._events[event_id]
        updated = current.model_copy(update={"status": status, "delivery_error": error})
        self._events[event_id] = updated
        self._append(updated)
        return updated

    def get(self, event_id: UUID) -> FallEvent:
        return self._events[event_id]

    def list_events(self) -> list[FallEvent]:
        return sorted(
            self._events.values(),
            key=lambda event: event.detected_at,
            reverse=True,
        )[:100]

    def read_snapshot(self, event_id: UUID) -> bytes:
        return self._events[event_id].snapshot_path.read_bytes()

    def _append(self, event: FallEvent) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        record = event.model_dump(mode="json")
        with self._events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()

    def _load(self) -> None:
        if not self._events_path.exists():
            return

        with self._events_path.open("r", encoding="utf-8") as handle:
            line_number = 0
            while True:
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    continue
                try:
                    event = FallEvent.model_validate_json(line)
                except (ValidationError, ValueError) as error:
                    if not handle.read().strip():
                        self.persistence_warning = "The final persistence record was incomplete and was skipped."
                        break
                    raise PersistenceError(
                        f"Malformed persistence record at line {line_number}."
                    ) from error
                self._events[event.event_id] = event
