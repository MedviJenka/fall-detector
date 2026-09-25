from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

FallClassification = Literal["fall", "not_fall", "uncertain"]
EventStatus = Literal["pending", "cancelled", "sending", "sent", "delivery_failed"]
MonitorState = Literal[
    "stopped",
    "monitoring",
    "reviewing",
    "countdown",
    "sending",
    "camera_error",
    "ai_error",
]


class FallReview(BaseModel):
    classification: FallClassification
    confidence: float = Field(ge=0.0, le=1.0)
    person_visible: bool
    reason: str = Field(min_length=1, max_length=240)


class FallEvent(BaseModel):
    event_id: UUID
    detected_at: datetime
    review: FallReview
    snapshot_path: Path
    status: EventStatus
    deadline: datetime
    delivery_error: str | None = None


class DeliveryResult(BaseModel):
    status: Literal["sent", "delivery_failed"]
    error: str | None = None


class MonitorSnapshot(BaseModel):
    state: MonitorState
    latest_review: FallReview | None = None
    active_event_id: UUID | None = None
    alert_deadline: datetime | None = None
    camera_status: Literal["off", "starting", "running", "error"] = "off"
    webhook_status: Literal["idle", "pending", "sending", "sent", "failed"] = "idle"
    error: str | None = None
    persistence_warning: str | None = None
