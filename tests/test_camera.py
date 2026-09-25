import asyncio
from dataclasses import FrozenInstanceError

import pytest

from src.services.camera import (
    CameraError,
    CandidateBundle,
    FramePacket,
    offer_camera_event,
    select_candidate_frames,
)


def test_candidate_frames_are_selected_nearest_each_ordered_target() -> None:
    packets = [
        FramePacket(timestamp=8.2, jpeg=b"early"),
        FramePacket(timestamp=8.6, jpeg=b"minus-1.5"),
        FramePacket(timestamp=9.5, jpeg=b"minus-0.5"),
        FramePacket(timestamp=10.0, jpeg=b"candidate"),
        FramePacket(timestamp=11.0, jpeg=b"plus-1"),
    ]

    bundle = select_candidate_frames(packets, candidate_timestamp=10.0)

    assert bundle.frames == (
        b"minus-1.5",
        b"minus-0.5",
        b"candidate",
        b"plus-1",
    )
    with pytest.raises(FrozenInstanceError):
        bundle.frames = ()  # type: ignore[misc]


def test_full_camera_event_queue_drops_new_bundle() -> None:
    queue: asyncio.Queue[CandidateBundle | CameraError] = asyncio.Queue(maxsize=1)
    first = CameraError("first")
    queue.put_nowait(first)

    offer_camera_event(
        queue,
        CandidateBundle(candidate_timestamp=10.0, frames=(b"1", b"2", b"3", b"4")),
    )

    assert queue.get_nowait() is first
