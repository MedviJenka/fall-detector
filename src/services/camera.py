import asyncio
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.services.motion import FallCandidateDetector, MotionSample


@dataclass(frozen=True, slots=True)
class FramePacket:
    timestamp: float
    jpeg: bytes


@dataclass(frozen=True, slots=True)
class CandidateBundle:
    candidate_timestamp: float
    frames: tuple[bytes, bytes, bytes, bytes]


@dataclass(frozen=True, slots=True)
class CameraError:
    message: str


CameraEvent = CandidateBundle | CameraError


def select_candidate_frames(
    packets: Sequence[FramePacket], candidate_timestamp: float
) -> CandidateBundle:
    if not packets:
        raise ValueError("Cannot create a candidate bundle without camera frames")
    targets = (-1.5, -0.5, 0.0, 1.0)
    frames = tuple(
        min(
            packets,
            key=lambda packet, target=candidate_timestamp + offset: abs(
                packet.timestamp - target
            ),
        ).jpeg
        for offset in targets
    )
    return CandidateBundle(
        candidate_timestamp=candidate_timestamp,
        frames=(frames[0], frames[1], frames[2], frames[3]),
    )


def offer_camera_event(queue: asyncio.Queue[CameraEvent], event: CameraEvent) -> None:
    if queue.full():
        return
    queue.put_nowait(event)


class CameraWorker:
    def __init__(
        self,
        camera_index: int,
        event_queue: asyncio.Queue[CameraEvent],
    ) -> None:
        self._camera_index = camera_index
        self._event_queue = event_queue
        self._latest_jpeg: bytes | None = None
        self._latest_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest_jpeg(self) -> bytes | None:
        with self._latest_lock:
            return self._latest_jpeg

    async def start(self) -> None:
        if self.running:
            return
        self._loop = asyncio.get_running_loop()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="fall-detector-camera",
            daemon=True,
        )
        self._thread.start()

    async def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, 2.0)
        self._thread = None
        with self._latest_lock:
            self._latest_jpeg = None

    def _capture_loop(self) -> None:
        capture: Any = cv2.VideoCapture(self._camera_index)
        try:
            if not capture.isOpened():
                self._publish(CameraError("Unable to open the configured camera."))
                return

            subtractor = cv2.createBackgroundSubtractorMOG2(
                history=100,
                varThreshold=32,
                detectShadows=False,
            )
            detector = FallCandidateDetector()
            ring: deque[FramePacket] = deque()
            last_processed = float("-inf")
            pending_candidate: float | None = None

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    self._publish(CameraError("The camera stopped providing frames."))
                    return

                timestamp = time.monotonic()
                resized = self._resize(frame)
                jpeg = self._encode_jpeg(resized)
                packet = FramePacket(timestamp=timestamp, jpeg=jpeg)
                ring.append(packet)
                self._trim_ring(ring, timestamp)
                with self._latest_lock:
                    self._latest_jpeg = jpeg

                if timestamp - last_processed >= 0.2:
                    sample = self._motion_sample(subtractor, resized, timestamp)
                    last_processed = timestamp
                    if pending_candidate is None and detector.observe(sample):
                        pending_candidate = timestamp

                if (
                    pending_candidate is not None
                    and timestamp >= pending_candidate + 1.0
                ):
                    self._publish(select_candidate_frames(ring, pending_candidate))
                    pending_candidate = None
        except (cv2.error, RuntimeError):
            self._publish(CameraError("Camera processing failed."))
        finally:
            capture.release()

    @staticmethod
    def _resize(frame: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        height, width = frame.shape[:2]
        if width <= 640:
            return frame
        scale = 640 / width
        return cv2.resize(
            frame,
            (640, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _encode_jpeg(frame: np.ndarray[Any, Any]) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 75],
        )
        if not ok:
            raise RuntimeError("Camera frame encoding failed")
        return encoded.tobytes()

    @staticmethod
    def _motion_sample(
        subtractor: Any,
        frame: np.ndarray[Any, Any],
        timestamp: float,
    ) -> MotionSample:
        mask = subtractor.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_width * frame_height
        largest = max(contours, key=cv2.contourArea, default=None)
        area_ratio = 0.0 if largest is None else cv2.contourArea(largest) / frame_area
        bbox = None
        if largest is not None and area_ratio >= 0.03:
            x, y, width, height = cv2.boundingRect(largest)
            bbox = (x, y, width, height)

        return MotionSample(
            timestamp=timestamp,
            frame_width=frame_width,
            frame_height=frame_height,
            bbox=bbox,
            foreground_area_ratio=area_ratio,
        )

    @staticmethod
    def _trim_ring(ring: deque[FramePacket], timestamp: float) -> None:
        while ring and timestamp - ring[0].timestamp > 4.0:
            ring.popleft()

    def _publish(self, event: CameraEvent) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                offer_camera_event, self._event_queue, event
            )
