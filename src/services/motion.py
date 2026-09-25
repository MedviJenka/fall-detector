from collections import deque
from dataclasses import dataclass

BoundingBox = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class MotionSample:
    timestamp: float
    frame_width: int
    frame_height: int
    bbox: BoundingBox | None
    foreground_area_ratio: float

    @property
    def centroid_y(self) -> float:
        if self.bbox is None:
            raise ValueError("A sample without a bounding box has no centroid")
        _, y, _, height = self.bbox
        return y + height / 2


class FallCandidateDetector:
    def __init__(
        self,
        *,
        warmup_frames: int = 10,
        minimum_foreground_ratio: float = 0.03,
        window_seconds: float = 2.0,
        transition_seconds: float = 1.2,
    ) -> None:
        self._warmup_remaining = warmup_frames
        self._minimum_foreground_ratio = minimum_foreground_ratio
        self._window_seconds = window_seconds
        self._transition_seconds = transition_seconds
        self._samples: deque[MotionSample] = deque(maxlen=11)

    def observe(self, sample: MotionSample) -> bool:
        if self._warmup_remaining:
            self._warmup_remaining -= 1
            return False

        self._discard_expired(sample.timestamp)
        if (
            sample.bbox is None
            or sample.foreground_area_ratio < self._minimum_foreground_ratio
        ):
            return False

        is_candidate = self._matches_prior_upright(sample)
        if is_candidate:
            self._samples.clear()
            return True

        self._samples.append(sample)
        return False

    def _discard_expired(self, timestamp: float) -> None:
        while (
            self._samples
            and timestamp - self._samples[0].timestamp > self._window_seconds
        ):
            self._samples.popleft()

    def _matches_prior_upright(self, final: MotionSample) -> bool:
        if not self._is_valid_final_posture(final):
            return False

        minimum_descent = final.frame_height * 0.18
        for earlier in self._samples:
            elapsed = final.timestamp - earlier.timestamp
            if elapsed < 0 or elapsed > self._transition_seconds:
                continue
            if not self._is_upright(earlier):
                continue
            if final.centroid_y - earlier.centroid_y >= minimum_descent:
                return True
        return False

    @staticmethod
    def _is_upright(sample: MotionSample) -> bool:
        if sample.bbox is None:
            return False
        _, _, width, height = sample.bbox
        return height >= 1.2 * width

    @staticmethod
    def _is_valid_final_posture(sample: MotionSample) -> bool:
        if sample.bbox is None:
            return False
        _, _, width, height = sample.bbox
        is_horizontal = width >= 1.1 * height
        is_low = sample.centroid_y >= 0.65 * sample.frame_height
        return is_horizontal or is_low
