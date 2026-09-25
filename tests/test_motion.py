from src.services.motion import FallCandidateDetector, MotionSample


def sample(
    timestamp: float,
    bbox: tuple[int, int, int, int] | None,
    *,
    foreground_area_ratio: float = 0.1,
) -> MotionSample:
    return MotionSample(
        timestamp=timestamp,
        frame_width=640,
        frame_height=480,
        bbox=bbox,
        foreground_area_ratio=foreground_area_ratio,
    )


def warm_up(detector: FallCandidateDetector) -> None:
    for index in range(10):
        assert detector.observe(sample(index * 0.2, None)) is False


def test_warmup_suppresses_candidate() -> None:
    detector = FallCandidateDetector()

    for index in range(9):
        detector.observe(sample(index * 0.2, None))

    assert detector.observe(sample(1.8, (250, 80, 80, 220))) is False
    assert detector.observe(sample(2.0, (180, 300, 260, 100))) is False


def test_insufficient_foreground_is_ignored() -> None:
    detector = FallCandidateDetector()
    warm_up(detector)

    detector.observe(sample(2.0, (250, 80, 80, 220)))

    assert (
        detector.observe(sample(2.8, (180, 300, 260, 100), foreground_area_ratio=0.02))
        is False
    )


def test_slow_descent_is_not_a_candidate() -> None:
    detector = FallCandidateDetector()
    warm_up(detector)

    detector.observe(sample(2.0, (250, 80, 80, 220)))

    assert detector.observe(sample(3.4, (180, 300, 260, 100))) is False


def test_rapid_upright_to_horizontal_descent_is_a_candidate() -> None:
    detector = FallCandidateDetector()
    warm_up(detector)

    detector.observe(sample(2.0, (250, 80, 80, 220)))

    assert detector.observe(sample(2.8, (180, 300, 260, 100))) is True


def test_low_final_centroid_is_a_candidate_without_horizontal_box() -> None:
    detector = FallCandidateDetector()
    warm_up(detector)

    detector.observe(sample(2.0, (250, 60, 80, 220)))

    assert detector.observe(sample(2.8, (260, 330, 90, 160))) is True


def test_detector_resets_after_trigger() -> None:
    detector = FallCandidateDetector()
    warm_up(detector)

    detector.observe(sample(2.0, (250, 80, 80, 220)))
    assert detector.observe(sample(2.8, (180, 300, 260, 100))) is True

    assert detector.observe(sample(3.0, (180, 310, 260, 100))) is False


def test_timestamp_window_expires_old_upright_sample() -> None:
    detector = FallCandidateDetector()
    warm_up(detector)

    detector.observe(sample(2.0, (250, 80, 80, 220)))
    detector.observe(sample(3.0, None))

    assert detector.observe(sample(4.1, (180, 300, 260, 100))) is False
