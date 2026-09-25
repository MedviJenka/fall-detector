import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.ai.agents.alert_agent.fall_reviewer import (
    CrewAIFallReviewer,
    FallAlertCrew,
    FallReviewError,
    build_fall_review_crew,
)
from src.ai.config import AgentConfig
from src.models import FallReview


class FakeCrew:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0
        self.inputs: dict[str, str] | None = None
        self.frames: list[bytes] = []

    def kickoff(self, *, inputs: dict[str, str]) -> Any:
        self.calls += 1
        self.inputs = inputs
        self.frames = [
            Path(inputs[f"image_{index}"]).read_bytes() for index in range(1, 5)
        ]
        if self.error is not None:
            raise self.error
        return self.result


def config() -> AgentConfig:
    return AgentConfig(model="vision-model", api_key="test-key")


@pytest.mark.asyncio
async def test_four_frames_remain_ordered_and_parsed_review_passes_through() -> None:
    parsed = FallReview(
        classification="fall",
        confidence=0.82,
        person_visible=True,
        reason="One person fell and remained down.",
    )
    crew = FakeCrew(SimpleNamespace(pydantic=parsed))
    reviewer = CrewAIFallReviewer(config=config(), crew_builder=lambda: crew)
    frames = (b"first", b"second", b"third", b"fourth")

    result = await reviewer.review(frames)

    assert result == parsed
    assert crew.frames == list(frames)
    assert crew.calls == 1


def test_default_crew_uses_crewbase_yaml_and_fall_alert_skill() -> None:
    assert issubclass(FallAlertCrew, AgentConfig)

    crew = build_fall_review_crew(config())

    assert len(crew.agents) == 1
    assert crew.agents[0].multimodal is True
    assert crew.tasks[0].output_pydantic is FallReview
    assert "{image_1}" in crew.tasks[0].description
    skill_paths = [Path(skill) for skill in crew.agents[0].skills or []]
    assert any(path.name == "fall-alert" for path in skill_paths)


@pytest.mark.asyncio
async def test_missing_structured_result_is_a_review_failure() -> None:
    reviewer = CrewAIFallReviewer(
        config=config(),
        crew_builder=lambda: FakeCrew(SimpleNamespace(pydantic=None)),
    )

    with pytest.raises(FallReviewError, match="structured result"):
        await reviewer.review((b"1", b"2", b"3", b"4"))


@pytest.mark.asyncio
async def test_crew_failure_is_sanitized() -> None:
    reviewer = CrewAIFallReviewer(
        config=config(),
        crew_builder=lambda: FakeCrew(error=RuntimeError("secret upstream detail")),
    )

    with pytest.raises(FallReviewError) as captured:
        await reviewer.review((b"1", b"2", b"3", b"4"))

    assert "secret upstream detail" not in str(captured.value)


@pytest.mark.asyncio
async def test_crew_timeout_is_a_review_failure() -> None:
    class SlowCrew(FakeCrew):
        def kickoff(self, *, inputs: dict[str, str]) -> Any:
            time.sleep(0.05)
            return SimpleNamespace(pydantic=None)

    reviewer = CrewAIFallReviewer(
        config=config(),
        crew_builder=SlowCrew,
        timeout_seconds=0.001,
    )

    with pytest.raises(FallReviewError, match="timed out"):
        await reviewer.review((b"1", b"2", b"3", b"4"))


@pytest.mark.asyncio
async def test_low_confidence_fall_is_returned_without_adapter_thresholding() -> None:
    parsed = FallReview(
        classification="fall",
        confidence=0.69,
        person_visible=True,
        reason="Possible transition to the floor.",
    )
    reviewer = CrewAIFallReviewer(
        config=config(),
        crew_builder=lambda: FakeCrew(SimpleNamespace(pydantic=parsed)),
    )

    assert await reviewer.review((b"1", b"2", b"3", b"4")) == parsed
