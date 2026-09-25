import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from src.ai.config import AgentConfig
from src.models import FallReview


class FallReviewError(RuntimeError):
    pass


class ReviewCrew(Protocol):
    def kickoff(self, *, inputs: dict[str, str]) -> Any: ...


CrewBuilder = Callable[[], ReviewCrew]
_SKILL_DIR = Path(__file__).with_name("skills") / "fall-alert"


@CrewBase
class FallAlertCrew(AgentConfig):
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def fall_reviewer(self) -> Agent:
        configured = Agent(
            config=self.agents_config["fall_reviewer"],  # type: ignore[index]
            llm=self.llm,
            verbose=self.verbose,
        )
        configured.skills = [_SKILL_DIR]
        return configured

    @task
    def fall_review_task(self) -> Task:
        return Task(
            config=self.tasks_config["fall_review_task"],  # type: ignore[index]
            output_pydantic=FallReview,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            memory=False,
            cache=False,
            verbose=self.verbose,
        )


def build_fall_review_crew(config: AgentConfig) -> Crew:
    return FallAlertCrew(
        model=config.model,
        api_key=config.api_key,
        verbose=config.verbose,
        timeout_seconds=config.timeout_seconds,
    ).crew()


class CrewAIFallReviewer:
    def __init__(
        self,
        *,
        config: AgentConfig,
        crew_builder: CrewBuilder | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._config = config
        self._crew_builder = crew_builder or (
            lambda: build_fall_review_crew(self._config)
        )
        self._timeout_seconds = timeout_seconds or config.timeout_seconds

    async def review(self, frames: Sequence[bytes]) -> FallReview:
        if len(frames) != 4:
            raise FallReviewError("AI review requires exactly four frames.")

        with TemporaryDirectory(prefix="fall-review-") as directory:
            root = Path(directory)
            image_paths = tuple(root / f"frame-{index}.jpg" for index in range(1, 5))
            for path, frame in zip(image_paths, frames):
                path.write_bytes(frame)
            inputs = {
                f"image_{index}": str(path.resolve())
                for index, path in enumerate(image_paths, start=1)
            }

            try:
                review_crew = self._crew_builder()
                result = await asyncio.wait_for(
                    asyncio.to_thread(review_crew.kickoff, inputs=inputs),
                    timeout=self._timeout_seconds,
                )
                parsed = getattr(result, "pydantic", None)
                if parsed is None:
                    raise FallReviewError("AI review returned no structured result.")
                return (
                    parsed
                    if isinstance(parsed, FallReview)
                    else FallReview.model_validate(parsed)
                )
            except TimeoutError as error:
                raise FallReviewError("AI review timed out.") from error
            except FallReviewError:
                raise
            except Exception as error:
                raise FallReviewError("AI review service was unavailable.") from error
