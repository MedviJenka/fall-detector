from functools import cached_property

from crewai import LLM, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from pydantic import SecretStr


class AgentConfig:
    agents_config: dict = "config/agents.yaml"
    tasks_config: dict = "config/tasks.yaml"
    agents: list[BaseAgent]
    tasks: list[Task]

    def __init__(
        self,
        *,
        model: str,
        api_key: SecretStr | str,
        verbose: bool = False,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.verbose = verbose
        self.timeout_seconds = timeout_seconds

    @cached_property
    def llm(self) -> LLM:
        key = (
            self.api_key.get_secret_value()
            if isinstance(self.api_key, SecretStr)
            else self.api_key
        )
        return LLM(
            model=self.model,
            api_key=key,
            temperature=0.0,
            timeout=self.timeout_seconds,
        )
