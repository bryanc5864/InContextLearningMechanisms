"""Base task class and registry for ICL experiments."""

from abc import ABC, abstractmethod
from typing import Literal
import random


class Task(ABC):
    """Abstract base class for ICL tasks."""

    name: str
    regime: str
    description: str

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    @abstractmethod
    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        """Generate n demonstration (input, output) pairs."""
        ...

    @abstractmethod
    def generate_test_inputs(self, n: int) -> list[str]:
        """Generate n unique test inputs (no overlap with demos)."""
        ...

    @abstractmethod
    def compute_answer(self, inp: str) -> str:
        """Compute the correct answer for a given input."""
        ...

    def score_output(self, inp: str, output: str) -> Literal["correct", "incorrect", "malformed"]:
        """Score a model output against the correct answer."""
        expected = self.compute_answer(inp)
        cleaned = output.strip().split("\n")[0].strip()
        if cleaned == expected:
            return "correct"
        if not cleaned or len(cleaned) > 3 * len(expected) + 10:
            return "malformed"
        return "incorrect"

    def format_prompt(self, demos: list[tuple[str, str]], test_input: str) -> str:
        """Format a few-shot prompt."""
        lines = []
        for inp, out in demos:
            lines.append(f"Input: {inp}")
            lines.append(f"Output: {out}")
            lines.append("")
        lines.append(f"Input: {test_input}")
        lines.append("Output:")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(regime={self.regime!r})"


class TaskRegistry:
    """Registry of all available tasks."""

    _tasks: dict[str, type[Task]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(task_cls: type[Task]):
            cls._tasks[name] = task_cls
            return task_cls
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> Task:
        if name not in cls._tasks:
            raise KeyError(f"Unknown task: {name}. Available: {list(cls._tasks.keys())}")
        return cls._tasks[name](**kwargs)

    @classmethod
    def all_tasks(cls, **kwargs) -> dict[str, Task]:
        return {name: cls.get(name, **kwargs) for name in cls._tasks}

    @classmethod
    def list_names(cls) -> list[str]:
        return list(cls._tasks.keys())
