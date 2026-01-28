"""Numeric tasks: Length counting, Linear regression (2x)."""

from .base import Task, TaskRegistry

_DEMO_WORDS = ["apple", "brain", "cloud", "delta", "flame"]
_TEST_WORDS = [
    "grape", "house", "juice", "knife", "lemon", "mango", "nerve", "olive",
    "pearl", "queen", "river", "stone", "tower", "uncle", "valve", "wheat",
    "youth", "zebra", "ankle", "badge", "candy", "dream", "eagle", "frost",
    "globe", "honey", "ivory", "jolly", "kayak", "lunar", "maple", "noble",
    "ocean", "piano", "quilt", "robin", "sugar", "trend", "urban", "vivid",
    "waltz", "pixel", "crane", "dwarf", "ember", "fjord", "glyph", "haven",
    "index", "joker",
]


@TaskRegistry.register("length")
class LengthTask(Task):
    name = "length"
    regime = "counting"
    description = "Count characters in the input"

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        words = _DEMO_WORDS[:n]
        return [(w, str(len(w))) for w in words]

    def generate_test_inputs(self, n: int) -> list[str]:
        pool = list(_TEST_WORDS)
        self.rng.shuffle(pool)
        return pool[:n]

    def compute_answer(self, inp: str) -> str:
        return str(len(inp))


@TaskRegistry.register("linear_2x")
class Linear2xTask(Task):
    name = "linear_2x"
    regime = "gd_like"
    description = "Multiply input by 2"

    _DEMO_NUMS = [3, 5, 7, 2, 9]
    _TEST_NUMS = list(range(1, 61))  # 1 through 60 (enough after excluding demos)

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        nums = self._DEMO_NUMS[:n]
        return [(str(x), str(2 * x)) for x in nums]

    def generate_test_inputs(self, n: int) -> list[str]:
        pool = [x for x in self._TEST_NUMS if x not in self._DEMO_NUMS]
        self.rng.shuffle(pool)
        return [str(x) for x in pool[:n]]

    def compute_answer(self, inp: str) -> str:
        return str(2 * int(inp))
