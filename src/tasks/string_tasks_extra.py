"""Replacement procedural string tasks: FirstLetter, RepeatWord."""

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


@TaskRegistry.register("first_letter")
class FirstLetterTask(Task):
    """Extract the first letter of the input word."""
    name = "first_letter"
    regime = "procedural"
    description = "Extract the first letter"

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        words = _DEMO_WORDS[:n]
        return [(w, w[0]) for w in words]

    def generate_test_inputs(self, n: int) -> list[str]:
        pool = list(_TEST_WORDS)
        self.rng.shuffle(pool)
        return pool[:n]

    def compute_answer(self, inp: str) -> str:
        return inp[0]

    def score_output(self, inp: str, output: str) -> str:
        expected = self.compute_answer(inp)
        cleaned = output.strip().split("\n")[0].strip()
        if cleaned == expected or cleaned == expected.upper() or cleaned == expected.lower():
            return "correct"
        if not cleaned:
            return "malformed"
        return "incorrect"


@TaskRegistry.register("repeat_word")
class RepeatWordTask(Task):
    """Repeat the input word twice with a space."""
    name = "repeat_word"
    regime = "procedural"
    description = "Repeat the word twice"

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        words = _DEMO_WORDS[:n]
        return [(w, f"{w} {w}") for w in words]

    def generate_test_inputs(self, n: int) -> list[str]:
        pool = list(_TEST_WORDS)
        self.rng.shuffle(pool)
        return pool[:n]

    def compute_answer(self, inp: str) -> str:
        return f"{inp} {inp}"
