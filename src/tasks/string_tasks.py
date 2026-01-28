"""String manipulation tasks: Reverse, Uppercase, Pig Latin."""

from .base import Task, TaskRegistry

# Word pools for string tasks (4-7 characters, lowercase)
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


@TaskRegistry.register("reverse")
class ReverseTask(Task):
    name = "reverse"
    regime = "procedural"
    description = "Reverse the input string"

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        words = _DEMO_WORDS[:n]
        return [(w, w[::-1]) for w in words]

    def generate_test_inputs(self, n: int) -> list[str]:
        pool = list(_TEST_WORDS)
        self.rng.shuffle(pool)
        return pool[:n]

    def compute_answer(self, inp: str) -> str:
        return inp[::-1]


@TaskRegistry.register("uppercase")
class UppercaseTask(Task):
    name = "uppercase"
    regime = "procedural"
    description = "Convert to uppercase"

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        words = _DEMO_WORDS[:n]
        return [(w, w.upper()) for w in words]

    def generate_test_inputs(self, n: int) -> list[str]:
        pool = list(_TEST_WORDS)
        self.rng.shuffle(pool)
        return pool[:n]

    def compute_answer(self, inp: str) -> str:
        return inp.upper()


@TaskRegistry.register("pig_latin")
class PigLatinTask(Task):
    name = "pig_latin"
    regime = "procedural"
    description = "Convert to pig latin"

    VOWELS = set("aeiou")

    def _to_pig_latin(self, word: str) -> str:
        if word[0] in self.VOWELS:
            return word + "ay"
        # Find first vowel
        for i, c in enumerate(word):
            if c in self.VOWELS:
                return word[i:] + word[:i] + "ay"
        return word + "ay"

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        words = _DEMO_WORDS[:n]
        return [(w, self._to_pig_latin(w)) for w in words]

    def generate_test_inputs(self, n: int) -> list[str]:
        pool = list(_TEST_WORDS)
        self.rng.shuffle(pool)
        return pool[:n]

    def compute_answer(self, inp: str) -> str:
        return self._to_pig_latin(inp)
