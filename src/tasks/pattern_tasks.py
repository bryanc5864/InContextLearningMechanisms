"""Pattern completion tasks: Sequence induction."""

from .base import Task, TaskRegistry


@TaskRegistry.register("pattern_completion")
class PatternCompletionTask(Task):
    name = "pattern_completion"
    regime = "induction"
    description = "Complete the alternating pattern"

    _DEMOS = [
        ("A B A B A", "B"),
        ("1 2 1 2 1", "2"),
        ("X Y X Y X", "Y"),
        ("P Q P Q P", "Q"),
        ("M N M N M", "N"),
    ]

    # Test: alternating patterns with different tokens
    _TEST_PAIRS = [
        ("C D C D C", "D"), ("E F E F E", "F"), ("G H G H G", "H"),
        ("I J I J I", "J"), ("K L K L K", "L"), ("R S R S R", "S"),
        ("T U T U T", "U"), ("V W V W V", "W"), ("3 4 3 4 3", "4"),
        ("5 6 5 6 5", "6"), ("7 8 7 8 7", "8"), ("9 0 9 0 9", "0"),
        ("a b a b a", "b"), ("c d c d c", "d"), ("e f e f e", "f"),
        ("g h g h g", "h"), ("i j i j i", "j"), ("k l k l k", "l"),
        ("m n m n m", "n"), ("o p o p o", "p"), ("q r q r q", "r"),
        ("s t s t s", "t"), ("u v u v u", "v"), ("w x w x w", "x"),
        ("y z y z y", "z"), ("Z Y Z Y Z", "Y"), ("W X W X W", "X"),
        ("Q R Q R Q", "R"), ("O N O N O", "N"), ("B C B C B", "C"),
        ("D E D E D", "E"), ("F G F G F", "G"), ("H I H I H", "I"),
        ("J K J K J", "K"), ("L M L M L", "M"), ("S T S T S", "T"),
        ("U V U V U", "V"), ("2 3 2 3 2", "3"), ("4 5 4 5 4", "5"),
        ("6 7 6 7 6", "7"), ("8 9 8 9 8", "9"), ("1 0 1 0 1", "0"),
        ("@ # @ # @", "#"), ("+ - + - +", "-"), ("! ? ! ? !", "?"),
        ("a c a c a", "c"), ("b d b d b", "d"), ("x z x z x", "z"),
        ("p q p q p", "q"), ("r t r t r", "t"), ("w y w y w", "y"),
    ]

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        return self._DEMOS[:n]

    def generate_test_inputs(self, n: int) -> list[str]:
        pool = list(self._TEST_PAIRS)
        self.rng.shuffle(pool)
        return [pair[0] for pair in pool[:n]]

    def compute_answer(self, inp: str) -> str:
        # For alternating A B A B A, return B
        tokens = inp.strip().split()
        if len(tokens) >= 2:
            return tokens[1]
        return tokens[0]

    def score_output(self, inp: str, output: str) -> str:
        expected = self.compute_answer(inp)
        cleaned = output.strip().split("\n")[0].strip().split()[0] if output.strip() else ""
        if cleaned == expected:
            return "correct"
        if not cleaned:
            return "malformed"
        return "incorrect"
