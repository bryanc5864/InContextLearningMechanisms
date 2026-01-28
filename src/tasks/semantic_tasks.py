"""Semantic tasks: Sentiment classification, Antonym generation."""

from .base import Task, TaskRegistry


@TaskRegistry.register("sentiment")
class SentimentTask(Task):
    name = "sentiment"
    regime = "bayesian"
    description = "Classify sentiment as positive or negative"

    _DEMOS = [
        ("joyful", "positive"),
        ("angry", "negative"),
        ("wonderful", "positive"),
        ("terrible", "negative"),
        ("cheerful", "positive"),
    ]

    _TEST_ITEMS = [
        ("happy", "positive"), ("sad", "negative"), ("excited", "positive"),
        ("fearful", "negative"), ("grateful", "positive"), ("hostile", "negative"),
        ("hopeful", "positive"), ("anxious", "negative"), ("proud", "positive"),
        ("jealous", "negative"), ("loving", "positive"), ("bitter", "negative"),
        ("peaceful", "positive"), ("furious", "negative"), ("delighted", "positive"),
        ("miserable", "negative"), ("confident", "positive"), ("worried", "negative"),
        ("content", "positive"), ("gloomy", "negative"), ("radiant", "positive"),
        ("devastated", "negative"), ("thrilled", "positive"), ("depressed", "negative"),
        ("amused", "positive"), ("disgusted", "negative"), ("inspired", "positive"),
        ("frustrated", "negative"), ("optimistic", "positive"), ("pessimistic", "negative"),
        ("blissful", "positive"), ("resentful", "negative"), ("enthusiastic", "positive"),
        ("melancholy", "negative"), ("ecstatic", "positive"), ("sorrowful", "negative"),
        ("compassionate", "positive"), ("vindictive", "negative"), ("serene", "positive"),
        ("agitated", "negative"), ("jubilant", "positive"), ("despondent", "negative"),
        ("affectionate", "positive"), ("spiteful", "negative"), ("thankful", "positive"),
        ("envious", "negative"), ("elated", "positive"), ("dejected", "negative"),
        ("generous", "positive"), ("cruel", "negative"),
    ]

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        return self._DEMOS[:n]

    def generate_test_inputs(self, n: int) -> list[str]:
        pool = [item[0] for item in self._TEST_ITEMS]
        self.rng.shuffle(pool)
        return pool[:n]

    def compute_answer(self, inp: str) -> str:
        lookup = {item[0]: item[1] for item in self._TEST_ITEMS + list(self._DEMOS)}
        return lookup.get(inp, "positive")  # fallback

    def score_output(self, inp: str, output: str) -> str:
        expected = self.compute_answer(inp)
        cleaned = output.strip().lower().split("\n")[0].strip()
        # Accept partial matches: "positive" in "positive sentiment"
        if expected in cleaned:
            return "correct"
        other = "negative" if expected == "positive" else "positive"
        if other in cleaned:
            return "incorrect"
        return "malformed"


@TaskRegistry.register("antonym")
class AntonymTask(Task):
    name = "antonym"
    regime = "retrieval"
    description = "Return the antonym"

    _DEMOS = [
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
        ("happy", "sad"),
        ("light", "dark"),
    ]

    _TEST_ITEMS = [
        ("tall", "short"), ("old", "young"), ("rich", "poor"),
        ("strong", "weak"), ("hard", "soft"), ("long", "short"),
        ("thick", "thin"), ("wide", "narrow"), ("deep", "shallow"),
        ("loud", "quiet"), ("clean", "dirty"), ("dry", "wet"),
        ("early", "late"), ("full", "empty"), ("open", "closed"),
        ("sharp", "dull"), ("smooth", "rough"), ("sweet", "bitter"),
        ("brave", "coward"), ("bright", "dim"), ("calm", "nervous"),
        ("cheap", "expensive"), ("clear", "cloudy"), ("alive", "dead"),
        ("love", "hate"), ("true", "false"), ("wise", "foolish"),
        ("north", "south"), ("east", "west"), ("win", "lose"),
        ("push", "pull"), ("rise", "fall"), ("give", "take"),
        ("buy", "sell"), ("start", "stop"), ("teach", "learn"),
        ("attack", "defend"), ("accept", "reject"), ("arrive", "depart"),
        ("ancient", "modern"), ("bold", "timid"), ("complex", "simple"),
        ("create", "destroy"), ("expand", "shrink"), ("forget", "remember"),
        ("generous", "selfish"), ("humble", "arrogant"), ("include", "exclude"),
        ("maximum", "minimum"), ("praise", "criticize"),
    ]

    def generate_demos(self, n: int) -> list[tuple[str, str]]:
        return self._DEMOS[:n]

    def generate_test_inputs(self, n: int) -> list[str]:
        pool = [item[0] for item in self._TEST_ITEMS]
        self.rng.shuffle(pool)
        return pool[:n]

    def compute_answer(self, inp: str) -> str:
        lookup = {item[0]: item[1] for item in self._TEST_ITEMS + list(self._DEMOS)}
        return lookup.get(inp, "unknown")

    # Accept multiple valid antonyms for each word
    _ACCEPTABLE = {
        "hot": {"cold", "cool"}, "big": {"small", "little", "tiny"},
        "fast": {"slow"}, "happy": {"sad", "unhappy", "miserable"},
        "light": {"dark", "heavy"}, "tall": {"short", "small"},
        "old": {"young", "new"}, "rich": {"poor"}, "strong": {"weak", "frail"},
        "hard": {"soft", "easy"}, "long": {"short"}, "thick": {"thin", "slim"},
        "wide": {"narrow", "thin"}, "deep": {"shallow"}, "loud": {"quiet", "soft", "silent"},
        "clean": {"dirty", "filthy"}, "dry": {"wet", "moist"}, "early": {"late"},
        "full": {"empty"}, "open": {"closed", "shut"}, "sharp": {"dull", "blunt"},
        "smooth": {"rough", "coarse"}, "sweet": {"bitter", "sour"},
        "brave": {"coward", "cowardly", "timid", "fearful"},
        "bright": {"dim", "dark", "dull"}, "calm": {"nervous", "agitated", "stormy", "anxious"},
        "cheap": {"expensive", "costly"}, "clear": {"cloudy", "unclear", "opaque"},
        "alive": {"dead"}, "love": {"hate", "hatred"}, "true": {"false", "untrue"},
        "wise": {"foolish", "stupid"}, "north": {"south"}, "east": {"west"},
        "win": {"lose", "loss"}, "push": {"pull"}, "rise": {"fall", "drop"},
        "give": {"take", "receive"}, "buy": {"sell"}, "start": {"stop", "end", "finish"},
        "teach": {"learn"}, "attack": {"defend", "defense", "retreat"},
        "accept": {"reject", "decline", "refuse"}, "arrive": {"depart", "leave"},
        "ancient": {"modern", "new", "contemporary"}, "bold": {"timid", "meek", "shy"},
        "complex": {"simple", "easy"}, "create": {"destroy", "demolish"},
        "expand": {"shrink", "contract", "reduce"}, "forget": {"remember", "recall"},
        "generous": {"selfish", "stingy", "greedy"}, "humble": {"arrogant", "proud", "vain"},
        "include": {"exclude", "omit"}, "maximum": {"minimum"},
        "praise": {"criticize", "blame", "condemn"},
    }

    def score_output(self, inp: str, output: str) -> str:
        cleaned = output.strip().lower().split("\n")[0].strip()
        if not cleaned:
            return "malformed"
        acceptable = self._ACCEPTABLE.get(inp, {self.compute_answer(inp)})
        if cleaned in acceptable:
            return "correct"
        return "incorrect"
