from .base import Task, TaskRegistry
from .string_tasks import ReverseTask, UppercaseTask, PigLatinTask
from .string_tasks_extra import FirstLetterTask, RepeatWordTask
from .numeric_tasks import LengthTask, Linear2xTask
from .semantic_tasks import SentimentTask, AntonymTask
from .pattern_tasks import PatternCompletionTask

__all__ = [
    "Task", "TaskRegistry",
    "ReverseTask", "UppercaseTask", "PigLatinTask",
    "FirstLetterTask", "RepeatWordTask",
    "LengthTask", "Linear2xTask",
    "SentimentTask", "AntonymTask",
    "PatternCompletionTask",
]
