#!/usr/bin/env python3
"""Tests for task implementations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tasks import TaskRegistry


def test_all_tasks_registered():
    names = TaskRegistry.list_names()
    expected = [
        "reverse", "uppercase", "pig_latin", "length",
        "linear_2x", "sentiment", "antonym", "pattern_completion",
    ]
    for name in expected:
        assert name in names, f"Task {name} not registered"
    print(f"OK: All {len(expected)} tasks registered")


def test_task_demos():
    tasks = TaskRegistry.all_tasks()
    for name, task in tasks.items():
        demos = task.generate_demos(5)
        assert len(demos) == 5, f"{name}: expected 5 demos, got {len(demos)}"
        for inp, out in demos:
            assert isinstance(inp, str) and isinstance(out, str)
            assert task.compute_answer(inp) == out, \
                f"{name}: demo mismatch for {inp!r}: expected {task.compute_answer(inp)!r}, got {out!r}"
    print("OK: All task demos are self-consistent")


def test_task_test_inputs():
    tasks = TaskRegistry.all_tasks()
    for name, task in tasks.items():
        test_inputs = task.generate_test_inputs(50)
        assert len(test_inputs) == 50, f"{name}: expected 50 test inputs, got {len(test_inputs)}"
        # Check no overlap with demos
        demo_inputs = {d[0] for d in task.generate_demos(5)}
        for ti in test_inputs:
            assert ti not in demo_inputs, f"{name}: test input {ti!r} overlaps with demo"
    print("OK: All tasks generate 50 non-overlapping test inputs")


def test_scoring():
    reverse = TaskRegistry.get("reverse")
    assert reverse.score_output("hello", "olleh") == "correct"
    assert reverse.score_output("hello", "hello") == "incorrect"
    assert reverse.score_output("hello", "") == "malformed"

    sentiment = TaskRegistry.get("sentiment")
    assert sentiment.score_output("happy", "positive") == "correct"
    assert sentiment.score_output("happy", "negative") == "incorrect"
    assert sentiment.score_output("happy", "this is positive sentiment") == "correct"

    pattern = TaskRegistry.get("pattern_completion")
    assert pattern.score_output("C D C D C", "D") == "correct"
    assert pattern.score_output("C D C D C", "C") == "incorrect"

    print("OK: Scoring works correctly")


def test_prompt_format():
    task = TaskRegistry.get("reverse")
    demos = task.generate_demos(3)
    prompt = task.format_prompt(demos, "test")
    assert "Input: apple" in prompt
    assert "Output: elppa" in prompt
    assert "Input: test" in prompt
    assert prompt.endswith("Output:")
    print("OK: Prompt formatting works")
    print(f"Sample prompt:\n{prompt}")


if __name__ == "__main__":
    test_all_tasks_registered()
    test_task_demos()
    test_task_test_inputs()
    test_scoring()
    test_prompt_format()
    print("\nAll tests passed!")
