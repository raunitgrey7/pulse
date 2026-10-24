"""Threshold gate over ``benchmarks/results.json``.

Thresholds are explicit rules (variant, metric, comparison, value) kept in
a JSON file next to the results, so the gate is reviewable in a diff and CI
fails loudly when a model regresses. Values were set below the measured
numbers with a margin for cross-platform floating point differences; see
docs/EVALUATION.md for the reasoning behind each.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPS: dict[str, Callable[[float, float], bool]] = {">=": operator.ge, "<=": operator.le}


@dataclass(frozen=True)
class Rule:
    """One threshold: ``variants[variant][metric] <op> value``."""

    variant: str
    metric: str
    op: str
    value: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Rule:
        if raw["op"] not in OPS:
            raise ValueError(f"unsupported comparison {raw['op']!r}")
        return cls(str(raw["variant"]), str(raw["metric"]), str(raw["op"]), float(raw["value"]))


@dataclass(frozen=True)
class RuleResult:
    rule: Rule
    actual: float | None
    passed: bool

    def line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        shown = "missing" if self.actual is None else f"{self.actual:.6g}"
        return f"{status} {self.rule.variant}.{self.rule.metric} = {shown} (want {self.rule.op} {self.rule.value:g})"


def load_rules(path: Path) -> list[Rule]:
    return [Rule.from_dict(r) for r in json.loads(path.read_text())["rules"]]


def evaluate(results: dict[str, Any], rules: list[Rule]) -> list[RuleResult]:
    """Apply every rule; a missing metric is a failure, never a silent pass."""
    out: list[RuleResult] = []
    for rule in rules:
        actual = results.get("variants", {}).get(rule.variant, {}).get(rule.metric)
        passed = actual is not None and OPS[rule.op](float(actual), rule.value)
        out.append(RuleResult(rule, None if actual is None else float(actual), passed))
    return out


def check(results_path: Path, thresholds_path: Path) -> bool:
    """Print one line per rule and return True when every rule passes."""
    results = json.loads(results_path.read_text())
    outcomes = evaluate(results, load_rules(thresholds_path))
    for outcome in outcomes:
        print(outcome.line())
    passed = all(o.passed for o in outcomes)
    print(f"{sum(o.passed for o in outcomes)}/{len(outcomes)} threshold rules passed")
    return passed
