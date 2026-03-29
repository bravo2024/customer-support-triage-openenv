"""Deterministic task graders (0.0-1.0) for easy/medium/hard tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class EasySample:
    predicted: str
    truth: str


def grade_easy(samples: List[EasySample]) -> float:
    if not samples:
        return 0.0
    correct = sum(1 for s in samples if s.predicted == s.truth)
    return max(0.0, min(1.0, correct / len(samples)))


def grade_medium(response: str, required_keywords: List[str]) -> float:
    """Keyword-coverage based deterministic grader."""
    if not required_keywords:
        return 1.0
    text = (response or "").lower()
    hits = sum(1 for k in required_keywords if k.lower() in text)
    return max(0.0, min(1.0, hits / len(required_keywords)))


def grade_hard(resolution_rate: float, sla_compliance: float, satisfaction: float) -> float:
    """Weighted deterministic triage score."""
    # weights sum to 1.0
    score = 0.4 * resolution_rate + 0.35 * sla_compliance + 0.25 * satisfaction
    return max(0.0, min(1.0, score))
