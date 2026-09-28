"""Explicit review capabilities and immutable requests; never infer from prose."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from pathlib import Path

STAGES = ("adoption", "source_compile", "source_rollout", "learning", "selection", "export",
          "target_validation", "gpt_review", "release")


class Rejection(ValueError):
    """An entry is not justified by its current, complete evidence."""


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise Rejection(f"{name} must be an integer >= {minimum} (booleans are invalid)")


def _seconds(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise Rejection(f"{name} must be finite and positive")


@dataclass(frozen=True)
class LearningRequest:
    iterations: int
    wall_seconds: float
    seed: int
    gpus: int
    budget_ledger_path: str

    def __post_init__(self):
        for name in ("iterations", "seed", "gpus"):
            _integer(getattr(self, name), name)
        _seconds(self.wall_seconds, "wall_seconds")
        if not 1000000 <= self.seed < 2000000:
            raise Rejection("Training seed must belong to the training partition")
        if not isinstance(self.budget_ledger_path, str) or not Path(self.budget_ledger_path).is_absolute():
            raise Rejection("Learning request requires an absolute budget ledger path")
        object.__setattr__(self, "budget_ledger_path", str(Path(self.budget_ledger_path).resolve()))

    def record(self):
        return {"iterations": self.iterations, "wall_seconds": self.wall_seconds,
                "seed": self.seed, "gpus": self.gpus, "budget_ledger_path": self.budget_ledger_path}


def authorize(review: dict, stage: str, request: LearningRequest | None = None) -> None:
    """Historical v1 records remain readable but grant no new capability."""
    if review.get("schema") != "microduck_root_review_v2":
        raise Rejection("Explicit v2 review authorization required; historical approval grants no new action")
    identifier = review.get("authorization_id")
    try:
        if not isinstance(identifier, str) or str(uuid.UUID(identifier)) != identifier:
            raise ValueError("noncanonical UUID")
    except (ValueError, AttributeError):
        raise Rejection("Review requires a canonical immutable authorization UUID") from None
    stages = review.get("authorized_stages")
    if (not isinstance(stages, list) or not stages or
            any(not isinstance(item, str) or item not in STAGES for item in stages) or
            len(stages) != len(set(stages))):
        raise Rejection("Review authorized stages are missing, duplicated or unknown")
    if type(review.get("learning_allowed")) is not bool:
        raise Rejection("Review learning_allowed must be an explicit boolean")
    if stage not in stages:
        raise Rejection(f"Review does not authorize stage {stage}")
    learning = "learning" in stages
    if learning != review["learning_allowed"]:
        raise Rejection("Learning stage and learning_allowed must agree")
    if not learning:
        if review.get("learning_limit", "missing") is not None:
            raise Rejection("Non-learning review must explicitly have learning_limit=null")
        if request is not None:
            raise Rejection("Learning request provided for a non-learning stage")
        return
    limit = review.get("learning_limit")
    required = {"iterations", "max_wall_seconds", "seed", "gpus", "max_runs", "budget_ledger_path"}
    if not isinstance(limit, dict) or set(limit) != required:
        raise Rejection("Learning review requires a complete exact limit")
    for name in ("iterations", "seed", "gpus", "max_runs"):
        _integer(limit[name], f"learning_limit.{name}")
    _seconds(limit["max_wall_seconds"], "learning_limit.max_wall_seconds")
    ledger = limit["budget_ledger_path"]
    if not isinstance(ledger, str) or not Path(ledger).is_absolute():
        raise Rejection("Review budget ledger path must be absolute")
    if not 1000000 <= limit["seed"] < 2000000:
        raise Rejection("Review seed is outside the training partition")
    if stage != "learning":
        if request is not None:
            raise Rejection("Learning request provided for a non-learning stage")
        return
    if not isinstance(request, LearningRequest):
        raise Rejection("Learning admission requires the complete actual immutable request")
    if (request.iterations > limit["iterations"] or request.wall_seconds > limit["max_wall_seconds"] or
            request.seed != limit["seed"] or request.gpus != limit["gpus"] or
            request.budget_ledger_path != str(Path(ledger).resolve())):
        raise Rejection("Actual learning request exceeds or differs from review limits")
