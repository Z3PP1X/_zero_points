"""Scheduling helpers for curated holdout evaluation during supervised training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CuratedEvalSchedule:
    """When to run curated holdout evaluation during training."""

    period: int = 5
    on_test_highscore: bool = True
    warmup: int = 0

    def __post_init__(self) -> None:
        if self.period < 0:
            raise ValueError(f"curated_eval_period must be >= 0, got {self.period}")
        if self.warmup < 0:
            raise ValueError(f"curated_eval_warmup must be >= 0, got {self.warmup}")


def plain_train_mapping(train_cfg: Any) -> dict[str, Any]:
    if train_cfg is None:
        return {}
    if isinstance(train_cfg, dict):
        return train_cfg
    if hasattr(train_cfg, "items"):
        return {str(key): value for key, value in train_cfg.items()}
    return {}


def parse_curated_eval_schedule(train_cfg: Any) -> CuratedEvalSchedule:
    """Read curated-eval settings from a train YAML / GraphGym cfg section."""
    mapping = plain_train_mapping(train_cfg)
    period = int(mapping.get("curated_eval_period", 5))
    on_test_highscore = bool(mapping.get("curated_eval_on_test_highscore", True))
    warmup = int(mapping.get("curated_eval_warmup", 0))
    return CuratedEvalSchedule(
        period=period, on_test_highscore=on_test_highscore, warmup=warmup
    )


def should_evaluate_curated(
    epoch: int,
    schedule: CuratedEvalSchedule,
    *,
    is_new_test_highscore: bool = False,
) -> tuple[bool, str | None]:
    """
    Decide whether curated holdout evaluation should run this epoch.

    ``epoch`` is 0-based (training loop index). Periodic checks use 1-based
    epoch numbers (5th, 10th, …).
    """
    # Warmup gate: suppress curated holdout eval for the first ``warmup`` epochs.
    # Early in training an underfit model on the out-of-distribution curated set can
    # produce anomalous — even inverted — metrics that would mislead diagnostics.
    # Mirrors _WarmupAwareEarlyStopping's ``current_epoch < warmup_epochs`` (0-based).
    if epoch < schedule.warmup:
        return False, None

    reasons: list[str] = []
    epoch_number = epoch + 1

    if schedule.period > 0 and epoch_number % schedule.period == 0:
        reasons.append(f"period={schedule.period}")

    if schedule.on_test_highscore and is_new_test_highscore:
        reasons.append("test_highscore")

    if not reasons:
        return False, None
    return True, "+".join(reasons)
