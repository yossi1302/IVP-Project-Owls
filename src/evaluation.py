from __future__ import annotations

from collections.abc import Sequence
from statistics import median

MINUTES_PER_HOUR = 60
HOURS_PER_CLOCK = 12
MINUTES_PER_12_HOURS = HOURS_PER_CLOCK * MINUTES_PER_HOUR
MAX_CIRCULAR_ERROR_MINUTES = MINUTES_PER_12_HOURS // 2

TOLERANCE_TIERS: tuple[tuple[int, float], ...] = (
    (2, 1.0),
    (10, 0.8),
    (30, 0.5),
)


def _validate_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return value


def parse_time(time_str: str) -> tuple[int, int]:
    if not isinstance(time_str, str):
        raise ValueError(f"Time must be a string in HH:MM format, got {time_str!r}")

    if len(time_str) != 5 or time_str[2] != ":":
        raise ValueError(f"Invalid time format (expected HH:MM): {time_str!r}")

    hour_str, minute_str = time_str[:2], time_str[3:]
    digits = hour_str + minute_str
    if not digits.isascii() or not digits.isdigit():
        raise ValueError(f"Time must contain ASCII digits (HH:MM): {time_str!r}")

    hour = int(hour_str)
    minute = int(minute_str)
    if not (1 <= hour <= 12 and 0 <= minute <= 59):
        raise ValueError(f"Time out of 12-hour range (01:00 to 12:59): {time_str!r}")

    return hour, minute


def format_time(hour: int, minute: int) -> str:
    hour = _validate_int(hour, "hour")
    minute = _validate_int(minute, "minute")

    if not (1 <= hour <= 12):
        raise ValueError(f"hour must be between 1 and 12, got {hour}")
    if not (0 <= minute <= 59):
        raise ValueError(f"minute must be between 0 and 59, got {minute}")

    return f"{hour:02d}:{minute:02d}"


def time_to_minutes(time_str: str) -> int:
    hour, minute = parse_time(time_str)
    return (hour % HOURS_PER_CLOCK) * MINUTES_PER_HOUR + minute


def minutes_to_time(total_minutes: int) -> str:
    total_minutes = _validate_int(total_minutes, "total_minutes")
    wrapped = total_minutes % MINUTES_PER_12_HOURS
    hour = wrapped // MINUTES_PER_HOUR
    minute = wrapped % MINUTES_PER_HOUR
    if hour == 0:
        hour = 12
    return format_time(hour, minute)


def circular_time_difference(time_a: str, time_b: str) -> int:
    minutes_a = time_to_minutes(time_a)
    minutes_b = time_to_minutes(time_b)
    raw_diff = abs(minutes_a - minutes_b)
    return min(raw_diff, MINUTES_PER_12_HOURS - raw_diff)


def score_time_error(error_minutes: int) -> float:
    error_minutes = _validate_int(error_minutes, "error_minutes")
    if not (0 <= error_minutes <= MAX_CIRCULAR_ERROR_MINUTES):
        raise ValueError(
            f"error_minutes must be between 0 and {MAX_CIRCULAR_ERROR_MINUTES}, "
            f"got {error_minutes}"
        )

    for max_error, points in TOLERANCE_TIERS:
        if error_minutes <= max_error:
            return points
    return 0.0


def score_prediction(predicted: str | None, actual: str) -> float:
    parse_time(actual)
    if predicted is None:
        return 0.0
    error = circular_time_difference(predicted, actual)
    return score_time_error(error)


def accuracy_with_tolerance(
    predictions: Sequence[str | None],
    targets: Sequence[str],
) -> float:
    if len(predictions) != len(targets):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(targets)} targets"
        )
    if len(targets) == 0:
        raise ValueError("Cannot compute accuracy_with_tolerance on empty sequences")

    total_score = sum(
        score_prediction(pred, truth) for pred, truth in zip(predictions, targets)
    )
    return total_score / len(targets)


def is_within_tolerance(
    predicted: str | None,
    actual: str,
    tolerance_minutes: int,
) -> bool:
    tolerance_minutes = _validate_int(tolerance_minutes, "tolerance_minutes")
    if tolerance_minutes < 0:
        raise ValueError(
            f"tolerance_minutes must be non-negative, got {tolerance_minutes}"
        )

    parse_time(actual)
    if predicted is None:
        return False

    return circular_time_difference(predicted, actual) <= tolerance_minutes


def tolerance_accuracy(
    predictions: Sequence[str | None],
    targets: Sequence[str],
    tolerance_minutes: int,
) -> float:
    if len(predictions) != len(targets):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(targets)} targets"
        )
    if len(targets) == 0:
        raise ValueError("Cannot compute tolerance_accuracy on empty sequences")

    hits = sum(
        1
        for pred, truth in zip(predictions, targets)
        if is_within_tolerance(pred, truth, tolerance_minutes)
    )
    return hits / len(targets)


def evaluate_predictions(
    predictions: Sequence[str | None],
    targets: Sequence[str],
    tolerances: Sequence[int] = (2, 10, 30),
) -> dict[str, object]:
    if len(predictions) != len(targets):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(targets)} targets"
        )
    if len(targets) == 0:
        raise ValueError("Cannot evaluate empty sequences")

    errors: list[int] = []
    scores: list[float] = []
    tier_counts = {
        "full_0_2": 0,
        "high_3_10": 0,
        "half_11_30": 0,
        "zero_31_plus": 0,
        "failed": 0,
    }

    for pred, truth in zip(predictions, targets):
        parse_time(truth)
        if pred is None:
            tier_counts["failed"] += 1
            scores.append(0.0)
            continue

        err = circular_time_difference(pred, truth)
        errors.append(err)
        pts = score_time_error(err)
        scores.append(pts)

        if err <= 2:
            tier_counts["full_0_2"] += 1
        elif err <= 10:
            tier_counts["high_3_10"] += 1
        elif err <= 30:
            tier_counts["half_11_30"] += 1
        else:
            tier_counts["zero_31_plus"] += 1

    accuracy_by_tol = {
        tol: tolerance_accuracy(predictions, targets, tol) for tol in tolerances
    }

    return {
        "total": len(targets),
        "predicted": len(errors),
        "failed": tier_counts["failed"],
        "accuracy_with_tolerance": sum(scores) / len(scores),
        "tier_counts": tier_counts,
        "mae_predicted": (sum(errors) / len(errors)) if errors else None,
        "median_predicted": median(errors) if errors else None,
        "accuracy_at_tolerance": accuracy_by_tol,
    }
