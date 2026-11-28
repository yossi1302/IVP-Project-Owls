import pytest

from src.evaluation import (
    MAX_CIRCULAR_ERROR_MINUTES,
    accuracy_with_tolerance,
    circular_time_difference,
    evaluate_predictions,
    format_time,
    is_within_tolerance,
    minutes_to_time,
    parse_time,
    score_prediction,
    score_time_error,
    time_to_minutes,
    tolerance_accuracy,
)


class TestParseAndFormat:
    @pytest.mark.parametrize(
        ("time_str", "expected"),
        [
            ("01:00", (1, 0)),
            ("03:13", (3, 13)),
            ("11:59", (11, 59)),
            ("12:00", (12, 0)),
            ("12:59", (12, 59)),
        ],
    )
    def test_parse_valid_times(self, time_str, expected):
        assert parse_time(time_str) == expected

    @pytest.mark.parametrize(
        "bad_time",
        [
            "",
            "3:05",
            "03:5",
            "00:00",
            "00:30",
            "13:00",
            "23:59",
            "12:60",
            "12:00:00",
            "12-00",
            "ab:cd",
            " 12:00",
            "12:00 ",
            "１２:００",
            None,
            1200,
        ],
    )
    def test_parse_invalid_times(self, bad_time):
        with pytest.raises(ValueError):
            parse_time(bad_time)

    @pytest.mark.parametrize(
        ("hour", "minute", "expected"),
        [
            (1, 0, "01:00"),
            (9, 5, "09:05"),
            (12, 0, "12:00"),
            (12, 59, "12:59"),
        ],
    )
    def test_format_valid_times(self, hour, minute, expected):
        assert format_time(hour, minute) == expected

    @pytest.mark.parametrize(
        ("hour", "minute"),
        [
            (0, 0),
            (13, 0),
            (-1, 10),
            (5, -1),
            (5, 60),
            (True, 0),
            (1, False),
            (1.0, 0),
        ],
    )
    def test_format_invalid_times(self, hour, minute):
        with pytest.raises(ValueError):
            format_time(hour, minute)

    def test_round_trip_all_720_minutes(self):
        for total_minutes in range(720):
            formatted = minutes_to_time(total_minutes)
            assert time_to_minutes(formatted) == total_minutes

    def test_minutes_to_time_wrapping(self):
        assert minutes_to_time(0) == "12:00"
        assert minutes_to_time(720) == "12:00"
        assert minutes_to_time(-1) == "11:59"
        assert minutes_to_time(721) == "12:01"
        assert minutes_to_time(-60) == "11:00"


class TestCircularTimeDifference:
    @pytest.mark.parametrize(
        ("time_a", "time_b", "expected"),
        [
            ("12:00", "12:00", 0),
            ("06:30", "06:30", 0),
            ("12:59", "01:01", 2),
            ("01:01", "12:59", 2),
            ("11:59", "12:00", 1),
            ("11:59", "12:01", 2),
            ("12:01", "11:59", 2),
            ("11:55", "12:05", 10),
            ("11:54", "12:05", 11),
            ("11:45", "12:15", 30),
            ("11:44", "12:15", 31),
            ("11:30", "12:30", 60),
            ("11:00", "01:00", 120),
            ("12:00", "06:00", MAX_CIRCULAR_ERROR_MINUTES),
            ("03:15", "09:15", MAX_CIRCULAR_ERROR_MINUTES),
            ("12:00", "06:01", 359),
            ("12:00", "05:59", 359),
            ("01:00", "07:05", 355),
        ],
    )
    def test_circular_difference_cases(self, time_a, time_b, expected):
        assert circular_time_difference(time_a, time_b) == expected
        assert circular_time_difference(time_b, time_a) == expected


class TestCompetitionScoring:
    @pytest.mark.parametrize(
        ("error_minutes", "expected_score"),
        [
            (0, 1.0),
            (1, 1.0),
            (2, 1.0),
            (3, 0.8),
            (6, 0.8),
            (10, 0.8),
            (11, 0.5),
            (20, 0.5),
            (30, 0.5),
            (31, 0.0),
            (60, 0.0),
            (360, 0.0),
        ],
    )
    def test_score_time_error_boundaries(self, error_minutes, expected_score):
        assert score_time_error(error_minutes) == pytest.approx(expected_score)

    @pytest.mark.parametrize("bad_error", [-1, 361, True, 2.5])
    def test_score_time_error_invalid(self, bad_error):
        with pytest.raises(ValueError):
            score_time_error(bad_error)

    @pytest.mark.parametrize(
        ("predicted", "actual", "expected_score"),
        [
            ("03:15", "03:15", 1.0),
            ("12:59", "01:01", 1.0),
            ("11:59", "12:01", 1.0),
            ("11:58", "12:01", 0.8),
            ("11:55", "12:05", 0.8),
            ("12:05", "11:55", 0.8),
            ("11:54", "12:05", 0.5),
            ("11:45", "12:15", 0.5),
            ("12:15", "11:45", 0.5),
            ("11:44", "12:15", 0.0),
            ("12:00", "06:00", 0.0),
            (None, "12:00", 0.0),
        ],
    )
    def test_score_prediction_boundaries_and_noon_crossings(
        self, predicted, actual, expected_score
    ):
        assert score_prediction(predicted, actual) == pytest.approx(expected_score)

    def test_invalid_actual_rejected_even_when_predicted_is_none(self):
        with pytest.raises(ValueError):
            score_prediction(None, "00:00")
        with pytest.raises(ValueError):
            is_within_tolerance(None, "00:00", 2)

    def test_accuracy_with_tolerance_and_evaluate_predictions(self):
        targets = ["12:00", "01:01", "12:05", "12:15", "06:00", "09:00"]
        preds = ["11:59", "12:59", "11:55", "11:45", "06:31", None]
        expected_score = (1.0 + 1.0 + 0.8 + 0.5 + 0.0 + 0.0) / 6.0

        assert accuracy_with_tolerance(preds, targets) == pytest.approx(expected_score)
        assert tolerance_accuracy(preds, targets, 2) == pytest.approx(2 / 6)
        assert tolerance_accuracy(preds, targets, 10) == pytest.approx(3 / 6)
        assert tolerance_accuracy(preds, targets, 30) == pytest.approx(4 / 6)

        summary = evaluate_predictions(preds, targets)
        assert summary["total"] == 6
        assert summary["predicted"] == 5
        assert summary["failed"] == 1
        assert summary["accuracy_with_tolerance"] == pytest.approx(expected_score)
        assert summary["tier_counts"] == {
            "full_0_2": 2,
            "high_3_10": 1,
            "half_11_30": 1,
            "zero_31_plus": 1,
            "failed": 1,
        }
        assert summary["mae_predicted"] == pytest.approx((1 + 2 + 10 + 30 + 31) / 5)
        assert summary["median_predicted"] == pytest.approx(10.0)
        assert summary["accuracy_at_tolerance"] == {
            2: pytest.approx(2 / 6),
            10: pytest.approx(3 / 6),
            30: pytest.approx(4 / 6),
        }

    def test_is_within_tolerance_validation(self):
        assert is_within_tolerance("12:59", "01:01", 2) is True
        assert is_within_tolerance("12:59", "01:01", 1) is False
        assert is_within_tolerance(None, "12:00", 30) is False
        with pytest.raises(ValueError):
            is_within_tolerance("12:00", "12:00", -1)
        with pytest.raises(ValueError):
            is_within_tolerance("12:00", "12:00", True)

    def test_all_failed_predictions(self):
        summary = evaluate_predictions([None, None], ["12:00", "03:00"])
        assert summary["predicted"] == 0
        assert summary["failed"] == 2
        assert summary["accuracy_with_tolerance"] == pytest.approx(0.0)
        assert summary["mae_predicted"] is None
        assert summary["median_predicted"] is None

    def test_sequence_validation(self):
        with pytest.raises(ValueError):
            accuracy_with_tolerance(["12:00"], ["12:00", "01:00"])
        with pytest.raises(ValueError):
            accuracy_with_tolerance([], [])
        with pytest.raises(ValueError):
            tolerance_accuracy(["12:00"], ["12:00", "01:00"], 2)
        with pytest.raises(ValueError):
            tolerance_accuracy([], [], 2)
        with pytest.raises(ValueError):
            evaluate_predictions([], [])
