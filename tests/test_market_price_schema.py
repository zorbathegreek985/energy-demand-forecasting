from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

from market_price_forecasting.schema import (
    MarketPriceObservation,
    validate_observations,
)


SYNTHETIC_DELIVERY_DAY = date(2025, 1, 15)


def synthetic_observation(block_number: int = 0) -> MarketPriceObservation:
    """A made-up fixture row; this is not IEX data."""
    block_start = datetime.combine(SYNTHETIC_DELIVERY_DAY, time.min) + timedelta(
        minutes=15 * block_number
    )
    return MarketPriceObservation(
        exchange="SYNTHETIC-EXCHANGE",
        product="SYNTHETIC-DAM",
        delivery_date=SYNTHETIC_DELIVERY_DAY,
        delivery_block_start=block_start,
        delivery_block_end=block_start + timedelta(minutes=15),
        price_definition="synthetic example price; not a confirmed market field",
        price_value=123.45 + block_number,
        price_unit="SYNTHETIC-UNIT",
        delivery_area_or_price_scope="SYNTHETIC-AREA",
        retrieval_timestamp=datetime(2025, 1, 16, 12, 0),
        source_provenance="synthetic-test-fixture-only",
        source_timezone=None,
    )


def synthetic_delivery_day(
    omit_block: int | None = None,
    block_count: int = 96,
) -> list[MarketPriceObservation]:
    return [
        synthetic_observation(block_number)
        for block_number in range(block_count)
        if block_number != omit_block
    ]


def synthetic_expected_intervals() -> list[tuple[datetime, datetime]]:
    """Explicit test-only wall-clock schedule; it is not an IEX calendar."""
    day_start = datetime.combine(SYNTHETIC_DELIVERY_DAY, time.min)
    return [
        (
            day_start + timedelta(minutes=15 * block_number),
            day_start + timedelta(minutes=15 * (block_number + 1)),
        )
        for block_number in range(96)
    ]


def test_valid_synthetic_observation_and_day_are_reported_without_timezone_guess() -> None:
    observations = synthetic_delivery_day()

    report = validate_observations(
        observations,
        delivery_day=SYNTHETIC_DELIVERY_DAY,
        expected_interval_minutes=15,
        expected_intervals=synthetic_expected_intervals(),
    )

    assert report.is_valid
    assert report.delivery_day_complete is True
    assert not report.missing_intervals
    assert any(
        "source timezone metadata is unresolved" in warning.lower()
        for warning in report.warnings
    )
    assert any("not localized or converted" in warning for warning in report.warnings)


def test_explicit_non_96_schedule_is_checked_as_supplied() -> None:
    observations = synthetic_delivery_day(block_count=92)
    expected_intervals = synthetic_expected_intervals()[:92]

    report = validate_observations(
        observations,
        delivery_day=SYNTHETIC_DELIVERY_DAY,
        expected_interval_minutes=15,
        expected_intervals=expected_intervals,
    )

    assert report.is_valid
    assert report.delivery_day_complete is True
    assert len(observations) == len(expected_intervals) == 92


def test_delivery_day_is_not_assumed_complete_without_explicit_schedule() -> None:
    report = validate_observations(
        synthetic_delivery_day(),
        delivery_day=SYNTHETIC_DELIVERY_DAY,
    )

    assert report.delivery_day_complete is None
    assert not report.missing_intervals
    assert any("supply an explicit expected_intervals" in warning for warning in report.warnings)


def test_empty_or_overlapping_expected_schedule_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_observations(
            [],
            delivery_day=SYNTHETIC_DELIVERY_DAY,
            expected_intervals=[],
        )

    with pytest.raises(ValueError, match="must not overlap"):
        validate_observations(
            [],
            delivery_day=SYNTHETIC_DELIVERY_DAY,
            expected_intervals=[
                ("2025-01-15T00:00:00", "2025-01-15T00:15:00"),
                ("2025-01-15T00:10:00", "2025-01-15T00:25:00"),
            ],
        )


def test_missing_required_canonical_field_is_reported() -> None:
    record = synthetic_observation()
    incomplete = deepcopy(record.__dict__)
    del incomplete["price_unit"]

    report = validate_observations([incomplete])

    assert not report.is_valid
    assert any("missing required field 'price_unit'" in error for error in report.row_errors)


@pytest.mark.parametrize("invalid_price", ["not-a-price", float("inf"), float("nan")])
def test_nonnumeric_or_nonfinite_prices_are_rejected(invalid_price) -> None:
    record = synthetic_observation()
    invalid_record = deepcopy(record.__dict__)
    invalid_record["price_value"] = invalid_price

    report = validate_observations([invalid_record])

    assert not report.is_valid
    assert any("numeric and finite" in error for error in report.row_errors)


@pytest.mark.parametrize("invalid_price", [True, Decimal("NaN"), Decimal("Infinity")])
def test_boolean_and_nonfinite_decimal_prices_are_rejected(invalid_price) -> None:
    record = deepcopy(synthetic_observation().__dict__)
    record["price_value"] = invalid_price

    report = validate_observations([record])

    assert not report.is_valid
    assert any("numeric and finite" in error for error in report.row_errors)


def test_exact_duplicate_is_reported_without_deduplication() -> None:
    record = synthetic_observation()
    observations = [record, deepcopy(record)]

    report = validate_observations(observations)

    assert len(observations) == 2
    assert len(report.duplicate_keys) == 1
    assert len(report.exact_duplicates) == 1
    assert not report.conflicting_duplicates
    assert any("duplicate observation key" in error for error in report.row_errors)


def test_duplicate_price_and_metadata_conflicts_are_distinguished() -> None:
    base = deepcopy(synthetic_observation().__dict__)
    price_conflict = deepcopy(base)
    price_conflict["price_value"] = 999.0
    metadata_conflict = deepcopy(base)
    metadata_conflict["revision_information"] = "synthetic second revision"

    report = validate_observations([base, price_conflict, metadata_conflict])

    assert not report.exact_duplicates
    assert len(report.conflicting_duplicates) == 2
    assert report.conflicting_duplicates[0].differing_fields == ("price_value",)
    assert report.conflicting_duplicates[1].differing_fields == ("revision_information",)


def test_missing_15_minute_interval_is_reported_not_filled() -> None:
    observations = synthetic_delivery_day(omit_block=37)
    original_count = len(observations)

    report = validate_observations(
        observations,
        delivery_day=SYNTHETIC_DELIVERY_DAY,
        expected_interval_minutes=15,
        expected_intervals=synthetic_expected_intervals(),
    )

    assert len(observations) == original_count == 95
    assert report.delivery_day_complete is False
    assert report.missing_intervals == (
        "2025-01-15T09:15:00 to 2025-01-15T09:30:00",
    )


def test_nonpositive_and_unexpected_interval_durations_are_rejected() -> None:
    record = deepcopy(synthetic_observation().__dict__)
    record["delivery_block_end"] = record["delivery_block_start"]
    nonpositive = validate_observations([record], expected_interval_minutes=15)
    assert any("must follow its start" in error for error in nonpositive.row_errors)

    short_block = deepcopy(synthetic_observation().__dict__)
    short_block["delivery_block_end"] = short_block["delivery_block_start"] + timedelta(
        minutes=10
    )
    wrong_duration = validate_observations([short_block], expected_interval_minutes=15)
    assert any("duration does not match" in error for error in wrong_duration.row_errors)

    reversed_block = deepcopy(synthetic_observation().__dict__)
    reversed_block["delivery_block_start"] = datetime(2025, 1, 15, 0, 15)
    reversed_block["delivery_block_end"] = datetime(2025, 1, 15, 0, 0)
    reversed_report = validate_observations([reversed_block])
    assert any("must follow its start" in error for error in reversed_report.row_errors)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("delivery_block_start", None),
        ("delivery_block_end", "not-a-timestamp"),
    ],
)
def test_missing_or_malformed_block_endpoints_are_reported(field, value) -> None:
    record = deepcopy(synthetic_observation().__dict__)
    record[field] = value

    report = validate_observations([record])

    assert not report.is_valid
    assert any(field in error for error in report.row_errors)


def test_overlapping_intervals_are_reported_separately_from_gaps() -> None:
    first = deepcopy(synthetic_observation().__dict__)
    second = deepcopy(synthetic_observation(1).__dict__)
    second["delivery_block_start"] = datetime(2025, 1, 15, 0, 10)
    second["delivery_block_end"] = datetime(2025, 1, 15, 0, 25)

    report = validate_observations(
        [first, second],
        expected_interval_minutes=15,
    )

    assert len(report.overlapping_intervals) == 1
    overlap = report.overlapping_intervals[0]
    assert (overlap.first_row_index, overlap.second_row_index) == (0, 1)
    assert overlap.overlap_start_wall_time == datetime(2025, 1, 15, 0, 10)
    assert overlap.overlap_end_wall_time == datetime(2025, 1, 15, 0, 15)
    assert not report.missing_intervals


def test_overlap_detection_respects_delivery_area_scope() -> None:
    first = deepcopy(synthetic_observation().__dict__)
    other_scope = deepcopy(synthetic_observation(1).__dict__)
    other_scope["delivery_block_start"] = datetime(2025, 1, 15, 0, 10)
    other_scope["delivery_block_end"] = datetime(2025, 1, 15, 0, 25)
    other_scope["delivery_area_or_price_scope"] = "SYNTHETIC-OTHER-AREA"

    report = validate_observations([first, other_scope], expected_interval_minutes=15)

    assert not report.overlapping_intervals


def test_timezone_offsets_are_not_converted_or_compared_as_local_time() -> None:
    first = deepcopy(synthetic_observation().__dict__)
    first["delivery_block_start"] = datetime(
        2025, 1, 15, 0, 0, tzinfo=timezone.utc
    )
    first["delivery_block_end"] = datetime(
        2025, 1, 15, 0, 15, tzinfo=timezone.utc
    )
    second = deepcopy(first)
    second["delivery_block_start"] = datetime(
        2025, 1, 15, 0, 10, tzinfo=timezone(timedelta(hours=1))
    )
    second["delivery_block_end"] = datetime(
        2025, 1, 15, 0, 25, tzinfo=timezone(timedelta(hours=1))
    )

    report = validate_observations([first, second], expected_interval_minutes=15)

    assert not report.overlapping_intervals
    assert any("offsets differ" in warning for warning in report.warnings)


def test_iso_timestamp_strings_are_parsed_without_timezone_conversion() -> None:
    record = deepcopy(synthetic_observation().__dict__)
    record["delivery_block_start"] = "2025-01-15T00:00:00"
    record["delivery_block_end"] = "2025-01-15T00:15:00"
    record["retrieval_timestamp"] = "2025-01-16T12:00:00"

    report = validate_observations([record])

    assert report.is_valid
    assert any("not localized or converted" in warning for warning in report.warnings)


def test_invalid_delivery_timestamp_is_reported() -> None:
    record = deepcopy(synthetic_observation().__dict__)
    record["delivery_block_start"] = "not-a-timestamp"

    report = validate_observations([record])

    assert not report.is_valid
    assert any("delivery_block_start" in error for error in report.row_errors)


def test_datetime_object_is_not_accepted_as_a_delivery_date() -> None:
    record = deepcopy(synthetic_observation().__dict__)
    record["delivery_date"] = datetime(2025, 1, 15)

    report = validate_observations([record])

    assert not report.is_valid
    assert any("delivery_date" in error for error in report.row_errors)


def test_date_only_string_is_not_accepted_as_a_block_timestamp() -> None:
    record = deepcopy(synthetic_observation().__dict__)
    record["delivery_block_start"] = "2025-01-15"

    report = validate_observations([record])

    assert not report.is_valid
    assert any("delivery_block_start" in error for error in report.row_errors)


def test_mixed_naive_and_aware_block_endpoints_are_rejected() -> None:
    record = deepcopy(synthetic_observation().__dict__)
    record["delivery_block_end"] = datetime(
        2025, 1, 15, 0, 15, tzinfo=timezone.utc
    )

    report = validate_observations([record])

    assert not report.is_valid
    assert any("both include an offset" in error for error in report.row_errors)