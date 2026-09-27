"""Canonical provisional market-price schema and non-mutating validation.

This module defines project-owned canonical field names, not confirmed IEX
export headers. It does not localize, convert, deduplicate, fill, or otherwise
modify observations. Timestamp strings must use ISO 8601 when passed directly.
"""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from math import isfinite
from numbers import Real
from typing import Any


REQUIRED_CANONICAL_FIELDS = (
    "exchange",
    "product",
    "delivery_date",
    "delivery_block_start",
    "delivery_block_end",
    "price_definition",
    "price_value",
    "price_unit",
    "delivery_area_or_price_scope",
    "retrieval_timestamp",
    "source_provenance",
)

OPTIONAL_SOURCE_METADATA_FIELDS = (
    "publication_timestamp",
    "revision_information",
    "source_timezone",
)

UNCONFIRMED_IEX_METADATA = (
    "source field names and mapping",
    "price definition",
    "price unit",
    "delivery area or price scope",
    "source timezone",
    "publication timestamp availability",
    "revision semantics",
)


@dataclass(frozen=True)
class MarketPriceObservation:
    """One canonical interval observation; values require source confirmation.

    ``delivery_date`` is a Python date or ISO ``YYYY-MM-DD`` string. Block
    endpoints and timestamps are datetimes or ISO-8601 strings. Naive delivery
    timestamps are preserved as source wall-clock values; no timezone is
    inferred. Price/unit/scope fields have no defaults and must be supplied by
    the caller rather than assumed.
    """

    exchange: str
    product: str
    delivery_date: date | str
    delivery_block_start: datetime | str
    delivery_block_end: datetime | str
    price_definition: str
    price_value: int | float | Decimal
    price_unit: str
    delivery_area_or_price_scope: str
    retrieval_timestamp: datetime | str
    source_provenance: str
    publication_timestamp: datetime | str | None = None
    revision_information: str | None = None
    source_timezone: str | None = None


@dataclass(frozen=True)
class DuplicateFinding:
    """One repeated logical key compared with its first occurrence."""

    key: tuple[Any, ...]
    first_row_index: int
    duplicate_row_index: int
    classification: str
    differing_fields: tuple[str, ...]


@dataclass(frozen=True)
class IntervalOverlap:
    """Two nonidentical intervals that overlap within the same market scope."""

    delivery_scope: tuple[Any, ...]
    first_row_index: int
    second_row_index: int
    overlap_start_wall_time: datetime
    overlap_end_wall_time: datetime


@dataclass(frozen=True)
class ValidationReport:
    """Validation findings; input observations are never changed or removed."""

    row_errors: tuple[str, ...]
    duplicate_findings: tuple[DuplicateFinding, ...]
    overlapping_intervals: tuple[IntervalOverlap, ...]
    missing_intervals: tuple[str, ...]
    warnings: tuple[str, ...]
    delivery_day_complete: bool | None

    @property
    def duplicate_keys(self) -> tuple[tuple[Any, ...], ...]:
        """Compatibility view of all exact and conflicting duplicate keys."""
        return tuple(finding.key for finding in self.duplicate_findings)

    @property
    def exact_duplicates(self) -> tuple[DuplicateFinding, ...]:
        return tuple(
            finding
            for finding in self.duplicate_findings
            if finding.classification == "exact"
        )

    @property
    def conflicting_duplicates(self) -> tuple[DuplicateFinding, ...]:
        return tuple(
            finding
            for finding in self.duplicate_findings
            if finding.classification == "conflicting"
        )

    @property
    def is_valid(self) -> bool:
        """Whether no schema, duplicate, or overlap errors were found."""
        return (
            not self.row_errors
            and not self.duplicate_findings
            and not self.overlapping_intervals
        )


def _as_mapping(record: object) -> Mapping[str, Any] | None:
    if isinstance(record, MarketPriceObservation):
        return asdict(record)
    if isinstance(record, Mapping):
        return record
    return None


def _parse_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.isoformat() == value else None
    return None


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        if "T" not in value and " " not in value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _numeric_finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        return False
    try:
        return isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _wall_clock(value: datetime) -> datetime:
    """Return literal datetime fields without timezone conversion."""
    return value.replace(tzinfo=None)


def _timezone_signature(value: datetime) -> str | None:
    """Describe the timestamp's explicit offset without consulting local time."""
    if value.tzinfo is None:
        return None
    offset = value.strftime("%z")
    return offset or str(value.tzinfo)


def _interval_key(start: datetime, end: datetime) -> tuple[str, str]:
    """Keep literal offsets in keys; equivalent instants are not normalized."""
    return start.isoformat(), end.isoformat()


def _stable_value(value: Any) -> Any:
    """Normalize common scalar types for exact-record comparison only."""
    if isinstance(value, datetime):
        return ("datetime", value.isoformat())
    if isinstance(value, date):
        return ("date", value.isoformat())
    if isinstance(value, Decimal):
        return ("decimal", str(value.normalize()))
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, Real):
        return ("number", float(value))
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _stable_value(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_stable_value(item) for item in value)
    return value


def _differing_fields(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> tuple[str, ...]:
    field_names = set(first) | set(second)
    return tuple(
        sorted(
            field
            for field in field_names
            if _stable_value(first.get(field)) != _stable_value(second.get(field))
        )
    )


def _observation_key(
    record: Mapping[str, Any],
    parsed_date: date,
    parsed_start: datetime,
    parsed_end: datetime,
) -> tuple[Any, ...]:
    """Key one market interval; excludes price and revision to expose conflicts.

    Two records for the same exchange/product/date/block/price-scope/price
    definition are reported as duplicate candidates even if their prices or
    revision labels differ. The validator does not choose a winning revision.
    Timestamp ISO strings preserve any explicit UTC offset and are not converted.
    """
    return (
        str(record["exchange"]).strip().casefold(),
        str(record["product"]).strip().casefold(),
        parsed_date,
        parsed_start.isoformat(),
        parsed_end.isoformat(),
        str(record["delivery_area_or_price_scope"]).strip().casefold(),
        str(record["price_definition"]).strip().casefold(),
    )


def validate_observations(
    observations: Iterable[MarketPriceObservation | Mapping[str, Any]],
    *,
    delivery_day: date | str | None = None,
    expected_interval_minutes: int | None = None,
    expected_intervals: Iterable[tuple[datetime | str, datetime | str]] | None = None,
) -> ValidationReport:
    """Validate records without mutation, timezone conversion, or implicit fill.

    ``expected_interval_minutes`` optionally validates every row's duration.
    Completeness is checked only when the caller supplies both ``delivery_day``
    and an explicit iterable of expected ``(start, end)`` timestamps. No 96-block
    assumption is made. All timestamp comparisons preserve their literal
    wall-clock values; differing explicit offsets are not converted or equated.
    """
    if expected_interval_minutes is not None and (
        isinstance(expected_interval_minutes, bool)
        or not isinstance(expected_interval_minutes, int)
        or expected_interval_minutes <= 0
    ):
        raise ValueError("expected_interval_minutes must be a positive integer or None.")

    parsed_delivery_day: date | None = None
    if delivery_day is not None:
        parsed_delivery_day = _parse_date(delivery_day)
        if parsed_delivery_day is None:
            raise ValueError("delivery_day must be a date or ISO YYYY-MM-DD string.")
    if expected_intervals is not None and parsed_delivery_day is None:
        raise ValueError("delivery_day is required when expected_intervals are supplied.")

    expected_by_key: dict[tuple[str, str], tuple[datetime, datetime]] | None = None
    if expected_intervals is not None:
        schedule = list(expected_intervals)
        if not schedule:
            raise ValueError("expected_intervals must not be empty.")
        expected_by_key = {}
        for index, interval_pair in enumerate(schedule):
            try:
                expected_start_value, expected_end_value = interval_pair
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"expected_intervals row {index} must contain a start/end pair."
                ) from error
            expected_start = _parse_datetime(expected_start_value)
            expected_end = _parse_datetime(expected_end_value)
            if expected_start is None or expected_end is None:
                raise ValueError(
                    f"expected_intervals row {index} must contain datetimes or ISO-8601 strings."
                )
            expected_start_wall = _wall_clock(expected_start)
            expected_end_wall = _wall_clock(expected_end)
            if expected_start_wall.date() != parsed_delivery_day:
                raise ValueError(
                    f"expected_intervals row {index} does not start on delivery_day."
                )
            if expected_end_wall <= expected_start_wall:
                raise ValueError(f"expected_intervals row {index} has a nonpositive duration.")
            if (expected_start.tzinfo is None) != (expected_end.tzinfo is None):
                raise ValueError(
                    f"expected_intervals row {index} has inconsistent timezone awareness."
                )
            if expected_interval_minutes is not None:
                duration = expected_end_wall - expected_start_wall
                if duration != timedelta(minutes=expected_interval_minutes):
                    raise ValueError(
                        f"expected_intervals row {index} does not match "
                        "expected_interval_minutes."
                    )
            key = _interval_key(expected_start, expected_end)
            if key in expected_by_key:
                raise ValueError(f"expected_intervals contains duplicate interval {key!r}.")
            for prior_start, prior_end in expected_by_key.values():
                if _timezone_signature(prior_start) != _timezone_signature(expected_start):
                    continue
                prior_start_wall = _wall_clock(prior_start)
                prior_end_wall = _wall_clock(prior_end)
                if (
                    expected_start_wall < prior_end_wall
                    and prior_start_wall < expected_end_wall
                ):
                    raise ValueError("expected_intervals must not overlap.")
            expected_by_key[key] = (expected_start, expected_end)

    records = list(observations)
    errors: list[str] = []
    warnings = [
        "IEX source field mapping, price definition/unit/area, timezone, publication "
        "timestamp availability, and revision semantics remain unconfirmed; source "
        "timezone is not assumed."
    ]
    duplicate_findings: list[DuplicateFinding] = []
    first_by_key: dict[tuple[Any, ...], tuple[int, Mapping[str, Any]]] = {}
    present_intervals: set[tuple[str, str]] = set()
    scope_intervals: dict[
        tuple[Any, ...], list[tuple[int, datetime, datetime, str | None]]
    ] = {}
    rows_without_source_timezone: list[int] = []

    for index, observation in enumerate(records):
        record = _as_mapping(observation)
        if record is None:
            errors.append(f"row {index}: expected a mapping or MarketPriceObservation.")
            continue

        missing_fields = [
            field for field in REQUIRED_CANONICAL_FIELDS if field not in record
        ]
        for field in missing_fields:
            errors.append(f"row {index}: missing required field '{field}'.")
        if missing_fields:
            continue

        for field in (
            "exchange",
            "product",
            "price_definition",
            "price_unit",
            "delivery_area_or_price_scope",
            "source_provenance",
        ):
            if not _nonempty_text(record[field]):
                errors.append(f"row {index}: '{field}' must be nonempty text.")

        parsed_date = _parse_date(record["delivery_date"])
        if parsed_date is None:
            errors.append(
                f"row {index}: 'delivery_date' must be a date or ISO YYYY-MM-DD string."
            )

        parsed_start = _parse_datetime(record["delivery_block_start"])
        parsed_end = _parse_datetime(record["delivery_block_end"])
        valid_interval = parsed_start is not None and parsed_end is not None
        if parsed_start is None:
            errors.append(
                f"row {index}: 'delivery_block_start' must be a datetime or ISO-8601 string."
            )
        if parsed_end is None:
            errors.append(
                f"row {index}: 'delivery_block_end' must be a datetime or ISO-8601 string."
            )
        if parsed_start is not None and parsed_end is not None:
            if (parsed_start.tzinfo is None) != (parsed_end.tzinfo is None):
                valid_interval = False
                errors.append(
                    f"row {index}: block endpoints must both include an offset or both be naive."
                )
            wall_start = _wall_clock(parsed_start)
            wall_end = _wall_clock(parsed_end)
            if wall_end <= wall_start:
                valid_interval = False
                errors.append(f"row {index}: delivery block end must follow its start.")
            if parsed_date is not None and wall_start.date() != parsed_date:
                valid_interval = False
                errors.append(
                    f"row {index}: block start date does not match delivery_date."
                )
            if (
                valid_interval
                and expected_interval_minutes is not None
                and wall_end - wall_start
                != timedelta(minutes=expected_interval_minutes)
            ):
                valid_interval = False
                errors.append(
                    f"row {index}: delivery block duration does not match "
                    f"expected_interval_minutes={expected_interval_minutes}."
                )
            if record.get("source_timezone") is None:
                rows_without_source_timezone.append(index)

        if not _numeric_finite(record["price_value"]):
            errors.append(f"row {index}: 'price_value' must be numeric and finite.")

        retrieval_timestamp = _parse_datetime(record["retrieval_timestamp"])
        if retrieval_timestamp is None:
            errors.append(
                f"row {index}: 'retrieval_timestamp' must be a datetime or ISO-8601 string."
            )

        if record.get("publication_timestamp") is not None and _parse_datetime(
            record["publication_timestamp"]
        ) is None:
            errors.append(
                f"row {index}: 'publication_timestamp' must be null or a datetime/ISO-8601 string."
            )
        for field in ("revision_information", "source_timezone"):
            value = record.get(field)
            if value is not None and not _nonempty_text(value):
                errors.append(f"row {index}: '{field}' must be null or nonempty text.")

        key_fields_valid = (
            parsed_date is not None
            and parsed_start is not None
            and parsed_end is not None
            and valid_interval
            and all(
                _nonempty_text(record[field])
                for field in (
                    "exchange",
                    "product",
                    "price_definition",
                    "delivery_area_or_price_scope",
                )
            )
        )
        if key_fields_valid:
            key = _observation_key(record, parsed_date, parsed_start, parsed_end)
            if key in first_by_key:
                first_index, first_record = first_by_key[key]
                differing = _differing_fields(first_record, record)
                classification = "exact" if not differing else "conflicting"
                duplicate_findings.append(
                    DuplicateFinding(
                        key=key,
                        first_row_index=first_index,
                        duplicate_row_index=index,
                        classification=classification,
                        differing_fields=differing,
                    )
                )
                errors.append(
                    f"row {index}: {classification} duplicate observation key {key!r}."
                )
            else:
                first_by_key[key] = (index, record)

        if valid_interval and parsed_date is not None and parsed_start and parsed_end:
            if all(
                _nonempty_text(record[field])
                for field in (
                    "exchange",
                    "product",
                    "price_definition",
                    "delivery_area_or_price_scope",
                )
            ):
                scope = (
                    str(record["exchange"]).strip().casefold(),
                    str(record["product"]).strip().casefold(),
                    parsed_date,
                    str(record["delivery_area_or_price_scope"]).strip().casefold(),
                    str(record["price_definition"]).strip().casefold(),
                )
                scope_intervals.setdefault(scope, []).append(
                    (
                        index,
                        _wall_clock(parsed_start),
                        _wall_clock(parsed_end),
                        _timezone_signature(parsed_start),
                    )
                )

        if (
            expected_by_key is not None
            and valid_interval
            and parsed_date == parsed_delivery_day
            and parsed_start is not None
            and parsed_end is not None
        ):
            interval_key = _interval_key(parsed_start, parsed_end)
            if interval_key in expected_by_key:
                present_intervals.add(interval_key)
            else:
                errors.append(
                    f"row {index}: interval is not present in the explicit expected_intervals set."
                )

    overlaps: list[IntervalOverlap] = []
    for scope, intervals in scope_intervals.items():
        timezone_signatures = {entry[3] for entry in intervals}
        if len(timezone_signatures) > 1:
            warnings.append(
                "Overlap comparison was skipped for delivery scope "
                f"{scope!r}: timestamp offsets differ and timezone semantics are unconfirmed."
            )
            continue
        ordered = sorted(intervals, key=lambda entry: (entry[1], entry[2], entry[0]))
        for left_index, first in enumerate(ordered):
            first_row, first_start, first_end, _ = first
            for second in ordered[left_index + 1 :]:
                second_row, second_start, second_end, _ = second
                if second_start >= first_end:
                    break
                if first_start == second_start and first_end == second_end:
                    continue
                overlap_start = max(first_start, second_start)
                overlap_end = min(first_end, second_end)
                if overlap_start < overlap_end:
                    overlaps.append(
                        IntervalOverlap(
                            delivery_scope=scope,
                            first_row_index=first_row,
                            second_row_index=second_row,
                            overlap_start_wall_time=overlap_start,
                            overlap_end_wall_time=overlap_end,
                        )
                    )

    missing_intervals: tuple[str, ...] = ()
    delivery_day_complete: bool | None = None
    if parsed_delivery_day is not None and expected_by_key is None:
        warnings.append(
            "Delivery-day completeness was not checked: supply an explicit "
            "expected_intervals schedule; no 96-block assumption is applied."
        )
    elif expected_by_key is not None:
        absent_keys = sorted(set(expected_by_key) - present_intervals)
        missing_intervals = tuple(
            f"{start} to {end}" for start, end in absent_keys
        )
        delivery_day_complete = not missing_intervals

    if rows_without_source_timezone:
        warnings.append(
            f"Source timezone metadata is unresolved for {len(rows_without_source_timezone)} "
            "row(s); timestamps were not localized or converted."
        )

    return ValidationReport(
        row_errors=tuple(errors),
        duplicate_findings=tuple(duplicate_findings),
        overlapping_intervals=tuple(overlaps),
        missing_intervals=missing_intervals,
        warnings=tuple(warnings),
        delivery_day_complete=delivery_day_complete,
    )