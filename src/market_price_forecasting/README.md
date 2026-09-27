# Market Price Forecasting (Provisional)

This package is separate from `energy_forecasting`, which predicts household
daily average active power in kW. It does not change or reuse that target/model.

The schema in `schema.py` is provisional and uses canonical project field
names; the actual IEX export headers, price definition, unit, delivery area,
timezone, publication timestamp, and revision semantics are unconfirmed.
Values for price definition/unit/scope must be supplied explicitly; this
package assigns no source-specific defaults.

The tests use clearly labelled synthetic fixtures only. They are not IEX data
and must not be presented as market observations or results. No market data
has been downloaded. Real-data ingestion is blocked pending IEX permission and
confirmation of the export format and usage terms. No price forecasting model,
trading simulation, downloader, or dashboard exists in this increment.

The validator reports schema errors, exact and conflicting duplicate keys,
overlapping intervals within a delivery scope, and missing intervals without
dropping, deduplicating, interpolating, or otherwise changing input records.
It does not assume 96 intervals: completeness is evaluated only when the
caller supplies an explicit expected interval schedule. An optional expected
interval duration can be supplied separately. Timestamp comparison preserves
literal values; no timezone is inferred, localized, or converted. Different
offset representations are not treated as equivalent, and overlap checks are
skipped when offsets within a scope disagree.