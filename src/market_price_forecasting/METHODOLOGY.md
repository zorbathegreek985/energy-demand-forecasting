# Proposed Market-Price Forecasting Methodology

> **Proposal only.** No IEX data has been downloaded, no IEX permission has
> been confirmed, and no market-price model or trading simulator has been
> implemented. This workflow must remain separate from the household-demand
> model, whose target is daily average `Global_active_power` in kW.

## Confirmed Facts and Unresolved Items

The existing project has a separate provisional `market_price_forecasting`
package and synthetic schema-validation tests. It has no real market input,
forecast model, or trading logic. The [IEX DAM snapshot](https://www.iexindia.com/market-data/day-ahead-market/market-snapshot)
currently describes 15-minute blocks and states that this format is available
from delivery day 1 April 2022. Its page displays an Export control; this does
not establish that automated retrieval or public redistribution is allowed.
The [IEX DAM description](https://www.iexindia.com/products/day-ahead-market)
describes a next-day physical market and a double-sided closed auction.

| Item | Status |
| --- | --- |
| IEX permission for manual/automated access, storage, or publication | **UNCONFIRMED** |
| Actual export headers and file format | **UNCONFIRMED** |
| Price definition and unit | **UNCONFIRMED** |
| Delivery area / price scope | **UNCONFIRMED** |
| Timestamp timezone and block-boundary convention | **UNCONFIRMED** |
| Publication cutoff, timestamp, and revision semantics | **UNCONFIRMED** |

IEX [website terms](https://www.iexindia.com/terms-of-uses) restrict
downloading, copying, extracting, or distributing site content unless
expressly authorized. Written permission and the actual export contract take
precedence over assumptions in this proposal.

## Forecasting Problems

| Problem | Prediction target | Potential use (not a profit claim) |
| --- | --- | --- |
| Next-block | Price for the next delivery interval after the forecast issue time; likely requires an intraday/real-time product and a defined cutoff | Near-term price monitoring and operational analysis |
| Day-ahead block-level | A price for each delivery block on the next delivery day, issued at a cutoff before those prices become known | Procurement, scheduling, or budget scenario analysis |
| Daily average | A defined daily aggregate of block prices for a delivery day | Trend analysis and broad day-to-day comparisons; it hides within-day peaks and scarcity intervals |

The daily aggregate must be specified from the verified price field and use
rule. Arithmetic averaging and volume weighting are different targets; neither
is assumed here.

## Recommended First Target

**Proposal:** predict the next delivery day's IEX DAM price for each 15-minute
delivery block. It is the most direct fit to the public IEX description of
15-minute DAM data and preserves the within-day price shape. Begin with a
forecast issued once per delivery day, rather than a trading strategy.

This choice assumes that a sufficiently continuous, legally usable history can
be obtained and that the export contains an unambiguous block-level price and
scope. It also assumes the issue cutoff can be reconstructed for every
historical forecast date. Those assumptions are not confirmed. If IEX access,
history, or vintage data is insufficient, reconsider a coarser descriptive
target using CERC reports; do not present monthly aggregates as 15-minute
forecasts.

## Forecast Cutoff and Features

For every forecast origin, define an explicit issue timestamp and the set of
information available by that instant. **Delivery time** identifies when
electricity is delivered; it is not the same as **publication/availability
time**, which determines whether a value was knowable to the forecaster.
Every predictor must have an availability timestamp no later than the issue
cutoff. Preserve source publication and retrieval times separately.

Candidate predictors, only if their historical availability can be verified:

- Prior prices for the same product, scope, and delivery block.
- Same-block prices from earlier delivery days or weeks, as published before
  the cutoff.
- Calendar and seasonal features known at the cutoff.
- Demand/load, renewable generation, weather, fuel, outage, and congestion
  information only in the versions published by the cutoff. Realized future
  values are not substitutes for historical forecasts.

Price revisions must not overwrite what was originally available at an earlier
forecast origin. If vintage/publication history is unavailable, document that
the model cannot reconstruct historical information sets and limit claims
accordingly.

## Baselines

Compare any future model to simple, cutoff-safe baselines:

- **Persistence:** the latest price for the same product/scope that was
  published strictly before the issue cutoff. Never use a later interval just
  because it is adjacent in the export.
- **Historical same-block:** a prior-day or prior-week price for the same
  delivery block, only if it was available by the cutoff. If that observation
  is absent, report the baseline as unavailable or define a fallback using
  earlier observations; do not fill it silently.

## Evaluation Proposal

Use contiguous chronological training, validation, and final test periods.
Choose boundaries by delivery date before excluding rows with missing targets
or unavailable features; report calendar boundaries, eligible dates, and
exclusions. Consider expanding-window rolling-origin validation within the
development period. Keep the final test period out of feature selection,
parameter selection, and model selection.

Report MAE and RMSE in the **verified price unit**, observation counts, and
scored date/block coverage. Compare each model with the same eligible dates
and the same baselines. Random splitting is unsuitable because it mixes
future and past market conditions across partitions, can leak repeated
calendar patterns, and does not reproduce a forecast made at a past cutoff.
No metrics or forecast results exist yet.

## Forecasting Is Not Trading

A price forecast is not investment advice or a trading strategy and does not imply profitability.
A later simulator would be a separate scope requiring explicit position
rules, decision timing, transaction fees, slippage, limits, settlement and
imbalance treatment, and risk measures. It must use only forecasts and data
available at each simulated decision time.

## Data and Permission Gate

Before real-data ingestion, obtain written IEX clarification covering manual
and automated access, local retention, public GitHub publication, derived
outputs/attribution, and fees. Then validate one authorized export and confirm
the actual columns, price field/unit, scope, time convention, publication
timestamp, and revision process. Keep restricted source files and artifacts
outside Git. The schema's canonical names are not claims about IEX's actual
export format.

## Phased Implementation

1. **Data audit:** permission, one authorized sample, field/unit/scope/time
   confirmation, interval completeness, duplicate/revision review.
2. **Baseline:** define cutoff and scoring dates; compute only persistence and
   historical same-block baselines from available information.
3. **Features:** add price lags and external predictors only with verified
   historical availability/vintages.
4. **Model comparison:** compare modest forecasters against the baselines using
   chronological development data.
5. **Held-out evaluation:** run once on the reserved period; report errors,
   coverage, and limitations without changing the selected model afterward.
6. **Optional simulator:** consider only after separate approval and explicit
   transaction-cost, settlement, constraint, and risk assumptions.

## Limitations and Open Questions

- Has IEX authorized the intended academic use, automated access, retention,
  and public reporting?
- What are the exact export headers, price definition/unit, market scope, and
  delivery timestamp/timezone convention?
- What is the publication cutoff for each historical auction and block? Are
  provisional and revised records identifiable and archived?
- Is there enough continuous block history from the stated availability date
  after gaps and revisions are handled?
- Which historical demand, weather, renewable, fuel, outage, or congestion
  predictors have usable as-published vintages and compatible licensing?
- If forecast vintages cannot be obtained, which feature set can be honestly
  evaluated without look-ahead bias?

Until these questions are resolved, the scope is methodology planning and
synthetic validation only; no market-price forecasting result should be
claimed.