# Data

## Symbols and source

| Symbol | Role | Source | Frequency |
| --- | --- | --- | --- |
| ADRO.JK | target | Yahoo Finance | daily |
| PTBA.JK | target | Yahoo Finance | daily |
| ITMG.JK | target / peer | Yahoo Finance | daily |
| IDR=X | external (USD/IDR) | Yahoo Finance | daily |

Targets and features use `Adj Close`. The loader keeps `Close` around for
the audit trail only.

## Requested period

```text
[2016-01-01, 2026-05-01)
```

Yahoo Finance is mutable: rows and adjusted values can change between
retrievals. Every snapshot is cached as CSV with a JSON sidecar recording
symbol, requested dates, interval, retrieval timestamp, and row count. A
cache is reused only when the sidecar matches the active config; a mismatch
stops the run. `results/data_manifest.json` records SHA-256 hashes so the
exact dataset behind a result is traceable.

## Availability assumptions

- A row dated `t` is treated as known after the target equity's daily close.
- Its target is the log return to the next observed target-equity trading
  day.
- External features use a conservative one-observation availability lag and
  backward as-of alignment: at equity date `t`, only the most recent source
  observation at or before `t` (shifted one step back) may enter. This is a
  safety assumption, not proof of perfect timestamp availability.
- No imputation: incomplete feature rows are dropped and documented.

## Coal-price policy

No coal-price feature is included. A coal variable enters only with a
historical value, a reference period, a release date, and a publication
timing rule. Backfilling a monthly HBA value would leak later-known
information into earlier rows.
