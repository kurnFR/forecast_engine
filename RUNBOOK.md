# Production Runbook

## Pre-Run Checklist

- Database connectivity
- Calendar freshness
- Target table refresh
- Validation tests passing

## Execute

python main.py

## Validate Results

Check:

- Row counts
- Missing regions
- Duplicate keys
- Null forecasts
- Negative forecasts

## Rollback

If forecast generation fails:

1. Keep previous forecast snapshot
2. Restore last successful output
3. Investigate logs
