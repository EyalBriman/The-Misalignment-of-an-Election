# Validation

Use this file to check that a local run created all expected figures, numeric summaries, and map-of-elections experiment files.

## Quick local check

Run:

```bash
python scripts/run_experiments.py
python - <<'PY'
from pathlib import Path
import csv

required = [
    'figures/figure1_misalignment_map.pdf',
    'figures/figure2_util_vs_ordinal.pdf',
    'figures/figure3_lift_robustness.pdf',
    'figures/figure4_wcr_meta_rule.pdf',
    'results/numbers.json',
    'results/robustness_table.txt',
    'experiments/misalignment/map.csv',
    'experiments/misalignment/summary.csv',
    'experiments/misalignment/distances/positionwise.csv',
    'experiments/misalignment/coordinates/mds.csv',
    'experiments/misalignment/features/misalignment.csv',
    'experiments/misalignment/features/ordinal_indices.csv',
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit('Missing files: ' + ', '.join(missing))

soc_files = list(Path('experiments/misalignment/elections').glob('*.soc'))
if len(soc_files) != 291:
    raise SystemExit(f'Expected 291 .soc files, found {len(soc_files)}')

for csv_path in [
    'experiments/misalignment/summary.csv',
    'experiments/misalignment/coordinates/mds.csv',
    'experiments/misalignment/features/misalignment.csv',
    'experiments/misalignment/features/ordinal_indices.csv',
]:
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if len(rows) != 292:
        raise SystemExit(f'{csv_path}: expected 291 data rows plus header, found {len(rows)-1}')

print('Validation passed.')
PY
```

## Expected counts

A successful run creates:

- 291 `.soc` elections in `experiments/misalignment/elections/`;
- 291 data rows in `experiments/misalignment/summary.csv`;
- 291 data rows in each feature CSV;
- four figure PDFs in `figures/`;
- `results/numbers.json` and `results/robustness_table.txt`.

`results/mapel_framework_status.json` records whether the package-level Mapel stage was available in the local environment.
