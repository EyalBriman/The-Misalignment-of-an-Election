# Validation

This repository is set up so that the framework-aligned script preserves the original paper outputs.

## What is checked

The script writes the original outputs first:

```text
figures/figure1_misalignment_map.pdf
figures/figure2_util_vs_ordinal.pdf
figures/figure3_lift_robustness.pdf
figures/figure4_wcr_meta_rule.pdf
results/numbers.json
results/robustness_table.txt
```

It then writes the Map-of-Elections/Mapel-compatible experiment folder:

```text
experiments/misalignment/
```

The framework export is derived from the same in-memory profiles, distance matrix, MDS coordinates, and feature values used by the paper figures.

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

## Notes about Mapel availability

`requirements.txt` includes `mapel` and `mapel-elections`. If they are installed, the script also attempts to construct a live Mapel experiment object and records the result in:

```text
results/mapel_framework_status.json
```

If the local Mapel API is unavailable, the offline Mapel-compatible experiment files are still produced. This avoids breaking the replication because of package/API differences outside the paper code.
