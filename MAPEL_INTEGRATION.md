# Mapel / Map-of-Elections integration

This version keeps the original empirical outputs of the misalignment paper, but moves the map part into the Mapel / Map-of-Elections style framework.

## Files to replace or add

Replace:

```text
scripts/run_experiments.py
requirements.txt
```

Add this note if desired:

```text
MAPEL_INTEGRATION.md
```

The data files stay where they already were:

```text
data/krakow_pb/Poland_Krakow_*.pb
```

## What stays the same

Running

```bash
python scripts/run_experiments.py
```

still writes the original paper outputs:

```text
figures/figure1_misalignment_map.pdf
figures/figure2_util_vs_ordinal.pdf
figures/figure3_lift_robustness.pdf
figures/figure4_wcr_meta_rule.pdf
results/numbers.json
results/robustness_table.txt
```

The original computations for synthetic profiles, Krakow downsampling, cardinal lifts, misalignment, correlations, WCR, and the final matplotlib figures are intentionally preserved.

## What is new

The script now also prepares a Mapel experiment under:

```text
experiments/misalignment/
```

with the following structure:

```text
experiments/misalignment/
├── map.csv
├── elections/
│   └── *.soc
├── distances/
│   └── positionwise.csv
├── coordinates/
│   └── mds.csv
├── features/
│   ├── misalignment.csv
│   └── ordinal_indices.csv
├── matrices/
└── summary.csv
```

It also writes:

```text
results/mapel_framework_status.json
```

This status file records whether the Mapel experiment object was successfully built and whether the custom Mapel distance matches the original distance matrix.

## Why coordinates are injected rather than recomputed

Mapel has its own embedding functions, including MDS. However, recomputing an embedding inside Mapel may rotate, rescale, or otherwise slightly change the coordinates. Since the goal here is to preserve the paper outputs exactly, the script computes the MDS coordinates using the original sklearn call and then injects those same coordinates into the Mapel experiment object.

## Requirements

Install with:

```bash
pip install -r requirements.txt
```

The important new dependency is:

```text
mapel>=2.0.1
```

The `mapel-elections` dependency is listed explicitly because this script uses `import mapel.elections as mapel`.

## Troubleshooting

If Mapel import or API calls fail, the original figures and result files are written first, and the failure details are written to:

```text
results/mapel_framework_status.json
```

To temporarily disable the Mapel framework phase, edit the top of `scripts/run_experiments.py`:

```python
USE_MAPEL_FRAMEWORK = False
REQUIRE_MAPEL = False
```

Do this only for debugging. For the Map-of-Elections-framework version, keep both as `True`.
