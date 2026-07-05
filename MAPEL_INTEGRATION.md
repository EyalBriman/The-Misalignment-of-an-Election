# Map-of-Elections / Mapel integration

This repository keeps the empirical outputs of *The Misalignment of an Election* and aligns the data flow with the Map-of-Elections/Mapel framework.

## Design principle

The original paper figures must remain reproducible. Therefore, the script keeps the original deterministic computations and exports their objects into a Mapel-compatible experiment layout:

```text
experiments/misalignment/
├── map.csv
├── summary.csv
├── elections/*.soc
├── distances/positionwise.csv
├── coordinates/mds.csv
├── features/misalignment.csv
├── features/ordinal_indices.csv
└── matrices/
```

The exported files correspond exactly to the objects used by the figures:

- `elections/*.soc` stores the generated/downsampled ordinal elections;
- `distances/positionwise.csv` stores the positionwise distance matrix;
- `coordinates/mds.csv` stores the two-dimensional MDS coordinates;
- `features/misalignment.csv` stores `mu_egal`, `mu_util`, `mu_nash`, best alternatives, WCR values, and the meta-rule pick;
- `features/ordinal_indices.csv` stores diversity, agreement, and polarization;
- `summary.csv` joins coordinates and features into one easy-to-read table.

## Optional live Mapel stage

At the end of `scripts/run_experiments.py`, the function `run_mapel_framework_stage(...)` does two things.

First, it always writes the offline Mapel-compatible experiment files above.

Second, if the local environment has `mapel` and `mapel-elections`, it attempts to build a live Mapel ordinal experiment object by using:

```python
import mapel.elections as mapel
experiment = mapel.prepare_online_ordinal_experiment()
```

The script then registers fixed generated profiles, a custom positionwise distance matching the paper metric, and three custom Borda-misalignment features.

The status is written to:

```text
results/mapel_framework_status.json
```

This file is diagnostic only. The paper figures and numeric results do not depend on whether the optional live Mapel API stage succeeds.

## Why the coordinates are not recomputed by Mapel

The map in the paper depends on the exact MDS output. MDS coordinates can change by rotation, reflection, scaling, or small implementation details even when the underlying distances are the same. To preserve the exact paper figures, the script computes the MDS coordinates once using the original deterministic call and exports those coordinates to the Mapel-compatible experiment folder.

This is the safest way to make the repository framework-aligned while keeping the published artifacts reproducible.

## How to rerun

```bash
python scripts/run_experiments.py
```

Expected main outputs:

```text
figures/figure1_misalignment_map.pdf
figures/figure2_util_vs_ordinal.pdf
figures/figure3_lift_robustness.pdf
figures/figure4_wcr_meta_rule.pdf
results/numbers.json
results/robustness_table.txt
experiments/misalignment/
```

## Developer switches

Near the top of `scripts/run_experiments.py`:

```python
USE_MAPEL_FRAMEWORK = True
REQUIRE_MAPEL = False
```

`USE_MAPEL_FRAMEWORK=True` means the script writes the Mapel-compatible experiment files and tries the live Mapel API stage.

`REQUIRE_MAPEL=False` means the paper outputs are still produced even if Mapel is not installed or if a future Mapel API version changes. Set it to `True` only when you want the script to fail on a Mapel API issue.
