# Map-of-Elections / Mapel experiment layout

This repository uses the map-of-elections viewpoint: an experiment consists of a collection of elections, a distance between elections, a two-dimensional representation, and feature values used to color or analyze the map.

Useful references:

- Drawing a Map of Elections code appendix: https://github.com/Project-PRAGMA/Journal---Drawing-a-Map-of-Elections
- Mapof-Elections documentation: https://science-for-democracy.github.io/mapof-elections/
- Mapof-Elections GitHub repository: https://github.com/science-for-democracy/mapof-elections
- Mapel GitHub repository: https://github.com/szufix/mapel

## Experiment folder

Running

```bash
python scripts/run_experiments.py
```

creates the experiment folder:

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

The files have the following roles:

- `elections/*.soc` stores the generated/downsampled ordinal elections;
- `distances/positionwise.csv` stores the positionwise distance matrix;
- `coordinates/mds.csv` stores the two-dimensional MDS coordinates;
- `features/misalignment.csv` stores `mu_egal`, `mu_util`, `mu_nash`, best alternatives, WCR values, and the meta-rule pick;
- `features/ordinal_indices.csv` stores diversity, agreement, and polarization;
- `summary.csv` joins coordinates and features into one table;
- `map.csv` provides the election metadata used by map-of-elections tools.

## Mapel package stage

The script imports Mapel through:

```python
import mapel.elections as mapel
```

when the package is available. It then constructs an online ordinal experiment, registers the fixed elections as custom cultures, registers the positionwise distance used in the paper, and registers the Borda-based misalignment values as features.

The status of this package-level stage is written to:

```text
results/mapel_framework_status.json
```

The CSV and `.soc` experiment files are always written, so the map-of-elections data are available even on machines where the Mapel package is not installed.

## Coordinates

The MDS coordinates are exported explicitly to `coordinates/mds.csv`. This makes the figures and the experiment folder refer to the same two-dimensional representation and avoids changes due to different MDS defaults across Python package versions.

## Developer switches

Near the top of `scripts/run_experiments.py`:

```python
USE_MAPEL_FRAMEWORK = True
REQUIRE_MAPEL = False
```

`USE_MAPEL_FRAMEWORK=True` writes the map-of-elections experiment files and attempts the Mapel package stage.

`REQUIRE_MAPEL=False` allows the CSV/`.soc` experiment files and figures to be generated even if Mapel is unavailable. Set it to `True` when package-level Mapel execution should be mandatory.
