# The Misalignment of an Election: Replication Package

This repository contains the empirical code and data for:

> **The Misalignment of an Election: Theoretical Foundations and Empirical Landscape**  
> Eyal Briman and Nimrod Talmon, EUMAS 2026.

The experiments are represented in a Map-of-Elections style layout: each election is stored as an ordinal profile, pairwise positionwise distances are computed, two-dimensional coordinates are produced, and the map is colored by misalignment and ordinal features.

Related framework resources:

- Drawing a Map of Elections code appendix: https://github.com/Project-PRAGMA/Journal---Drawing-a-Map-of-Elections
- Mapof-Elections documentation: https://science-for-democracy.github.io/mapof-elections/
- Mapof-Elections GitHub repository: https://github.com/science-for-democracy/mapof-elections
- Mapel GitHub repository: https://github.com/szufix/mapel

## Repository structure

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── MAPEL_INTEGRATION.md
├── VALIDATION.md
├── requirements.txt
├── scripts/
│   └── run_experiments.py
├── data/
│   ├── README.md
│   └── krakow_pb/
│       ├── Poland_Krakow_2018.pb
│       ├── Poland_Krakow_2019.pb
│       ├── Poland_Krakow_2020.pb
│       ├── Poland_Krakow_2021.pb
│       ├── Poland_Krakow_2022.pb
│       ├── Poland_Krakow_2023.pb
│       ├── Poland_Krakow_2024.pb
│       └── Poland_Krakow_2025.pb
├── experiments/
│   └── misalignment/        # generated map-of-elections experiment
├── figures/                 # generated paper figures
└── results/                 # generated numeric summaries
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows Git Bash
source .venv/Scripts/activate

pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## Reproducing the experiments

From the repository root, run:

```bash
python scripts/run_experiments.py
```

The script generates 283 synthetic elections and 8 Krakow participatory-budgeting elections, computes the positionwise map, computes the cardinal and ordinal features, and writes the figures and result files.

## Main outputs

| Artifact | Output file |
|---|---|
| Figure 1, three misalignment indices on the map of elections | `figures/figure1_misalignment_map.pdf` |
| Figure 2, utilitarian misalignment vs. ordinal indices | `figures/figure2_util_vs_ordinal.pdf` |
| Figure 3, robustness across cardinal lifts | `figures/figure3_lift_robustness.pdf` |
| Figure 4, worst-case-regret meta-rule evaluation | `figures/figure4_wcr_meta_rule.pdf` |
| Numerical summaries | `results/numbers.json` |
| Robustness table | `results/robustness_table.txt` |

The map-of-elections experiment is written to:

```text
experiments/misalignment/
├── map.csv
├── summary.csv
├── elections/
│   └── *.soc
├── distances/
│   └── positionwise.csv
├── coordinates/
│   └── mds.csv
├── features/
│   ├── misalignment.csv
│   └── ordinal_indices.csv
└── matrices/
```

## Method in brief

The experiment uses:

- 50 voters and 10 alternatives/projects per profile;
- 283 synthetic profiles from IC, Mallows, urn, Euclidean, single-peaked, and group-separable cultures;
- 8 Krakow participatory-budgeting profiles from Pabulib, downsampled reproducibly to 50 voters and 10 projects;
- positionwise distance with Hungarian matching;
- MDS coordinates for the two-dimensional map;
- Borda utilities for the main misalignment values;
- positional and exponential utilities for robustness checks;
- diversity, agreement, and polarization as ordinal descriptors;
- a worst-case-regret meta-rule over the egalitarian, utilitarian, and Nash objectives.

## Data

The eight real participatory-budgeting instances in `data/krakow_pb/` are Krakow municipal PB files from **Pabulib**. They are downsampled reproducibly to 50 voters and 10 projects. See `data/README.md` and `DATA_LICENSE.md` for source and licensing notes.

When using the Pabulib data, cite the Pabulib data/tools paper and the Pabulib format/library paper listed in `CITATION.bib`.

## License

The code in this repository is released under the MIT License. The Pabulib `.pb` data files are not relicensed by this repository; see `DATA_LICENSE.md`.
