# The Misalignment of an Election: Replication Package

This repository contains the empirical replication code for:

> **The Misalignment of an Election: Theoretical Foundations and Empirical Landscape**  
> Eyal Briman and Nimrod Talmon, EUMAS 2026.

The pipeline reproduces the empirical part of the paper: synthetic elections, Krakow participatory-budgeting elections from Pabulib, the map-of-elections embedding, the three cardinal misalignment indices, robustness across cardinal lifts, and the worst-case-regret meta-rule.

The code is now organized around a Map-of-Elections/Mapel-compatible experiment layout. The original paper outputs are preserved, while the generated elections, distances, coordinates, and features are also exported to `experiments/misalignment/`.

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
│   └── misalignment/        # generated Map-of-Elections/Mapel-compatible experiment
├── figures/                 # generated paper figures
└── results/                 # generated numeric summaries
```

## What running the script produces

From the repository root, run:

```bash
python scripts/run_experiments.py
```

The script writes the same paper outputs as the standalone version:

| Paper artifact | Output file |
|---|---|
| Figure 1, three misalignment indices on the map of elections | `figures/figure1_misalignment_map.pdf` |
| Figure 2, utilitarian misalignment vs. ordinal indices | `figures/figure2_util_vs_ordinal.pdf` |
| Figure 3, robustness across cardinal lifts | `figures/figure3_lift_robustness.pdf` |
| Figure 4, worst-case-regret meta-rule evaluation | `figures/figure4_wcr_meta_rule.pdf` |
| Numerical summaries | `results/numbers.json` |
| Robustness table | `results/robustness_table.txt` |

It also writes the Map-of-Elections/Mapel-compatible experiment files:

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

The `results/mapel_framework_status.json` file records whether the optional live Mapel API stage was available in the local Python environment.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows Git Bash
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## What is preserved from the original script

The following computations are intentionally unchanged:

- synthetic population generation with seed 42;
- Krakow Pabulib parsing and downsampling with seed 42;
- Borda, positional, and exponential cardinal lifts;
- egalitarian, utilitarian, and Nash misalignment;
- diversity, agreement, and polarization descriptors;
- positionwise distance with Hungarian matching;
- MDS map used for the paper figures;
- correlation/robustness analysis;
- worst-case-regret meta-rule.

The Mapel-compatible files are exported from these same objects, so the framework data and the paper figures refer to the same elections, distances, coordinates, and feature values.

## Data

The eight real participatory-budgeting instances in `data/krakow_pb/` are Krakow municipal PB files from **Pabulib**. They are downsampled reproducibly to 50 voters and 10 projects. See `data/README.md` and `DATA_LICENSE.md` for source and licensing notes.

When using the Pabulib data, cite the Pabulib data/tools paper and the Pabulib format/library paper listed in `CITATION.bib`.

## Notes

The live Mapel API is optional at runtime. The script always exports the Map-of-Elections-compatible folder structure. If Mapel is installed, it additionally tries to build the corresponding online Mapel experiment object and records the result in `results/mapel_framework_status.json`. This keeps the replication robust while preserving the exact paper outputs.

## License

The code in this repository is released under the MIT License. The Pabulib `.pb` data files are not relicensed by this repository; see `DATA_LICENSE.md`.
