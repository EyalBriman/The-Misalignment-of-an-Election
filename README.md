# The Misalignment of an Election: Replication Package

This repository contains the empirical replication code for:

> **The Misalignment of an Election: Theoretical Foundations and Empirical Landscape**  
> Eyal Briman and Nimrod Talmon, EUMAS 2026.

The code reproduces the empirical part of the paper: synthetic elections, Krakow participatory-budgeting instances from Pabulib, the map-of-elections embedding, cardinal misalignment indices, robustness across cardinal lifts, and the worst-case regret meta-rule.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── DATA_LICENSE.md
├── CITATION.bib
├── CITATION.cff
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
├── figures/        # generated figures are written here
└── results/        # generated JSON/tables are written here
```

## What the script generates

Running the script generates exactly the empirical artifacts described in the paper:

| Paper figure | Output file |
|---|---|
| Figure 1, three misalignment indices on the map of elections | `figures/figure1_misalignment_map.pdf` |
| Figure 2, utilitarian misalignment vs. ordinal indices | `figures/figure2_util_vs_ordinal.pdf` |
| Figure 3, robustness across cardinal lifts | `figures/figure3_lift_robustness.pdf` |
| Figure 4, WCR meta-rule evaluation | `figures/figure4_wcr_meta_rule.pdf` |

The script also writes:

```text
results/numbers.json
results/robustness_table.txt
```

These files contain the numerical summaries used in the prose: misalignment ranges, correlation statistics, Pabulib downsampling records, robustness values, and WCR summaries.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the experiments

From the repository root:

```bash
python scripts/run_experiments.py
```

The script uses fixed random seeds. The synthetic population uses `seed = 42`; each Krakow instance is downsampled with `seed = 42` to 50 voters and 10 projects.

## Data

The eight real participatory-budgeting instances in `data/krakow_pb/` are Krakow municipal PB files from **Pabulib**. Pabulib stores participatory-budgeting data in the `.pb` format with `META`, `PROJECTS`, and `VOTES` sections. See `data/README.md` and `DATA_LICENSE.md` for source and licensing notes.

When using the Pabulib data, cite the Pabulib data/tools paper and the Pabulib format/library paper listed in `CITATION.bib`.

## Notes on scope

This repository intentionally contains only the empirical pipeline described in the paper. It does not include unrelated simulations or exploratory figures.

## License

The code in this repository is released under the MIT License. The Pabulib `.pb` data files are not relicensed by this repository; see `DATA_LICENSE.md`.
