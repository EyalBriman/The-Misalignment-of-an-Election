# Data

This directory contains the Krakow participatory-budgeting files used in the paper.

## Source

The files in `krakow_pb/` are Pabulib `.pb` files for municipal participatory budgeting in Krakow, Poland, 2018--2025:

```text
Poland_Krakow_2018.pb
Poland_Krakow_2019.pb
Poland_Krakow_2020.pb
Poland_Krakow_2021.pb
Poland_Krakow_2022.pb
Poland_Krakow_2023.pb
Poland_Krakow_2024.pb
Poland_Krakow_2025.pb
```

Pabulib is the Participatory Budgeting Library: <https://pabulib.org/>.

## Format

The `.pb` format is a UTF-8 text format with three sections:

- `META`
- `PROJECTS`
- `VOTES`

The replication script reads the `vote` column from the `VOTES` section and uses the metadata fields `vote_type` and `num_projects` from the `META` section.

## Downsampling used in the paper

Each Krakow election is downsampled reproducibly to match the synthetic instance dimensions:

- 50 voters
- 10 projects
- random seed 42

The project-selection score is

```text
score(p) = sum over sampled voters i with p in ballot_i of (20 - rank_i(p)).
```

The top 10 projects by this score are retained. For each sampled voter, retained projects are ordered by the voter's observed order, and unranked retained projects are appended deterministically.

The exact sampled voter indices and selected project IDs are written to `results/numbers.json` under `pabulib_sampling` when `scripts/run_experiments.py` is run.

## Citation

When using these data, cite the Pabulib sources listed in `CITATION.bib`.
