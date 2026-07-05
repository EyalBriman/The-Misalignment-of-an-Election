# Reproducibility notes

## One-command reproduction

From the repository root:

```bash
python scripts/run_experiments.py
```

This regenerates the paper figures, the result summaries, and the Map-of-Elections/Mapel-compatible experiment folder.

## Random seeds

All random choices are fixed by seed 42.

- synthetic profile generation: `GLOBAL_SEED = 42`
- Pabulib voter downsampling: `PABULIB_SEED = 42`
- MDS embedding: `random_state = 42`

The MDS call also fixes `n_init=4` and `init="random"` to avoid silent changes across newer `scikit-learn` defaults.

## Synthetic population

The script generates 283 synthetic profiles with 50 voters and 10 alternatives:

| Culture family | Count |
|---|---:|
| IC | 25 |
| Mallows, five phi values, 12 each | 60 |
| Urn, four alpha values, 12 each | 48 |
| Euclidean 1D/2D/3D, 25 each | 75 |
| Single-peaked Conitzer | 25 |
| Single-peaked Walsh | 25 |
| Group-separable | 25 |

Total: 283 synthetic profiles.

## Real data

The script reads eight Krakow municipal participatory-budgeting files from Pabulib:

```text
data/krakow_pb/Poland_Krakow_2018.pb
...
data/krakow_pb/Poland_Krakow_2025.pb
```

Each Krakow file is downsampled to:

- 50 voters;
- 10 projects;
- seed 42.

The exact sampled voter indices and selected project IDs are written to:

```text
results/numbers.json
```

under `pabulib_sampling`.

## Cardinal lifts

The robustness analysis uses the three lifts described in the paper:

- Borda: `(m - 1 - rank) / (m - 1)`
- positional: `1 / (rank + 1)`
- exponential: `0.5 ** rank`

Ranks are zero-indexed in the code.

## Output files

The paper outputs are:

```text
figures/figure1_misalignment_map.pdf
figures/figure2_util_vs_ordinal.pdf
figures/figure3_lift_robustness.pdf
figures/figure4_wcr_meta_rule.pdf
results/numbers.json
results/robustness_table.txt
```

The framework-aligned outputs are:

```text
experiments/misalignment/map.csv
experiments/misalignment/summary.csv
experiments/misalignment/elections/*.soc
experiments/misalignment/distances/positionwise.csv
experiments/misalignment/coordinates/mds.csv
experiments/misalignment/features/misalignment.csv
experiments/misalignment/features/ordinal_indices.csv
results/mapel_framework_status.json
```

## Validation checklist

After running the script, check that:

1. `results/numbers.json` exists;
2. `results/robustness_table.txt` exists;
3. the four `figure*.pdf` files exist under `figures/`;
4. `experiments/misalignment/elections/` contains 291 `.soc` files;
5. `experiments/misalignment/features/misalignment.csv` has 291 data rows;
6. `experiments/misalignment/coordinates/mds.csv` has 291 data rows.
