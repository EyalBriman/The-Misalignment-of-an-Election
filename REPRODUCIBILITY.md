# Reproducibility notes

## Random seeds

All random choices are fixed by seed 42.

- synthetic profile generation: `GLOBAL_SEED = 42`
- Pabulib voter downsampling: `PABULIB_SEED = 42`
- MDS embedding: `random_state = 42`

## Synthetic population

The script generates 283 synthetic profiles with 50 voters and 10 alternatives:

- IC: 25
- Mallows: 5 parameter values, 12 profiles each, total 60
- Urn: 4 parameter values, 12 profiles each, total 48
- Euclidean: 1D/2D/3D, 25 profiles each, total 75
- Single-peaked: Conitzer and Walsh, 25 profiles each, total 50
- Group-separable: 25

Total: 25 + 60 + 48 + 75 + 50 + 25 = 283.

## Real data

The script reads eight Krakow municipal PB files from Pabulib and downsampled each to 50 voters and 10 projects.

The exact sampled voters and retained projects are recorded in `results/numbers.json` after running the script.

## Cardinal lifts

The robustness analysis uses exactly the three lifts described in the paper:

- Borda: `(m - 1 - rank) / (m - 1)`
- positional: `1 / (rank + 1)`
- exponential: `0.5^rank`

Ranks are zero-indexed in the code.

## Generated files

The script writes four figures and two result files:

```text
figures/figure1_misalignment_map.pdf
figures/figure2_util_vs_ordinal.pdf
figures/figure3_lift_robustness.pdf
figures/figure4_wcr_meta_rule.pdf
results/numbers.json
results/robustness_table.txt
```
