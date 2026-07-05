#!/usr/bin/env python3
"""
Replication pipeline for "The Misalignment of an Election".

This script generates exactly the empirical artifacts described in the paper:

  Figure 1: three misalignment indices on the map of elections
  Figure 2: utilitarian misalignment vs. ordinal indices
  Figure 3: robustness of correlations across cardinal lifts
  Figure 4: worst-case regret meta-rule evaluation

It uses:
  - 283 synthetic ordinal profiles from canonical statistical cultures;
  - 8 Krakow participatory-budgeting profiles from Pabulib, downsampled
    reproducibly to 50 voters x 10 projects.

Outputs are written to figures/ and results/.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import prefsampling.ordinal as ps
import scipy.stats as stats
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import MDS

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(divide="ignore", invalid="ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "krakow_pb"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
MAPEL_DIR = ROOT / "mapel" / "misalignment"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MAPEL_DIR.mkdir(parents=True, exist_ok=True)

N_VOTERS = 50
N_ALTS = 10
N_PER_CULTURE = 25
N_BOOTSTRAP = 200
GLOBAL_SEED = 42

PABULIB_VOTERS = 50
PABULIB_ALTS = 10
PABULIB_SEED = 42

np.random.seed(GLOBAL_SEED)


# ---------------------------------------------------------------------------
# Pabulib parser and downsampling
# ---------------------------------------------------------------------------

def parse_pb_file(filepath: Path) -> Tuple[str, List[List[str]], int, Dict[str, str]]:
    """Parse a Pabulib .pb file.

    Returns (vote_type, votes, num_projects, meta), where votes is a list of
    project-id lists, read from the VOTES section's ``vote`` column.
    """
    with filepath.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    sections: Dict[str, List[str]] = {"META": [], "PROJECTS": [], "VOTES": []}
    current = None
    for line in lines:
        stripped = line.strip()
        if stripped in sections:
            current = stripped
        elif current is not None:
            sections[current].append(line)

    meta: Dict[str, str] = {}
    for line in sections["META"][1:]:
        if ";" not in line:
            continue
        key, value = line.split(";", 1)
        meta[key.strip()] = value.strip()

    vote_type = meta.get("vote_type", "unknown")
    num_projects = int(meta.get("num_projects", 0))

    votes: List[List[str]] = []
    if len(sections["VOTES"]) > 1:
        header = [h.strip() for h in sections["VOTES"][0].split(";")]
        vote_col = header.index("vote") if "vote" in header else -1
        for line in sections["VOTES"][1:]:
            if not line.strip():
                continue
            parts = line.split(";")
            if 0 <= vote_col < len(parts):
                vote_str = parts[vote_col].strip()
                if vote_str:
                    votes.append([p.strip() for p in vote_str.split(",") if p.strip()])

    return vote_type, votes, num_projects, meta


def downsample_and_rank(
    votes: List[List[str]],
    n_voters: int = PABULIB_VOTERS,
    n_alts: int = PABULIB_ALTS,
    seed: int = PABULIB_SEED,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Downsample a Pabulib election to a complete ranking matrix.

    The paper uses 50 voters x 10 projects for each Krakow instance.
    Voters are sampled uniformly without replacement. Projects are selected
    by frequency-weighted rank score over the sampled voters:

        score(p) = sum_{i: p in ballot_i} (20 - rank_i(p)).

    The retained projects are then ordered per voter by the voter's observed
    order, with unranked retained projects appended deterministically.
    """
    rng = np.random.default_rng(seed)

    if len(votes) > n_voters:
        voter_indices = sorted(rng.choice(len(votes), n_voters, replace=False).tolist())
        sampled_votes = [votes[i] for i in voter_indices]
    else:
        voter_indices = list(range(len(votes)))
        sampled_votes = votes

    project_scores: Dict[str, int] = {}
    for vote in sampled_votes:
        for rank, pid in enumerate(vote[:20]):
            project_scores[pid] = project_scores.get(pid, 0) + (20 - rank)

    top_projects = sorted(project_scores.keys(), key=lambda x: project_scores[x], reverse=True)[:n_alts]
    pid_to_idx = {pid: i for i, pid in enumerate(top_projects)}

    rankings: List[List[int]] = []
    for vote in sampled_votes:
        voter_ranking: List[int] = []
        for pid in vote:
            if pid in pid_to_idx:
                voter_ranking.append(pid_to_idx[pid])
        unranked = [i for i in range(n_alts) if i not in voter_ranking]
        rankings.append((voter_ranking + unranked)[:n_alts])

    sample_info = {
        "voter_indices": [int(i) for i in voter_indices],
        "selected_projects": list(top_projects),
        "original_voters": int(len(votes)),
        "sampled_voters": int(len(sampled_votes)),
    }
    return np.array(rankings, dtype=int), sample_info


# ---------------------------------------------------------------------------
# Synthetic population
# ---------------------------------------------------------------------------

def gen_synthetic_population() -> List[Tuple[str, np.ndarray]]:
    """Generate the 283 synthetic profiles used in the paper."""
    rng = np.random.default_rng(GLOBAL_SEED)
    profiles: List[Tuple[str, np.ndarray]] = []

    for _ in range(N_PER_CULTURE):
        profiles.append(("IC", ps.impartial(N_VOTERS, N_ALTS, seed=int(rng.integers(1e9)))))

    for phi in [0.1, 0.3, 0.5, 0.7, 0.95]:
        for _ in range(N_PER_CULTURE // 2):
            profiles.append((
                f"Mallows-phi={phi}",
                ps.mallows(N_VOTERS, N_ALTS, phi=phi, seed=int(rng.integers(1e9))),
            ))

    for alpha in [0.1, 0.5, 1.0, 5.0]:
        for _ in range(N_PER_CULTURE // 2):
            profiles.append((
                f"Urn-alpha={alpha}",
                ps.urn(N_VOTERS, N_ALTS, alpha=alpha, seed=int(rng.integers(1e9))),
            ))

    for d in [1, 2, 3]:
        for _ in range(N_PER_CULTURE):
            space = ps.EuclideanSpace.UNIFORM_CUBE
            profiles.append((
                f"Euclidean-{d}D",
                ps.euclidean(
                    N_VOTERS,
                    N_ALTS,
                    num_dimensions=d,
                    voters_positions=space,
                    candidates_positions=space,
                    seed=int(rng.integers(1e9)),
                ),
            ))

    for _ in range(N_PER_CULTURE):
        profiles.append(("SP-Conitzer", ps.single_peaked_conitzer(N_VOTERS, N_ALTS, seed=int(rng.integers(1e9)))))
    for _ in range(N_PER_CULTURE):
        profiles.append(("SP-Walsh", ps.single_peaked_walsh(N_VOTERS, N_ALTS, seed=int(rng.integers(1e9)))))

    for _ in range(N_PER_CULTURE):
        profiles.append(("Group-separable", ps.group_separable(N_VOTERS, N_ALTS, seed=int(rng.integers(1e9)))))

    return [(name, np.asarray(ranks, dtype=int)) for name, ranks in profiles]


# ---------------------------------------------------------------------------
# Cardinal lifts and misalignment
# ---------------------------------------------------------------------------

def borda_utilities(rankings: np.ndarray) -> np.ndarray:
    rankings = np.asarray(rankings, dtype=int)
    n, m = rankings.shape
    u = np.zeros((n, m))
    for v in range(n):
        for r in range(m):
            u[v, rankings[v, r]] = (m - 1 - r) / (m - 1)
    return u


def positional_utilities(rankings: np.ndarray) -> np.ndarray:
    rankings = np.asarray(rankings, dtype=int)
    n, m = rankings.shape
    u = np.zeros((n, m))
    for v in range(n):
        for r in range(m):
            u[v, rankings[v, r]] = 1.0 / (r + 1)
    return u


def exponential_utilities(rankings: np.ndarray, beta: float = 0.5) -> np.ndarray:
    rankings = np.asarray(rankings, dtype=int)
    n, m = rankings.shape
    u = np.zeros((n, m))
    for v in range(n):
        for r in range(m):
            u[v, rankings[v, r]] = beta ** r
    return u


def relative_satisfaction(u: np.ndarray) -> np.ndarray:
    u_star = u.max(axis=1, keepdims=True)
    u_star = np.where(u_star == 0, 1.0, u_star)
    return u / u_star


def geometric_mean_with_zeros(s: np.ndarray, axis: int = 0) -> np.ndarray:
    """Geometric mean with the paper's zero convention."""
    log_s = np.where(s > 0, np.log(s), -50.0)
    return np.exp(log_s.mean(axis=axis))


def misalignment(u: np.ndarray) -> Dict[str, float | int]:
    s = relative_satisfaction(u)
    egal_per_a = s.min(axis=0)
    util_per_a = s.mean(axis=0)
    nash_per_a = geometric_mean_with_zeros(s, axis=0)
    return {
        "egal": float(1.0 - egal_per_a.max()),
        "util": float(1.0 - util_per_a.max()),
        "nash": float(1.0 - nash_per_a.max()),
        "best_egal_alt": int(egal_per_a.argmax()),
        "best_util_alt": int(util_per_a.argmax()),
        "best_nash_alt": int(nash_per_a.argmax()),
    }


# ---------------------------------------------------------------------------
# Ordinal indices and map-of-elections embedding
# ---------------------------------------------------------------------------

def position_matrix(rankings: np.ndarray) -> np.ndarray:
    rankings = np.asarray(rankings, dtype=int)
    n, m = rankings.shape
    P = np.zeros((m, m))
    for v in range(n):
        for r in range(m):
            P[rankings[v, r], r] += 1
    return P / n


def diversity(rankings: np.ndarray) -> float:
    """Normalized mean rank-position entropy."""
    P = position_matrix(rankings)
    m = P.shape[1]
    H = 0.0
    for r in range(m):
        col = P[:, r]
        nz = col[col > 0]
        H -= float((nz * np.log(nz)).sum()) if len(nz) else 0.0
    return float(H / (m * np.log(m)))


def agreement(rankings: np.ndarray) -> float:
    """Mean absolute pairwise majority margin."""
    rankings = np.asarray(rankings, dtype=int)
    n, m = rankings.shape
    pair = np.zeros((m, m))
    for v in range(n):
        rk = np.zeros(m, dtype=int)
        for r, a in enumerate(rankings[v]):
            rk[a] = r
        for a in range(m):
            for b in range(m):
                if a != b and rk[a] < rk[b]:
                    pair[a, b] += 1
    margins = np.abs(pair - n / 2) / (n / 2)
    return float(margins[np.triu_indices(m, k=1)].mean())


def polarization(rankings: np.ndarray) -> float:
    """Weighted position-distribution variance proxy."""
    P = position_matrix(rankings)
    m = P.shape[1]
    ranks = np.arange(m)
    pol = 0.0
    for a in range(m):
        if P[a].sum() == 0:
            continue
        mean = (P[a] * ranks).sum()
        var = (P[a] * (ranks - mean) ** 2).sum()
        ent = -np.sum(P[a, P[a] > 0] * np.log(P[a, P[a] > 0]))
        pol += var * (np.log(m) - ent) / m
    return float(pol)


def positionwise_distance(P1: np.ndarray, P2: np.ndarray) -> float:
    cost = np.abs(P1[:, None, :] - P2[None, :, :]).sum(axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].sum())


def compute_mds_embedding(profiles: List[Dict[str, object]]) -> np.ndarray:
    P_mats = [position_matrix(d["ranks"]) for d in profiles]
    n = len(P_mats)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = positionwise_distance(P_mats[i], P_mats[j])
    try:
        return MDS(n_components=2, dissimilarity="precomputed", random_state=GLOBAL_SEED, normalized_stress="auto").fit_transform(D)
    except TypeError:
        return MDS(n_components=2, dissimilarity="precomputed", random_state=GLOBAL_SEED).fit_transform(D)


# ---------------------------------------------------------------------------
# Map-of-elections framework export
# ---------------------------------------------------------------------------

def safe_id(text: str) -> str:
    """Create a filesystem-safe election id."""
    text = str(text).replace("=", "-").replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def election_id(index: int, profile: Dict[str, object]) -> str:
    """Stable id used in all MapEl-style output files."""
    return f"{index:03d}_{safe_id(str(profile['culture']))}"


def rankings_to_vote_counts(rankings: np.ndarray) -> Dict[Tuple[int, ...], int]:
    """Compress rankings into vote multiplicities, as in .soc files."""
    counts: Dict[Tuple[int, ...], int] = {}
    for row in np.asarray(rankings, dtype=int):
        vote = tuple(int(x) for x in row)
        counts[vote] = counts.get(vote, 0) + 1
    return counts


def write_soc_file(path: Path, rankings: np.ndarray, title: str) -> None:
    """Write one election in a PrefLib/MapEl-readable strict-order .soc format."""
    rankings = np.asarray(rankings, dtype=int)
    n, m = rankings.shape
    counts = rankings_to_vote_counts(rankings)

    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# FILE NAME: {path.name}\n")
        f.write(f"# TITLE: {title}\n")
        f.write("# DATA TYPE: soc\n")
        f.write(f"# NUMBER ALTERNATIVES: {m}\n")
        f.write(f"# NUMBER VOTERS: {n}\n")
        for a in range(m):
            f.write(f"# ALTERNATIVE NAME {a}: c{a}\n")
        f.write(f"{n}, {len(counts)}, {len(counts)}\n")
        for vote, count in sorted(counts.items()):
            f.write(f"{count}: " + ",".join(str(a) for a in vote) + "\n")


def compute_positionwise_distance_matrix(profiles: List[Dict[str, object]]) -> np.ndarray:
    """Distance matrix used by the map: positionwise distance with Hungarian matching."""
    p_mats = [position_matrix(d["ranks"]) for d in profiles]
    n = len(p_mats)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = positionwise_distance(p_mats[i], p_mats[j])
    return dist


def export_map_of_elections_files(
    profiles: List[Dict[str, object]],
    embed: np.ndarray,
    out_dir: Path = MAPEL_DIR,
) -> None:
    """Export the experiment in the style of the Map of Elections pipeline.

    This does not change the experiment. It only writes the generated elections,
    the positionwise distance matrix, the 2D coordinates, and the feature values
    used for coloring the map.
    """
    elections_dir = out_dir / "elections"
    features_dir = out_dir / "features"
    distances_dir = out_dir / "distances"
    coordinates_dir = out_dir / "coordinates"
    for directory in [elections_dir, features_dir, distances_dir, coordinates_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    ids = [election_id(i, d) for i, d in enumerate(profiles)]

    for eid, d in zip(ids, profiles):
        write_soc_file(elections_dir / f"{eid}.soc", np.asarray(d["ranks"], dtype=int), str(d["culture"]))

    dist = compute_positionwise_distance_matrix(profiles)
    with (distances_dir / "positionwise.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["election_id"] + ids)
        for eid, row in zip(ids, dist):
            writer.writerow([eid] + [f"{float(x):.12g}" for x in row])

    with (coordinates_dir / "mds.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["election_id", "x", "y"])
        for eid, (x, y) in zip(ids, embed):
            writer.writerow([eid, f"{float(x):.12g}", f"{float(y):.12g}"])

    with (features_dir / "misalignment.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "election_id", "mu_egal", "mu_util", "mu_nash",
            "best_egal_alt", "best_util_alt", "best_nash_alt",
            "wcr_egal", "wcr_util", "wcr_nash", "wcr_meta", "meta_pick",
        ])
        for eid, d in zip(ids, profiles):
            mis = d["mis"]
            wcrs = d.get("wcrs", {})
            meta = min(wcrs.values()) if wcrs else ""
            writer.writerow([
                eid,
                f"{float(mis['egal']):.12g}",
                f"{float(mis['util']):.12g}",
                f"{float(mis['nash']):.12g}",
                int(mis["best_egal_alt"]),
                int(mis["best_util_alt"]),
                int(mis["best_nash_alt"]),
                f"{float(wcrs.get('egal', np.nan)):.12g}" if wcrs else "",
                f"{float(wcrs.get('util', np.nan)):.12g}" if wcrs else "",
                f"{float(wcrs.get('nash', np.nan)):.12g}" if wcrs else "",
                f"{float(meta):.12g}" if wcrs else "",
                d.get("meta_pick", ""),
            ])

    with (features_dir / "ordinal_indices.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["election_id", "diversity", "agreement", "polarization"])
        for eid, d in zip(ids, profiles):
            idx = d["idx"]
            writer.writerow([
                eid,
                f"{float(idx['diversity']):.12g}",
                f"{float(idx['agreement']):.12g}",
                f"{float(idx['polarization']):.12g}",
            ])

    with (out_dir / "map.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "election_id", "culture", "source", "x", "y",
            "mu_egal", "mu_util", "mu_nash",
            "diversity", "agreement", "polarization",
            "meta_pick",
        ])
        for eid, d, (x, y) in zip(ids, profiles, embed):
            writer.writerow([
                eid, d["culture"], d["source"], f"{float(x):.12g}", f"{float(y):.12g}",
                f"{float(d['mis']['egal']):.12g}",
                f"{float(d['mis']['util']):.12g}",
                f"{float(d['mis']['nash']):.12g}",
                f"{float(d['idx']['diversity']):.12g}",
                f"{float(d['idx']['agreement']):.12g}",
                f"{float(d['idx']['polarization']):.12g}",
                d.get("meta_pick", ""),
            ])


# ---------------------------------------------------------------------------
# Statistical summaries
# ---------------------------------------------------------------------------

def bootstrap_correlation(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOTSTRAP) -> Tuple[float, float]:
    boot_corrs = []
    n = len(x)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        rho, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(rho):
            boot_corrs.append(rho)
    return float(np.percentile(boot_corrs, 2.5)), float(np.percentile(boot_corrs, 97.5))


def correlation_with_significance(x: np.ndarray, y: np.ndarray) -> Dict[str, float | str]:
    rho, p_val = stats.spearmanr(x, y)
    ci_low, ci_high = bootstrap_correlation(x, y)
    if p_val < 0.001:
        sig = "***"
    elif p_val < 0.01:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    return {
        "rho": float(rho),
        "p_value": float(p_val),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significance": sig,
    }


# ---------------------------------------------------------------------------
# WCR meta-rule
# ---------------------------------------------------------------------------

def excess_regret(u: np.ndarray, alt: int, agg: str) -> float:
    s = relative_satisfaction(u)
    mu = misalignment(u)[agg]
    if agg == "egal":
        value = s[:, alt].min()
    elif agg == "util":
        value = s[:, alt].mean()
    elif agg == "nash":
        value = geometric_mean_with_zeros(s[:, alt], axis=0)
    else:
        raise ValueError(f"Unknown aggregator: {agg}")
    return float(max(0.0, (1.0 - value) - float(mu)))


def compute_wcr(profiles: List[Dict[str, object]]) -> Dict[str, List[float]]:
    results = {"egal": [], "util": [], "nash": [], "meta": []}
    for d in profiles:
        u = d["u"]
        mis = d["mis"]
        wcrs = {}
        for label, alt_key in [
            ("egal", "best_egal_alt"),
            ("util", "best_util_alt"),
            ("nash", "best_nash_alt"),
        ]:
            alt = int(mis[alt_key])
            wcrs[label] = max(
                excess_regret(u, alt, "egal"),
                excess_regret(u, alt, "util"),
                excess_regret(u, alt, "nash"),
            )
            results[label].append(wcrs[label])
        results["meta"].append(min(wcrs.values()))
        d["wcrs"] = wcrs
        d["meta_pick"] = min(wcrs, key=wcrs.get)
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def configure_plots() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })


def plot_figure1(profiles: List[Dict[str, object]], embed: np.ndarray) -> None:
    vals_all = {
        "egal": np.array([d["mis"]["egal"] for d in profiles]),
        "util": np.array([d["mis"]["util"] for d in profiles]),
        "nash": np.array([d["mis"]["nash"] for d in profiles]),
    }
    synthetic_mask = np.array([d["source"] == "synthetic" for d in profiles])
    pabulib_mask = ~synthetic_mask

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, key, title in zip(
        axes,
        ["egal", "util", "nash"],
        [r"$\mu_{\rm egal}$", r"$\mu_{\rm util}$", r"$\mu_{\rm nash}$"],
    ):
        sc = ax.scatter(
            embed[synthetic_mask, 0],
            embed[synthetic_mask, 1],
            c=vals_all[key][synthetic_mask],
            s=20,
            cmap="viridis",
            vmin=0,
            vmax=1,
            edgecolors="white",
            linewidths=0.4,
            alpha=0.75,
        )
        if pabulib_mask.sum() > 0:
            ax.scatter(
                embed[pabulib_mask, 0],
                embed[pabulib_mask, 1],
                marker="s",
                s=80,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label="Krakow PB",
            )
            ax.legend(loc="upper right", fontsize=7)
        plt.colorbar(sc, ax=ax, label="misalignment", fraction=0.046, pad=0.03)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure1_misalignment_map.pdf")
    plt.close()


def plot_figure2(mu_util: np.ndarray, ord_indices: Dict[str, np.ndarray], correlations: Dict[str, Dict[str, float | str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, olabel in zip(axes, ["diversity", "agreement", "polarization"]):
        ovals = ord_indices[olabel]
        c = correlations[f"util_{olabel}"]
        ax.scatter(ovals, mu_util, s=12, alpha=0.6, color="C0")
        ax.set_title(
            rf"$\mu_{{\rm util}}$ vs {olabel} "
            rf"($\rho={float(c['rho']):+.2f}${c['significance']}, $p={float(c['p_value']):.3f}$)",
            fontsize=9,
        )
        ax.set_xlabel(olabel)
        ax.set_ylabel(r"$\mu_{\rm util}$")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure2_util_vs_ordinal.pdf")
    plt.close()


def plot_figure3(robustness: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    lift_names = ["Borda", "Positional", "Exponential"]
    corr_names = [
        "egal_diversity", "egal_agreement", "egal_polarization",
        "util_diversity", "util_agreement", "util_polarization",
        "nash_diversity", "nash_agreement", "nash_polarization",
    ]
    heatmap_data = np.zeros((len(corr_names), len(lift_names)))
    for i, cname in enumerate(corr_names):
        for j, lname in enumerate(lift_names):
            heatmap_data[i, j] = robustness[lname][cname]["rho"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap_data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(lift_names)))
    ax.set_xticklabels(lift_names)
    ax.set_yticks(range(len(corr_names)))
    ax.set_yticklabels([cn.replace("_", " × ") for cn in corr_names], fontsize=8)
    ax.set_title("Correlation Robustness Across Lifts")
    for i in range(len(corr_names)):
        for j in range(len(lift_names)):
            ax.text(j, i, f"{heatmap_data[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    plt.colorbar(im, ax=ax, label="Spearman $\\rho$")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure3_lift_robustness.pdf")
    plt.close()


def plot_figure4(profiles: List[Dict[str, object]], embed: np.ndarray, wcr_results: Dict[str, List[float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.05, 1]})

    ax = axes[0]
    for label, color in [("egal", "C0"), ("util", "C1"), ("nash", "C2"), ("meta", "C3")]:
        vals = np.sort(np.array(wcr_results[label]))
        ax.plot(
            vals,
            np.arange(1, len(vals) + 1) / len(vals),
            label=f"$R_{{\\rm {label}}}$" if label != "meta" else r"$R^\star_\Phi$ (meta)",
            color=color,
            lw=1.8,
        )
    ax.set_xlabel("worst-case excess regret")
    ax.set_ylabel("ECDF")
    ax.set_title("WCR distributions")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    colors = {"egal": "#1f77b4", "util": "#ff7f0e", "nash": "#2ca02c"}
    labels_seen = set()
    for d, (x, y) in zip(profiles, embed):
        pick = d["meta_pick"]
        label = f"meta = $R_{{\\rm {pick}}}$" if pick not in labels_seen else None
        if label:
            labels_seen.add(pick)
        ax.scatter(x, y, c=colors[pick], s=20, edgecolors="white", linewidths=0.4, label=label)
    ax.set_title(r"$R^\star_\Phi$ on the embedding")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure4_wcr_meta_rule.pdf")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    configure_plots()

    print("Generating synthetic population...")
    synthetic_population = gen_synthetic_population()
    print(f"  Generated {len(synthetic_population)} synthetic profiles")

    data: List[Dict[str, object]] = []
    for label, ranks in synthetic_population:
        u = borda_utilities(ranks)
        data.append({
            "culture": label,
            "source": "synthetic",
            "ranks": ranks,
            "u": u,
            "mis": misalignment(u),
            "idx": {
                "diversity": diversity(ranks),
                "agreement": agreement(ranks),
                "polarization": polarization(ranks),
            },
        })

    print("Loading Krakow Pabulib data...")
    pabulib_data: List[Dict[str, object]] = []
    pabulib_sampling_info: Dict[str, Dict[str, object]] = {}
    for fpath in sorted(DATA_DIR.glob("Poland_Krakow_*.pb")):
        year = fpath.stem.split("_")[-1]
        vote_type, votes, num_projects, meta = parse_pb_file(fpath)
        if not votes:
            print(f"  {fpath.name}: skipped, no votes parsed")
            continue
        ranks, sample_info = downsample_and_rank(votes)
        u = borda_utilities(ranks)
        pabulib_data.append({
            "culture": f"Krakow-{year}",
            "source": "pabulib",
            "year": year,
            "ranks": ranks,
            "u": u,
            "mis": misalignment(u),
            "idx": {
                "diversity": diversity(ranks),
                "agreement": agreement(ranks),
                "polarization": polarization(ranks),
            },
            "vote_type": vote_type,
            "original_num_projects": num_projects,
            "original_num_votes": sample_info["original_voters"],
        })
        pabulib_sampling_info[year] = sample_info
        print(f"  Krakow {year}: {sample_info['original_voters']} votes -> {PABULIB_VOTERS} x {PABULIB_ALTS}")

    all_data = data + pabulib_data
    print(f"Total profiles: {len(all_data)} ({len(data)} synthetic + {len(pabulib_data)} Krakow)")

    print("Computing MDS embedding...")
    embed = compute_mds_embedding(all_data)

    synthetic_mu = {
        "egal": np.array([d["mis"]["egal"] for d in data]),
        "util": np.array([d["mis"]["util"] for d in data]),
        "nash": np.array([d["mis"]["nash"] for d in data]),
    }
    ord_indices = {
        "diversity": np.array([d["idx"]["diversity"] for d in data]),
        "agreement": np.array([d["idx"]["agreement"] for d in data]),
        "polarization": np.array([d["idx"]["polarization"] for d in data]),
    }

    print("Computing correlations with bootstrap confidence intervals...")
    correlations: Dict[str, Dict[str, float | str]] = {}
    for m_name, m_vals in synthetic_mu.items():
        for o_name, o_vals in ord_indices.items():
            correlations[f"{m_name}_{o_name}"] = correlation_with_significance(o_vals, m_vals)

    print("Computing robustness across cardinal lifts...")
    lifts: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "Borda": borda_utilities,
        "Positional": positional_utilities,
        "Exponential": exponential_utilities,
    }
    robustness: Dict[str, Dict[str, Dict[str, float]]] = {}
    for lift_name, lift_fn in lifts.items():
        robustness[lift_name] = {}
        lifted_mu = {"egal": [], "util": [], "nash": []}
        for d in data:
            m = misalignment(lift_fn(d["ranks"]))
            for key in lifted_mu:
                lifted_mu[key].append(m[key])
        for m_name in lifted_mu:
            m_vals = np.array(lifted_mu[m_name])
            for o_name, o_vals in ord_indices.items():
                rho, p_val = stats.spearmanr(o_vals, m_vals)
                robustness[lift_name][f"{m_name}_{o_name}"] = {"rho": float(rho), "p_value": float(p_val)}

    print("Computing WCR meta-rule on all profiles...")
    wcr_results = compute_wcr(all_data)

    print("Exporting Map-of-Elections files...")
    export_map_of_elections_files(all_data, embed)

    print("Writing figures...")
    plot_figure1(all_data, embed)
    plot_figure2(synthetic_mu["util"], ord_indices, correlations)
    plot_figure3(robustness)
    plot_figure4(all_data, embed, wcr_results)

    print("Writing result files...")
    p90 = {k: float(np.quantile(v, 0.90)) for k, v in wcr_results.items()}
    medians = {k: float(np.median(v)) for k, v in wcr_results.items()}
    meta_picks = {k: int(sum(1 for d in all_data if d.get("meta_pick") == k)) for k in ["egal", "util", "nash"]}

    q1_pol = np.quantile(ord_indices["polarization"], 0.25)
    q4_pol = np.quantile(ord_indices["polarization"], 0.75)
    q1_util = np.quantile(synthetic_mu["util"], 0.25)
    q4_util = np.quantile(synthetic_mu["util"], 0.75)
    disagreement = (ord_indices["polarization"] <= q1_pol) & (synthetic_mu["util"] >= q4_util)
    mirror_disagreement = (ord_indices["polarization"] >= q4_pol) & (synthetic_mu["util"] <= q1_util)

    numbers = {
        "configuration": {
            "seed": GLOBAL_SEED,
            "n_voters": N_VOTERS,
            "n_alternatives": N_ALTS,
            "n_synthetic": len(data),
            "n_pabulib_krakow": len(pabulib_data),
            "pabulib_downsampling": {
                "n_voters": PABULIB_VOTERS,
                "n_projects": PABULIB_ALTS,
                "seed": PABULIB_SEED,
            },
        },
        "misalignment_ranges_synthetic_borda": {
            key: {
                "min": float(vals.min()),
                "max": float(vals.max()),
                "median": float(np.median(vals)),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
            }
            for key, vals in synthetic_mu.items()
        },
        "krakow_summary": [
            {
                "year": d["year"],
                "vote_type": d["vote_type"],
                "original_num_votes": int(d["original_num_votes"]),
                "original_num_projects": int(d["original_num_projects"]),
                "misalignment": {k: float(d["mis"][k]) for k in ["egal", "util", "nash"]},
            }
            for d in pabulib_data
        ],
        "pabulib_sampling": pabulib_sampling_info,
        "correlations_with_stats_synthetic": correlations,
        "robustness_synthetic": robustness,
        "disagreement_sets_synthetic": {
            "low_polarization_high_util_misalignment": int(disagreement.sum()),
            "high_polarization_low_util_misalignment": int(mirror_disagreement.sum()),
            "n_profiles": int(len(data)),
        },
        "wcr_all_profiles": {
            "medians": medians,
            "p90": p90,
            "meta_picks": meta_picks,
            "p90_reduction_vs_worst_base": float(max(p90["egal"], p90["util"], p90["nash"]) / p90["meta"]) if p90["meta"] > 0 else math.inf,
            "p90_reduction_vs_best_base": float(min(p90["egal"], p90["util"], p90["nash"]) / p90["meta"]) if p90["meta"] > 0 else math.inf,
        },
    }

    with (RESULTS_DIR / "numbers.json").open("w", encoding="utf-8") as f:
        json.dump(numbers, f, indent=2)

    with (RESULTS_DIR / "robustness_table.txt").open("w", encoding="utf-8") as f:
        f.write("Correlation          | Borda    | Positional | Exponential | Max Delta\n")
        f.write("-" * 74 + "\n")
        for cname in [
            "egal_diversity", "egal_agreement", "egal_polarization",
            "util_diversity", "util_agreement", "util_polarization",
            "nash_diversity", "nash_agreement", "nash_polarization",
        ]:
            rhos = [robustness[lift][cname]["rho"] for lift in ["Borda", "Positional", "Exponential"]]
            f.write(f"{cname:20s} | {rhos[0]:+.3f}   | {rhos[1]:+.3f}     | {rhos[2]:+.3f}      | {max(rhos) - min(rhos):.3f}\n")

    print("Done.")
    print(f"Figures written to: {FIG_DIR}")
    print(f"Results written to: {RESULTS_DIR}")
    print(f"Map-of-elections files written to: {MAPEL_DIR}")


if __name__ == "__main__":
    main()
