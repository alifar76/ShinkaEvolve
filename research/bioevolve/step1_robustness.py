"""Step 1: Robustness-Augmented Archive.

Biological concept: canalization / mutational robustness (Wagner 2005). A
genotype sitting on a wide, flat fitness plateau keeps producing viable,
similarly-fit offspring under perturbation; a genotype on a narrow spike
mostly produces worse offspring. Robust genotypes are therefore better
*parents* for continued evolution even when a fragile genotype scores
higher once.

Engine change under test: shinka/database/dbase.py now supports a
"robustness" archive_criteria criterion (std-dev of combined_score across
repeated evaluation runs; lower = more robust). Adding it with a negative
weight makes the archive prefer low-variance solutions as parents, without
touching the champion-tracking logic (get_best_program still tracks the
single highest raw score ever seen -- that distinction matters, see below).

Hypothesis: penalizing high eval-variance in ARCHIVE RETENTION (i.e. in
which programs seed future proposals) steers the population toward the
robust plateau instead of squandering evaluation budget mutating around a
fragile spike whose neighborhood is mostly bad -- yielding a healthier
population (higher mean offspring quality) and a higher long-run best score,
at the cost of sometimes not being the single luckiest noisy peak.

This script runs both configurations across many seeds and reports:
  - best_score_so_far trajectory (checkpoints)
  - mean child true_fitness over the last 25% of generations (population
    health / neighborhood quality being explored)
  - which basin (fragile spike vs robust plateau) the final archive's
    top-ranked program sits in, and its ground-truth local_robustness
"""

import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from landscape import make_landscape  # noqa: E402
from sim import run_simulation  # noqa: E402

NUM_SEEDS = 16
NUM_GENERATIONS = 600
CHECKPOINTS = [100, 200, 400, 600]
DIM = 4


def champion_of_archive(db, dim: int) -> Dict:
    """Pick the archive member ranked best by the engine's own
    rank-based archive_criteria scoring (what actually gets favored as a
    parent), not just raw combined_score."""
    archive = db._get_archive_programs()
    if not archive:
        return {}
    best_p, best_s = None, -float("inf")
    for p in archive:
        rest = [q for q in archive if q.id != p.id]
        s = db._compute_archive_score_ranked(p, rest)
        if s > best_s:
            best_s, best_p = s, p
    x = np.array(best_p.embedding) if best_p.embedding else np.zeros(dim)
    return {
        "id": best_p.id,
        "combined_score": best_p.combined_score,
        "true_fitness": best_p.metadata.get("true_fitness"),
        "eval_score_std": best_p.metadata.get("eval_score_std"),
        "x": x,
    }


def dist_to(x: np.ndarray, center: np.ndarray) -> float:
    return float(np.linalg.norm(x - center))


def run_condition(label: str, archive_criteria: Dict[str, float]) -> Dict:
    per_seed = []
    for seed in range(NUM_SEEDS):
        res = run_simulation(
            archive_criteria=archive_criteria,
            num_generations=NUM_GENERATIONS,
            dim=DIM,
            seed=seed,
        )
        history = res.history
        checkpoint_scores = {
            c: next(r.best_score_so_far for r in history if r.generation == c)
            for c in CHECKPOINTS
        }
        tail = history[-int(NUM_GENERATIONS * 0.25):]
        mean_tail_true_fitness = statistics.fmean(r.child_true_fitness for r in tail)

        champ = champion_of_archive(res.db, DIM)
        fragile_center = np.full(DIM, 0.8)
        robust_center = np.full(DIM, -0.6)
        d_fragile = dist_to(champ["x"], fragile_center)
        d_robust = dist_to(champ["x"], robust_center)
        basin = "fragile" if d_fragile < d_robust else "robust"
        local_rob = res.landscape.local_robustness(
            champ["x"], rng=np.random.default_rng(seed + 1000)
        )

        per_seed.append(
            {
                "seed": seed,
                "checkpoints": checkpoint_scores,
                "mean_tail_true_fitness": mean_tail_true_fitness,
                "champion_true_fitness": champ["true_fitness"],
                "champion_eval_score_std": champ["eval_score_std"],
                "champion_basin": basin,
                "champion_local_robustness": local_rob,
            }
        )
        res.db.close()

    def agg(key, sub=None):
        vals = [
            (r["checkpoints"][sub] if sub is not None else r[key]) for r in per_seed
        ]
        return statistics.fmean(vals), statistics.pstdev(vals)

    summary = {
        "label": label,
        "archive_criteria": archive_criteria,
        "checkpoints": {c: agg(None, c) for c in CHECKPOINTS},
        "mean_tail_true_fitness": agg("mean_tail_true_fitness"),
        "champion_true_fitness": agg("champion_true_fitness"),
        "champion_local_robustness": agg("champion_local_robustness"),
        "fraction_champion_robust_basin": statistics.fmean(
            1.0 if r["champion_basin"] == "robust" else 0.0 for r in per_seed
        ),
        "per_seed": per_seed,
    }
    return summary


def main():
    conditions = [
        ("baseline (combined_score only)", {"combined_score": 1.0}),
        ("robustness-aware (weight -0.4)", {"combined_score": 1.0, "robustness": -0.4}),
        ("robustness-aware (weight -0.8)", {"combined_score": 1.0, "robustness": -0.8}),
    ]

    results = []
    for label, criteria in conditions:
        print(f"Running condition: {label} ...")
        results.append(run_condition(label, criteria))

    header = " ".join(f"best@{c:<9d}" for c in CHECKPOINTS)
    print("\n" + "=" * 100)
    print(f"{'Condition':38s} | {header}   (mean ± stdev over seeds)")
    print("-" * 100)
    for r in results:
        row = "  ".join(
            f"{r['checkpoints'][c][0]:.4f}±{r['checkpoints'][c][1]:.3f}" for c in CHECKPOINTS
        )
        print(f"{r['label']:38s} | {row}")

    print("\n" + "=" * 100)
    print(
        f"{'Condition':38s} | {'tail_true_fit':>15s} {'champ_true_fit':>16s} "
        f"{'champ_robust':>16s} {'%robust_basin':>13s}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r['label']:38s} | "
            f"{r['mean_tail_true_fitness'][0]:6.4f}±{r['mean_tail_true_fitness'][1]:.3f}   "
            f"{r['champion_true_fitness'][0]:6.4f}±{r['champion_true_fitness'][1]:.3f}    "
            f"{r['champion_local_robustness'][0]:6.4f}±{r['champion_local_robustness'][1]:.3f}   "
            f"{r['fraction_champion_robust_basin']*100:12.1f}%"
        )

    out_path = Path(__file__).resolve().parent / "results" / "step1_robustness.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
