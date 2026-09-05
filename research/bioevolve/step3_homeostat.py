"""Step 3: Homeostatic Feedback Controller -- does closing the loop help?

Compares three conditions, all using the SAME baseline archive_criteria
(combined_score only, i.e. Step 1's robustness weighting is deliberately
left OFF here so we isolate Step 3's own contribution):

  1. baseline        -- fixed step_scale / migration_rate (sim.py defaults)
  2. static-explore   -- fixed but PERMANENTLY HIGHER step_scale/migration
                         (a control condition: is "more exploration all the
                         time" just as good as reacting to the diversity
                         signal, or does timing matter?)
  3. homeostat        -- the Step 2/3 negative-feedback controller: reacts
                         to falling diversity (and compounds with recent
                         fragility) by temporarily raising step_scale and
                         migration_rate, then relaxing back down.

Same landscape and generation budget as Step 1 for comparability.
"""

import statistics
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from homeostat import make_homeostat  # noqa: E402
from sim import run_simulation  # noqa: E402

NUM_SEEDS = 16
NUM_GENERATIONS = 600
CHECKPOINTS = [100, 200, 400, 600]
DIM = 4
STAGNATION_THRESHOLD = 40  # gens without improvement counted as "stagnating"


def gens_since_improvement_series(best_so_far: np.ndarray) -> np.ndarray:
    counter = np.zeros_like(best_so_far)
    since = 0
    prev = best_so_far[0]
    for i, b in enumerate(best_so_far):
        if b > prev + 1e-9:
            since = 0
            prev = b
        else:
            since += 1
        counter[i] = since
    return counter


def run_condition(label: str, run_kwargs: Dict) -> Dict:
    per_seed = []
    for seed in range(NUM_SEEDS):
        kwargs = dict(run_kwargs)
        controller_factory = kwargs.pop("controller_factory", None)
        controller = controller_factory() if controller_factory else None
        res = run_simulation(
            archive_criteria={"combined_score": 1.0},
            num_generations=NUM_GENERATIONS,
            dim=DIM,
            seed=seed,
            controller=controller,
            **kwargs,
        )
        history = res.history
        best_so_far = np.array([r.best_score_so_far for r in history])
        checkpoint_scores = {
            c: next(r.best_score_so_far for r in history if r.generation == c)
            for c in CHECKPOINTS
        }
        tail = history[-int(NUM_GENERATIONS * 0.25):]
        mean_tail_true_fitness = statistics.fmean(r.child_true_fitness for r in tail)

        since_series = gens_since_improvement_series(best_so_far)
        frac_stagnating = float(np.mean(since_series >= STAGNATION_THRESHOLD))
        longest_stagnation = int(since_series.max())

        per_seed.append(
            {
                "seed": seed,
                "checkpoints": checkpoint_scores,
                "mean_tail_true_fitness": mean_tail_true_fitness,
                "frac_stagnating": frac_stagnating,
                "longest_stagnation": longest_stagnation,
            }
        )
        res.db.close()

    def agg(key, sub=None):
        vals = [(r["checkpoints"][sub] if sub is not None else r[key]) for r in per_seed]
        return statistics.fmean(vals), statistics.pstdev(vals)

    return {
        "label": label,
        "checkpoints": {c: agg(None, c) for c in CHECKPOINTS},
        "mean_tail_true_fitness": agg("mean_tail_true_fitness"),
        "frac_stagnating": agg("frac_stagnating"),
        "longest_stagnation": agg("longest_stagnation"),
        "per_seed": per_seed,
    }


def main():
    conditions = [
        ("1. baseline (fixed step/migration)", dict(step_scale=0.07, db_kwargs={"migration_rate": 0.1})),
        (
            "2. static-explore (always high)",
            dict(step_scale=0.15, db_kwargs={"migration_rate": 0.25}),
        ),
        (
            "3. homeostat (reactive)",
            dict(step_scale=0.07, db_kwargs={"migration_rate": 0.1}, controller_factory=make_homeostat),
        ),
    ]

    results = [run_condition(label, kwargs) for label, kwargs in conditions]

    header = " ".join(f"best@{c:<9d}" for c in CHECKPOINTS)
    print("\n" + "=" * 105)
    print(f"{'Condition':38s} | {header}")
    print("-" * 105)
    for r in results:
        row = "  ".join(
            f"{r['checkpoints'][c][0]:.4f}±{r['checkpoints'][c][1]:.3f}" for c in CHECKPOINTS
        )
        print(f"{r['label']:38s} | {row}")

    print("\n" + "=" * 105)
    print(
        f"{'Condition':38s} | {'tail_true_fit':>15s} {'%time_stagnating':>17s} "
        f"{'longest_stagnation':>19s}"
    )
    print("-" * 105)
    for r in results:
        print(
            f"{r['label']:38s} | "
            f"{r['mean_tail_true_fitness'][0]:6.4f}±{r['mean_tail_true_fitness'][1]:.3f}    "
            f"{r['frac_stagnating'][0]*100:6.1f}%±{r['frac_stagnating'][1]*100:5.1f}pp        "
            f"{r['longest_stagnation'][0]:6.1f}±{r['longest_stagnation'][1]:.1f}"
        )


if __name__ == "__main__":
    main()
