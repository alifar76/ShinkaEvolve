"""Step 4: Motif Conservation Tracking.

Compares isotropic mutation (baseline `landscape.mutate` -- every
dimension gets the same step size, exactly as in Steps 1-3) against
`motif_aware_mutate` (reads the archive's own per-dimension variance and
shrinks step size on dimensions the archive has already converged on),
on a landscape where a "core" subset of dimensions gates fitness sharply
and a "peripheral" subset is far more tolerant (motif_landscape.py).

Metrics:
  - generations to first reach true_fitness >= COMPETENCE_THRESHOLD
    ("time to find the core")
  - catastrophic mutation rate AFTER competence is first reached: fraction
    of children whose true_fitness is less than REGRESSION_FRACTION of
    their parent's true_fitness (a proxy for "broke the core by accident")
  - final best true_fitness reached
"""

import statistics
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from landscape import mutate as isotropic_mutate  # noqa: E402
from motif_landscape import make_motif_landscape, motif_aware_mutate  # noqa: E402
from sim import run_simulation  # noqa: E402

NUM_SEEDS = 16
NUM_GENERATIONS = 900
DIM = 6
NUM_CORE = 3
COMPETENCE_THRESHOLD = 0.4
REGRESSION_FRACTION = 0.3


def motif_landscape_factory(dim: int, seed: int):
    return make_motif_landscape(dim=dim, seed=seed, num_core=NUM_CORE)


def time_to_competence(history) -> Optional[int]:
    for r in history:
        if r.child_true_fitness >= COMPETENCE_THRESHOLD:
            return r.generation
    return None


def catastrophic_rate_after(db, first_competent_gen: int) -> float:
    """Fraction of post-competence children whose true_fitness collapsed
    to less than REGRESSION_FRACTION of their PARENT's true_fitness."""
    events = 0
    total = 0
    programs = {p.id: p for p in db.get_all_programs()}
    for p in programs.values():
        if p.generation <= first_competent_gen or p.parent_id is None:
            continue
        parent = programs.get(p.parent_id)
        if parent is None:
            continue
        parent_fit = parent.metadata.get("true_fitness")
        child_fit = p.metadata.get("true_fitness")
        if parent_fit is None or child_fit is None or parent_fit < COMPETENCE_THRESHOLD:
            continue
        total += 1
        if child_fit < REGRESSION_FRACTION * parent_fit:
            events += 1
    return events / total if total else float("nan")


def run_condition(label: str, mutate_fn) -> Dict:
    per_seed = []
    for seed in range(NUM_SEEDS):
        res = run_simulation(
            archive_criteria={"combined_score": 1.0},
            num_generations=NUM_GENERATIONS,
            dim=DIM,
            seed=seed,
            landscape_factory=motif_landscape_factory,
            mutate_fn=mutate_fn,
        )
        history = res.history
        t_comp = time_to_competence(history)
        cat_rate = (
            catastrophic_rate_after(res.db, t_comp)
            if t_comp is not None
            else float("nan")
        )
        final_true_fit = history[-1].best_score_so_far
        per_seed.append(
            {
                "seed": seed,
                "time_to_competence": t_comp if t_comp is not None else NUM_GENERATIONS,
                "reached_competence": t_comp is not None,
                "catastrophic_rate": cat_rate,
                "final_best": final_true_fit,
            }
        )
        res.db.close()

    reached = [r for r in per_seed if r["reached_competence"]]
    cat_rates = [r["catastrophic_rate"] for r in reached if not np.isnan(r["catastrophic_rate"])]

    def agg(vals):
        return (statistics.fmean(vals), statistics.pstdev(vals)) if vals else (float("nan"), 0.0)

    return {
        "label": label,
        "frac_reached_competence": len(reached) / len(per_seed),
        "time_to_competence": agg([r["time_to_competence"] for r in per_seed]),
        "catastrophic_rate": agg(cat_rates),
        "final_best": agg([r["final_best"] for r in per_seed]),
        "per_seed": per_seed,
    }


def main():
    gated_motif_mutate = partial(motif_aware_mutate, activation_threshold=0.15)
    conditions = [
        ("isotropic (baseline)", isotropic_mutate),
        ("motif-aware (naive)", motif_aware_mutate),
        ("motif-aware (gated)", gated_motif_mutate),
    ]
    results = [run_condition(label, fn) for label, fn in conditions]

    print("\n" + "=" * 95)
    print(
        f"{'Condition':24s} | {'%reached comp.':>14s} {'time_to_comp':>15s} "
        f"{'catastrophic_rate':>18s} {'final_best':>13s}"
    )
    print("-" * 95)
    for r in results:
        print(
            f"{r['label']:24s} | "
            f"{r['frac_reached_competence']*100:13.1f}% "
            f"{r['time_to_competence'][0]:8.1f}±{r['time_to_competence'][1]:.1f}   "
            f"{r['catastrophic_rate'][0]*100:9.2f}%±{r['catastrophic_rate'][1]*100:5.2f}pp   "
            f"{r['final_best'][0]:6.4f}±{r['final_best'][1]:.3f}"
        )


if __name__ == "__main__":
    main()
