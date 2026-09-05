"""Rigorous, honest benchmark: HomeoCMA (the unified engine) vs. plain
isotropic mutation vs. the best single-mechanism variants already
validated in Steps 1-4, on BOTH testbeds, same generation budgets and seed
counts used throughout this research so results are directly comparable
to the earlier step1-4 numbers.

One HomeoCMA configuration (mu=6, lambda_robust=0.4, activation_threshold
=0.1) is used unchanged on both landscapes -- the point of unifying Steps
1-4 into one engine is that it should NOT need per-landscape hand-tuning.
"""

import statistics
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from homeocma import make_homeocma_mutate  # noqa: E402
from landscape import make_landscape, mutate as isotropic_mutate  # noqa: E402
from motif_landscape import make_motif_landscape, motif_aware_mutate  # noqa: E402
from sim import run_simulation  # noqa: E402

NUM_SEEDS = 16


def homeocma_factory(dim: int):
    return lambda: make_homeocma_mutate(
        dim=dim, mu=6, sigma0=0.07, lambda_robust=0.4, activation_threshold=0.1
    )[0]


# ---------------------------------------------------------------------------
# Landscape A: fragile-trap vs. robust-plateau (dim=4), same setup as Step 1/3
# ---------------------------------------------------------------------------
DIM_A = 4
NUM_GENERATIONS_A = 600
CHECKPOINTS_A = [100, 200, 400, 600]


def run_condition_a(label: str, mutate_fn_factory) -> Dict:
    per_seed = []
    for seed in range(NUM_SEEDS):
        mutate_fn = mutate_fn_factory() if mutate_fn_factory else isotropic_mutate
        res = run_simulation(
            archive_criteria={"combined_score": 1.0},
            num_generations=NUM_GENERATIONS_A,
            dim=DIM_A,
            seed=seed,
            landscape_factory=lambda d, s: make_landscape(dim=d, seed=s),
            mutate_fn=mutate_fn,
        )
        history = res.history
        checkpoint_scores = {
            c: next(r.best_score_so_far for r in history if r.generation == c)
            for c in CHECKPOINTS_A
        }
        tail = history[-int(NUM_GENERATIONS_A * 0.25):]
        mean_tail_true_fitness = statistics.fmean(r.child_true_fitness for r in tail)
        per_seed.append({"checkpoints": checkpoint_scores, "tail": mean_tail_true_fitness})
        res.db.close()

    def agg(key, sub=None):
        vals = [(r["checkpoints"][sub] if sub is not None else r[key]) for r in per_seed]
        return statistics.fmean(vals), statistics.pstdev(vals)

    return {
        "label": label,
        "checkpoints": {c: agg(None, c) for c in CHECKPOINTS_A},
        "tail": agg("tail"),
    }


def print_landscape_a(results: List[Dict]):
    header = " ".join(f"best@{c:<9d}" for c in CHECKPOINTS_A)
    print("\n" + "=" * 100)
    print("LANDSCAPE A: fragile-trap vs. robust-plateau (dim=4, 600 gens)")
    print("=" * 100)
    print(f"{'Condition':30s} | {header}   tail_true_fit")
    print("-" * 100)
    for r in results:
        row = "  ".join(
            f"{r['checkpoints'][c][0]:.4f}±{r['checkpoints'][c][1]:.3f}" for c in CHECKPOINTS_A
        )
        print(f"{r['label']:30s} | {row}   {r['tail'][0]:.4f}±{r['tail'][1]:.3f}")


# ---------------------------------------------------------------------------
# Landscape B: core/periphery motif (dim=6), same setup as Step 4
# ---------------------------------------------------------------------------
DIM_B = 6
NUM_CORE_B = 3
NUM_GENERATIONS_B = 900
COMPETENCE_THRESHOLD_B = 0.4
REGRESSION_FRACTION_B = 0.3


def motif_landscape_factory(dim: int, seed: int):
    return make_motif_landscape(dim=dim, seed=seed, num_core=NUM_CORE_B)


def time_to_competence(history):
    for r in history:
        if r.child_true_fitness >= COMPETENCE_THRESHOLD_B:
            return r.generation
    return None


def catastrophic_rate_after(db, first_competent_gen: int) -> float:
    events, total = 0, 0
    programs = {p.id: p for p in db.get_all_programs()}
    for p in programs.values():
        if p.generation <= first_competent_gen or p.parent_id is None:
            continue
        parent = programs.get(p.parent_id)
        if parent is None:
            continue
        parent_fit = parent.metadata.get("true_fitness")
        child_fit = p.metadata.get("true_fitness")
        if parent_fit is None or child_fit is None or parent_fit < COMPETENCE_THRESHOLD_B:
            continue
        total += 1
        if child_fit < REGRESSION_FRACTION_B * parent_fit:
            events += 1
    return events / total if total else float("nan")


def run_condition_b(label: str, mutate_fn_factory) -> Dict:
    per_seed = []
    for seed in range(NUM_SEEDS):
        mutate_fn = mutate_fn_factory() if mutate_fn_factory else isotropic_mutate
        res = run_simulation(
            archive_criteria={"combined_score": 1.0},
            num_generations=NUM_GENERATIONS_B,
            dim=DIM_B,
            seed=seed,
            landscape_factory=motif_landscape_factory,
            mutate_fn=mutate_fn,
        )
        history = res.history
        t_comp = time_to_competence(history)
        cat_rate = catastrophic_rate_after(res.db, t_comp) if t_comp is not None else float("nan")
        per_seed.append(
            {
                "reached": t_comp is not None,
                "time_to_competence": t_comp if t_comp is not None else NUM_GENERATIONS_B,
                "catastrophic_rate": cat_rate,
                "final_best": history[-1].best_score_so_far,
            }
        )
        res.db.close()

    reached = [r for r in per_seed if r["reached"]]
    cat_rates = [r["catastrophic_rate"] for r in reached if not np.isnan(r["catastrophic_rate"])]

    def agg(vals):
        return (statistics.fmean(vals), statistics.pstdev(vals)) if vals else (float("nan"), 0.0)

    return {
        "label": label,
        "frac_reached": len(reached) / len(per_seed),
        "time_to_competence": agg([r["time_to_competence"] for r in per_seed]),
        "catastrophic_rate": agg(cat_rates),
        "final_best": agg([r["final_best"] for r in per_seed]),
    }


def print_landscape_b(results: List[Dict]):
    print("\n" + "=" * 100)
    print("LANDSCAPE B: core/periphery motif conservation (dim=6, num_core=3, 900 gens)")
    print("=" * 100)
    print(
        f"{'Condition':30s} | {'%reached comp.':>14s} {'time_to_comp':>15s} "
        f"{'catastrophic_rate':>18s} {'final_best':>13s}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r['label']:30s} | "
            f"{r['frac_reached']*100:13.1f}% "
            f"{r['time_to_competence'][0]:8.1f}±{r['time_to_competence'][1]:.1f}   "
            f"{r['catastrophic_rate'][0]*100:9.2f}%±{r['catastrophic_rate'][1]*100:5.2f}pp   "
            f"{r['final_best'][0]:6.4f}±{r['final_best'][1]:.3f}"
        )


def main():
    conditions_a = [
        ("isotropic (baseline)", None),
        ("HomeoCMA (unified)", homeocma_factory(DIM_A)),
    ]
    results_a = [run_condition_a(label, fn) for label, fn in conditions_a]
    print_landscape_a(results_a)

    gated_motif_mutate = partial(motif_aware_mutate, activation_threshold=0.15)
    conditions_b = [
        ("isotropic (baseline)", None),
        ("motif-aware (gated, Step 4)", lambda: gated_motif_mutate),
        ("HomeoCMA (unified)", homeocma_factory(DIM_B)),
    ]
    results_b = [run_condition_b(label, fn) for label, fn in conditions_b]
    print_landscape_b(results_b)


if __name__ == "__main__":
    main()
