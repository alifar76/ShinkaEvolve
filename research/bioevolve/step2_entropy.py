"""Step 2: Entropy-Flux Instrumentation (passive, no engine-behavior change).

Biological concept: living systems are dissipative structures maintained
far from thermodynamic equilibrium (Prigogine) -- they stay organized only
by continuously importing variation/free-energy and exporting entropy. Left
alone, any such system relaxes toward equilibrium: for an evolving
population, "equilibrium" is a genetically collapsed population where
everyone is a near-copy of the current best -- selection continuously
consumes diversity, and only ongoing mutation/migration replenishes it. If
that replenishment can't keep pace with selection's pull, diversity
collapses -- and once it has, there is no more raw material nearby for
further improvement, i.e. stagnation.

This script does NOT change the engine. It adds a passive diversity
observable (shinka/database/dbase.py: ProgramDatabase.get_population_
diversity, read-only, no effect on sampling/archiving) and asks a factual
question with the baseline engine: does a drop in population diversity
actually precede stagnation (a long gap until the next best-score
improvement), or not? If it doesn't, building a controller around this
signal in Step 3 would be chasing a non-effect.
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim import run_simulation  # noqa: E402

NUM_SEEDS = 20
NUM_GENERATIONS = 500
DIVERSITY_WINDOW = 25  # generations of recent history used per diversity reading
SAMPLE_EVERY = 5  # how often (generations) to take a diversity reading
DIM = 4


def collect_series(seed: int) -> Dict[str, np.ndarray]:
    diversity_readings: List[tuple] = []  # (generation, diversity)
    best_so_far: List[float] = []

    def on_gen(gen, db, record):
        best_so_far.append(record.best_score_so_far)
        if gen % SAMPLE_EVERY == 0:
            since = max(0, gen - DIVERSITY_WINDOW)
            div = db.get_population_diversity(since_generation=since, metric="variance")
            diversity_readings.append((gen, div if div is not None else np.nan))

    res = run_simulation(
        archive_criteria={"combined_score": 1.0},
        num_generations=NUM_GENERATIONS,
        dim=DIM,
        seed=seed,
        on_generation=on_gen,
    )
    res.db.close()

    gens = np.array([g for g, _ in diversity_readings], dtype=float)
    divs = np.array([d for _, d in diversity_readings], dtype=float)
    best = np.array(best_so_far, dtype=float)  # indexed by generation-1
    return {"gens": gens, "diversity": divs, "best_so_far": best}


def generations_until_next_improvement(best_so_far: np.ndarray, at_gen: int) -> float:
    """From generation `at_gen` (1-indexed), how many generations until
    best_so_far next increases? Right-censored at run end (returns the
    remaining run length as a lower bound, marked via the censored flag
    by the caller)."""
    idx = at_gen - 1
    current_best = best_so_far[idx]
    for j in range(idx + 1, len(best_so_far)):
        if best_so_far[j] > current_best + 1e-9:
            return j - idx
    return len(best_so_far) - idx  # censored: no improvement before run end


def main():
    diversity_vals = []
    stagnation_ahead_vals = []
    censored_flags = []

    for seed in range(NUM_SEEDS):
        series = collect_series(seed)
        for gen, div in zip(series["gens"], series["diversity"]):
            gen = int(gen)
            if np.isnan(div) or gen >= NUM_GENERATIONS:
                continue
            wait = generations_until_next_improvement(series["best_so_far"], gen)
            censored = wait == (len(series["best_so_far"]) - (gen - 1))
            diversity_vals.append(div)
            stagnation_ahead_vals.append(wait)
            censored_flags.append(censored)
        print(f"seed {seed}: collected {len(series['gens'])} diversity readings")

    diversity_vals = np.array(diversity_vals)
    stagnation_ahead_vals = np.array(stagnation_ahead_vals)
    censored_flags = np.array(censored_flags)

    # Uncensored-only analysis (the honest test: excludes points where we
    # don't actually know when the next improvement would have happened).
    uncensored = ~censored_flags
    rho, pval = stats.spearmanr(
        diversity_vals[uncensored], stagnation_ahead_vals[uncensored]
    )

    print("\n" + "=" * 70)
    print("Entropy-flux vs. stagnation-ahead: passive baseline diagnostic")
    print("=" * 70)
    print(f"Total (generation, diversity) samples: {len(diversity_vals)}")
    print(f"  of which right-censored (no improvement before run end): "
          f"{censored_flags.sum()} ({100*censored_flags.mean():.1f}%)")
    print(
        f"\nSpearman correlation (diversity vs. generations-until-next-"
        f"improvement), uncensored points only:"
    )
    print(f"  rho = {rho:.3f}   p = {pval:.2e}   n = {uncensored.sum()}")
    print(
        "\nInterpretation: rho < 0 and significant means LOWER diversity "
        "precedes a LONGER wait for the next improvement -- i.e. entropy "
        "collapse is a leading indicator of stagnation, supporting a "
        "Step 3 controller that reacts to it. rho ~ 0 or positive would "
        "mean the signal isn't predictive here."
    )

    # Split into low/high diversity halves for an effect-size sanity check.
    median_div = np.median(diversity_vals[uncensored])
    low = stagnation_ahead_vals[uncensored][diversity_vals[uncensored] <= median_div]
    high = stagnation_ahead_vals[uncensored][diversity_vals[uncensored] > median_div]
    print(
        f"\nMean generations-to-next-improvement: "
        f"low-diversity half = {low.mean():.2f} (n={len(low)}), "
        f"high-diversity half = {high.mean():.2f} (n={len(high)})"
    )
    u_stat, u_p = stats.mannwhitneyu(low, high, alternative="greater")
    print(f"Mann-Whitney U (low > high): U={u_stat:.0f}, p={u_p:.2e}")


if __name__ == "__main__":
    main()
