"""Synthetic fitness landscape for testing bio-inspired engine mechanisms.

Standard technique from the robustness/evolvability literature (Wagner 2005,
2008): model "genotypes" as points in a continuous space and build a
landscape with a tradeoff between peak height and basin width. A narrow,
tall peak is a *fragile* optimum -- small perturbations (mutation, or noisy
re-evaluation) fall off it fast. A wide, slightly-lower plateau is a
*robust* optimum -- it stays good under perturbation. This mirrors how a
fragile shortcut in code (relies on undocumented invariants) can score well
once but degrades under minor edits or noisy inputs, versus a robust
implementation that keeps working under small changes.

This stands in for ShinkaEvolve's real (LLM writes code -> evaluate.py
scores it) loop so we can run thousands of cheap, reproducible generations
and isolate the effect of *database/selection* mechanisms (what this
research targets) from LLM proposal quality (a large, expensive confound).
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass
class Bump:
    center: np.ndarray
    height: float
    width: float  # larger = flatter/wider = more robust


@dataclass
class Landscape:
    dim: int
    bumps: List[Bump] = field(default_factory=list)

    def true_fitness(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        val = 0.0
        for b in self.bumps:
            d2 = float(np.sum((x - b.center) ** 2))
            val += b.height * np.exp(-d2 / (2.0 * b.width**2))
        return val

    def local_robustness(
        self,
        x: np.ndarray,
        radius: float = 0.06,
        n_samples: int = 40,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Mean fitness retained under small random perturbation, as a
        fraction of the fitness at x. 1.0 = perfectly flat/robust locally;
        close to 0 = sitting on a narrow, fragile spike."""
        rng = rng or np.random.default_rng()
        base = self.true_fitness(x)
        if base <= 1e-9:
            return 0.0
        deltas = rng.normal(0.0, radius, size=(n_samples, self.dim))
        vals = [self.true_fitness(x + d) for d in deltas]
        return float(np.mean(vals) / base)

    def noisy_eval(
        self,
        x: np.ndarray,
        num_runs: int,
        rng: np.random.Generator,
        env_sigma: float = 0.05,
        meas_sigma: float = 0.015,
    ) -> np.ndarray:
        """Simulates repeated stochastic evaluation of the same genome,
        e.g. re-running a stochastic policy across different random seeds
        or a benchmark with varying conditions.

        Each run perturbs the genome by a small "environmental" offset
        (env_sigma) before scoring -- this is what makes variance across
        repeated runs an actual proxy for local basin width/robustness: a
        genome on a narrow spike swings wildly under a tiny perturbation,
        one on a wide plateau barely moves. A small amount of pure
        measurement noise (meas_sigma) is added on top for realism. Flat
        (position-independent) noise alone would NOT carry this signal,
        since it's identical everywhere on the landscape.
        """
        dim = x.shape[0] if x.ndim else len(x)
        env_deltas = rng.normal(0.0, env_sigma, size=(num_runs, dim))
        vals = np.array([self.true_fitness(x + d) for d in env_deltas])
        return vals + rng.normal(0.0, meas_sigma, size=num_runs)


def make_landscape(dim: int = 6, seed: int = 0) -> Landscape:
    """One tall, narrow (fragile) optimum placed CLOSE to the start so an
    unguided search finds it early; one slightly-lower, wide (robust)
    optimum placed further away so it's found later, if at all, without
    active protection. This mirrors a common real trap: an early "quick
    win" edit that overfits/relies on a brittle shortcut vs. a more solid
    rewrite that takes longer to arrive at. Several smaller decoy bumps
    add ruggedness."""
    rng = np.random.default_rng(seed)

    fragile_center = np.full(dim, 0.35)
    robust_center = np.full(dim, -0.75)

    bumps = [
        Bump(center=fragile_center, height=1.00, width=0.05),
        Bump(center=robust_center, height=0.88, width=0.38),
    ]
    for _ in range(4):
        c = rng.uniform(-1.1, 1.1, size=dim)
        h = rng.uniform(0.20, 0.45)
        w = rng.uniform(0.08, 0.18)
        bumps.append(Bump(center=c, height=h, width=w))

    return Landscape(dim=dim, bumps=bumps)


def mutate(
    parent_x: np.ndarray,
    rng: np.random.Generator,
    dim: int,
    patch_type: str = "diff",
    cross_partner: Optional[np.ndarray] = None,
    step_scale: float = 0.07,
    bounds: float = 1.6,
    db: Any = None,
) -> np.ndarray:
    """Synthetic stand-in for an LLM-authored code patch, mirroring
    ShinkaEvolve's patch_types: 'diff' = small localized edit, 'full' =
    larger rewrite, 'cross' = combine traits from two parents/inspirations.

    `db` is accepted and ignored here (isotropic mutation doesn't use
    population state); it's part of the shared mutate_fn interface so a
    motif-aware mutator (Step 4) can read archive statistics through the
    same call site in sim.py without a special case.
    """
    x = np.array(parent_x, dtype=float, copy=True)
    if patch_type == "diff":
        k = int(rng.integers(1, max(2, dim // 2 + 1)))
        idx = rng.choice(dim, size=k, replace=False)
        x[idx] += rng.normal(0.0, step_scale, size=k)
    elif patch_type == "full":
        x = x + rng.normal(0.0, step_scale * 5, size=dim)
    elif patch_type == "cross" and cross_partner is not None:
        mask = rng.random(dim) < 0.5
        x = np.where(mask, x, np.asarray(cross_partner, dtype=float))
    else:
        x = x + rng.normal(0.0, step_scale, size=dim)
    return np.clip(x, -bounds, bounds)
