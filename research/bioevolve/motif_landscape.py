"""Landscape for testing motif/homology conservation.

Biological concept: deep homology (Hox genes, ATP synthase, the genetic
code itself) -- a small "core" of components is so tightly coupled to
function that almost any change is catastrophic, while everything else
("periphery") tolerates a much wider range of variation and is where most
of the ongoing adaptive tinkering actually happens. Evolution doesn't
"know" in advance which parts are core; it finds out empirically, because
lineages that mutate the core die and lineages that mutate the periphery
mostly survive -- so the surviving population's own variance pattern
reveals which dimensions are conserved.

A subset of dimensions ("core") must sit near a fixed target with a tight
tolerance for fitness to be non-negligible at all (a multiplicative gate);
the remaining dimensions ("periphery") contribute a much more tolerant,
independent bonus on top. Isotropic mutation (equal perturbation on any
randomly chosen dimension) will, once the core has been found, keep
re-breaking it by chance at the same rate it explores the periphery --
pure waste, since the core basically never needs to move again once
found. A motif-aware mutator that reads which dimensions the ARCHIVE has
already converged on (low cross-archive variance = conserved) and shrinks
its own step size there should waste less effort on self-inflicted
core damage while still fully exploring the periphery.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class MotifLandscape:
    dim: int
    core_idx: np.ndarray
    periph_idx: np.ndarray
    core_target: np.ndarray
    core_width: float
    periph_center: np.ndarray
    periph_width: float
    base_floor: float = 0.05

    def true_fitness(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        core_x = x[self.core_idx]
        core_gate = np.exp(
            -np.sum((core_x - self.core_target) ** 2) / (2.0 * self.core_width**2)
        )
        periph_x = x[self.periph_idx]
        periph_score = self.base_floor + (1.0 - self.base_floor) * np.exp(
            -np.sum((periph_x - self.periph_center) ** 2) / (2.0 * self.periph_width**2)
        )
        return float(core_gate * periph_score)

    def noisy_eval(
        self,
        x: np.ndarray,
        num_runs: int,
        rng: np.random.Generator,
        env_sigma: float = 0.05,
        meas_sigma: float = 0.015,
    ) -> np.ndarray:
        dim = x.shape[0] if x.ndim else len(x)
        env_deltas = rng.normal(0.0, env_sigma, size=(num_runs, dim))
        vals = np.array([self.true_fitness(x + d) for d in env_deltas])
        return vals + rng.normal(0.0, meas_sigma, size=num_runs)


def make_motif_landscape(dim: int = 6, seed: int = 0, num_core: int = 3) -> MotifLandscape:
    rng = np.random.default_rng(seed)
    core_idx = np.arange(num_core)
    periph_idx = np.arange(num_core, dim)
    core_target = np.full(num_core, 0.6)
    periph_center = rng.uniform(-0.7, 0.7, size=dim - num_core)
    return MotifLandscape(
        dim=dim,
        core_idx=core_idx,
        periph_idx=periph_idx,
        core_target=core_target,
        core_width=0.14,
        periph_center=periph_center,
        periph_width=0.5,
    )


def motif_aware_mutate(
    parent_x: np.ndarray,
    rng: np.random.Generator,
    dim: int,
    patch_type: str = "diff",
    cross_partner: Optional[np.ndarray] = None,
    step_scale: float = 0.07,
    bounds: float = 1.6,
    db: Any = None,
    min_archive: int = 8,
    protect_frac: float = 0.4,
    protect_factor: float = 0.15,
    boost_factor: float = 1.4,
) -> np.ndarray:
    """Same patch-type structure as landscape.mutate, but with a per-
    dimension step size derived from the archive's own observed variance:
    the `protect_frac` fraction of dimensions with the LOWEST variance
    across current archive members (empirically conserved) get their step
    size shrunk by `protect_factor`; the rest get boosted by
    `boost_factor` so total exploration effort is redirected, not just
    reduced. Falls back to plain isotropic mutation until the archive has
    at least `min_archive` members with embeddings (nothing to condition
    on yet)."""
    x = np.array(parent_x, dtype=float, copy=True)
    per_dim_scale = np.full(dim, step_scale)

    if db is not None:
        archive = db._get_archive_programs()
        embeds = [
            p.embedding for p in archive if p.embedding and len(p.embedding) == dim
        ]
        if len(embeds) >= min_archive:
            var = np.var(np.array(embeds), axis=0)
            order = np.argsort(var)
            n_protect = max(1, int(dim * protect_frac))
            conserved_dims = order[:n_protect]
            variable_dims = order[n_protect:]
            per_dim_scale[conserved_dims] *= protect_factor
            per_dim_scale[variable_dims] *= boost_factor

    if patch_type == "diff":
        k = int(rng.integers(1, max(2, dim // 2 + 1)))
        idx = rng.choice(dim, size=k, replace=False)
        x[idx] += rng.normal(0.0, 1.0, size=k) * per_dim_scale[idx]
    elif patch_type == "full":
        x = x + rng.normal(0.0, 1.0, size=dim) * (per_dim_scale * 5)
    elif patch_type == "cross" and cross_partner is not None:
        mask = rng.random(dim) < 0.5
        x = np.where(mask, x, np.asarray(cross_partner, dtype=float))
    else:
        x = x + rng.normal(0.0, 1.0, size=dim) * per_dim_scale
    return np.clip(x, -bounds, bounds)
