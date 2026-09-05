"""HomeoCMA: one compact, principled regulator unifying the four
bio-inspired mechanisms explored in Steps 1-4.

Steps 1-4 each proposed a separate heuristic:
  1. penalize high eval-variance in archive retention (canalization)
  2. observe population diversity as an "entropy" signal
  3. react to diversity drops by boosting mutation/migration (homeostasis)
  4. protect low-variance archive dimensions from mutation (conservation)

Formalized correctly, these are not four separate ideas -- they are the
four faces of Evolution Strategies' CMA-ES (Hansen & Ostermeier), one of
the most theoretically grounded and empirically dominant continuous
black-box optimizers known, itself explicitly modeled on biological
evolution:

  - risk-adjusted (robust) fitness  <-> rank-based selection
  - population diversity            <-> the evolution-path / step-size state
  - homeostatic step-size reaction  <-> Cumulative Step-size Adaptation (CSA)
  - per-dimension conservation      <-> (diagonal / separable) covariance
                                         matrix adaptation

Step 4's failure (naive per-dimension variance tracking mistook genetic
drift for real conservation before any selection signal existed) is fixed
STRUCTURALLY here, not with a manually chosen activation threshold: CMA-ES
weights the top-mu archive members by fitness RANK. When there is no real
fitness signal yet, ranking is close to arbitrary and the weighted update
stays close to an uninformative, near-isotropic prior; once real signal
appears, the same update sharpens automatically around what elites
actually agree on. The gate emerges from the math instead of being bolted
on.

This module is a *separable* (diagonal-covariance) CMA-style regulator:
O(d) state and O(d) work per step, not O(d^2) -- deliberately compact and
cheap, since the target use case is regulating an LLM-driven, archive/
island-structured code-evolution loop, not solving a textbook continuous
optimization benchmark. The novel contribution here is not the ES math
itself (well-established) but applying it as the statistical "how much
and where to vary" layer underneath an LLM's "what to try" proposals,
reading its ranking/covariance signal directly from ShinkaEvolve's real
archive rather than from a private single-point search state.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


def log_weights(mu: int) -> np.ndarray:
    """Standard CMA-ES recombination weights: log-linear, favor the best,
    sum to 1. mu_eff = 1 / sum(w^2) is the 'effective number of parents'
    actually driving the update."""
    raw = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
    return raw / raw.sum()


@dataclass
class HomeoCMA:
    dim: int
    mu: int = 6
    sigma0: float = 0.07
    sigma_min: float = 0.01
    sigma_max: float = 0.5
    target_success: float = 0.2  # Rechenberg's 1/5 rule
    k_sigma: float = 0.15
    lambda_robust: float = 0.4
    c_cov: float = 0.2  # covariance learning rate (damping, not full replacement)
    eps: float = 1e-3
    activation_threshold: Optional[float] = None

    mean: Optional[np.ndarray] = None
    c: Optional[np.ndarray] = None  # diagonal covariance ("conservation" state)
    sigma: Optional[float] = None
    diversity: Optional[float] = None  # sum(c); logged/diagnostic (Step 2), not a control input

    def __post_init__(self):
        if self.mean is None:
            self.mean = np.zeros(self.dim)
        if self.c is None:
            self.c = np.ones(self.dim)
        if self.sigma is None:
            self.sigma = self.sigma0

    def robust_fitness(self, score: float, score_std: float) -> float:
        """Step 1: canalization-adjusted fitness used for ranking only."""
        return score - self.lambda_robust * score_std

    def _has_signal(self, best_score: Optional[float]) -> bool:
        if self.activation_threshold is None:
            return True
        return best_score is not None and best_score >= self.activation_threshold

    def update(self, elite_x: np.ndarray, best_score: Optional[float] = None) -> None:
        """elite_x: (mu, dim) genomes of the top-mu archive members ranked
        best-first by robust_fitness. Updates mean and covariance (Steps
        1+4). Step-size (Steps 2+3) is handled separately by
        `adapt_sigma`, driven by an absolute, bounded reference (mutation
        success rate) rather than a self-referential moving diversity
        target -- see adapt_sigma's docstring for why.

        The mean always updates (drifting toward wherever the archive's
        current ranking points, even under noisy/tied fitness, is
        harmless). The COVARIANCE update is what needs `best_score` to
        clear `activation_threshold` first: on a landscape with a large
        near-zero-gradient region before any real optimum is found (see
        motif_landscape.py), elites are fitness-tied and their variance
        reflects pure sampling noise, not function -- shrinking C on that
        noise cements an arbitrary, likely-wrong bottleneck (this
        reproduces, at the covariance level, the exact drift-vs-signal
        confound diagnosed for the standalone motif-conservation
        mechanism in Step 4). Biologically: canalization around a trait
        only makes sense once a lineage has a working phenotype to
        canalize -- there is nothing to conserve before that."""
        mu = elite_x.shape[0]
        if mu < 2:
            return
        w = log_weights(mu)

        deviations = elite_x - self.mean  # deviations from the OLD mean
        self.mean = w @ elite_x
        if not self._has_signal(best_score):
            return
        new_estimate = w @ (deviations**2)
        # Damped blend, not full replacement: real CMA-ES's covariance
        # learning rate exists precisely because replacing C outright with
        # a single generation's (noisy, small-mu) estimate is unstable --
        # a tight-by-chance elite cluster would shrink C to ~0, which then
        # produces even-tighter children next round, a runaway collapse.
        self.c = np.maximum((1.0 - self.c_cov) * self.c + self.c_cov * new_estimate, self.eps)
        self.diversity = float(np.sum(self.c))  # Step 2: logged/diagnostic only

    def adapt_sigma(self, success_rate: Optional[float], best_score: Optional[float] = None) -> None:
        """Steps 2+3: homeostatic step-size control via Rechenberg's 1/5
        success rule (1973), not a diversity-EMA comparison.

        Step 3's original design compared current diversity to its OWN
        recent EMA -- but a slow, steady decline drags the EMA down right
        alongside the signal, so a real collapse can be arbitrarily far
        along before the *relative* drop looks large enough to react to
        (confirmed empirically: it silently converged to the eps floor).
        Success rate is bounded in [0, 1] against a fixed target (0.2),
        so it can't decay together with its own reference -- if fewer
        than 1/5 of recent mutations beat their parent, the population is
        over-exploring relative to the landscape's local structure and
        sigma should shrink; if more than 1/5 succeed, it's under-
        exploring and sigma should grow. This is the standard, decades-
        validated form of CSA-style step-size control.

        Same activation gate as `update`: on a near-zero-gradient plateau
        before any real optimum is found, "success" (child edges out
        parent) is close to a coin flip and NOT evidence that step size is
        wrong -- shrinking sigma there just makes the still-undiscovered
        target even harder to stumble onto. Hold sigma at its current
        (exploratory) value until real signal exists."""
        if success_rate is None or not self._has_signal(best_score):
            return
        self.sigma = float(
            np.clip(
                self.sigma * np.exp(self.k_sigma * (success_rate - self.target_success)),
                self.sigma_min,
                self.sigma_max,
            )
        )

    def sample(self, parent_x: np.ndarray, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        """Step 4 (corrected): anisotropic mutation, informed by elite
        covariance rather than a hand-set per-dimension bias."""
        z = rng.normal(size=self.dim)
        return parent_x + scale * self.sigma * np.sqrt(self.c) * z


def recent_success_rate(db: Any, dim: int, lambda_robust: float, window: int = 20) -> Optional[float]:
    """Fraction of the last `window` children whose robust_fitness beat
    their own parent's. Read-only; used only to drive adapt_sigma."""
    db.cursor.execute(
        "SELECT * FROM programs ORDER BY generation DESC LIMIT ?", (window,)
    )
    rows = db.cursor.fetchall()
    successes, total = 0, 0
    for row in rows:
        child = db._program_from_row(row)
        if child is None or not child.parent_id or not child.embedding:
            continue
        if len(child.embedding) != dim:
            continue
        parent = db.get(child.parent_id)
        if parent is None or not parent.embedding or len(parent.embedding) != dim:
            continue
        child_f = child.combined_score - lambda_robust * (
            child.metadata.get("eval_score_std", 0.0) or 0.0
        )
        parent_f = parent.combined_score - lambda_robust * (
            parent.metadata.get("eval_score_std", 0.0) or 0.0
        )
        total += 1
        if child_f > parent_f:
            successes += 1
    return successes / total if total else None


def make_homeocma_mutate(
    dim: int,
    mu: int = 6,
    min_archive: int = 6,
    success_window: int = 20,
    full_scale: float = 5.0,
    **cma_kwargs: Any,
):
    """Factory returning (mutate_fn, cma) where mutate_fn matches sim.py's
    mutate_fn interface: (parent_x, rng, dim, patch_type, cross_partner,
    step_scale, bounds, db) -> new genome. `cma` is exposed so callers can
    inspect state (sigma trace, covariance) for diagnostics/plots.
    """
    cma = HomeoCMA(dim=dim, mu=mu, **cma_kwargs)

    def mutate_fn(
        parent_x: np.ndarray,
        rng: np.random.Generator,
        dim: int,
        patch_type: str = "diff",
        cross_partner: Optional[np.ndarray] = None,
        step_scale: float = 0.07,
        bounds: float = 1.6,
        db: Any = None,
    ) -> np.ndarray:
        if db is not None:
            archive = db._get_archive_programs()
            best_score = max((p.combined_score or 0.0) for p in archive) if archive else None
            valid = [p for p in archive if p.embedding and len(p.embedding) == dim]
            if len(valid) >= min_archive:
                ranked = sorted(
                    valid,
                    key=lambda p: cma.robust_fitness(
                        p.combined_score or 0.0, p.metadata.get("eval_score_std", 0.0) or 0.0
                    ),
                    reverse=True,
                )[: cma.mu]
                cma.update(np.array([p.embedding for p in ranked]), best_score=best_score)
            cma.adapt_sigma(
                recent_success_rate(db, dim, cma.lambda_robust, window=success_window),
                best_score=best_score,
            )

        if patch_type == "cross" and cross_partner is not None:
            mask = rng.random(dim) < 0.5
            x = np.where(mask, parent_x, np.asarray(cross_partner, dtype=float))
        else:
            scale = full_scale if patch_type == "full" else 1.0
            x = cma.sample(parent_x, rng, scale=scale)
        return np.clip(x, -bounds, bounds)

    return mutate_fn, cma
