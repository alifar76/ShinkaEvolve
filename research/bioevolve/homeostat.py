"""Step 3: Homeostatic feedback controller.

Biological concept: homeostasis via negative feedback (e.g. blood-glucose
regulation) -- a controller senses a state variable, compares it to a
setpoint, and pushes the system back toward that setpoint whenever it
drifts away in EITHER direction. Applied here: population diversity
(Step 2's entropy-flux signal) is the regulated variable. When it falls
below a floor, the controller injects more variation (larger mutations,
more inter-island migration) to counteract selection's steady pull toward
collapse. It does NOT try to maximize diversity -- pure negative feedback
around a floor, not open-loop cranking of exploration to the max (which
would just prevent convergence).

On top of the pure negative-feedback loop, there is a second,
compounding term: when the population is ALSO in a high fragility state
(Step 1's eval_score_std signal averaged over the recent window) while
diversity is low, that is the specific signature of "trapped on a fragile
local optimum" -- the controller pushes exploration harder in that
combined case than either signal would alone. This is the sense in which
Steps 1-3 compose: the same two observables (robustness, entropy) both
feed the archive's retention criterion (Step 1) and this live regulator
(Step 3).
"""

from typing import Any, Dict, List, Optional

import numpy as np

from sim import GenRecord, ProgramDatabase


def make_homeostat(
    base_step_scale: float = 0.07,
    min_step_scale: float = 0.02,
    max_step_scale: float = 0.35,
    base_migration_rate: float = 0.1,
    max_migration_rate: float = 0.4,
    diversity_window: int = 25,
    sample_every: int = 5,
    ema_alpha: float = 0.06,
    drop_threshold: float = 0.2,
    k_p: float = 1.5,
    fragility_gain: float = 1.0,
    fragility_window: int = 15,
):
    """Returns a controller closure compatible with sim.run_simulation's
    `controller` hook.

    Rather than a fixed setpoint calibrated once, the regulated baseline is
    a slow exponential moving average (EMA) of diversity -- the "recent
    trend". Diversity naturally rises during early exploration and falls
    during convergence; what matters for the homeostat is not "diversity
    below some absolute number" but "diversity dropping faster than its
    own recent trend", i.e. a collapse, not just the ordinary decline that
    accompanies normal convergence. drop_threshold is how large a
    fractional dip below the EMA (as measured just before the EMA itself
    is updated) counts as triggering corrective action.
    """
    state: Dict[str, Any] = {"ema": None, "last_signal": {}}

    def controller(
        gen: int, db: ProgramDatabase, history: List[GenRecord]
    ) -> Optional[Dict[str, Any]]:
        if gen % sample_every != 0:
            return state["last_signal"] or None

        since = max(0, gen - diversity_window)
        diversity = db.get_population_diversity(since_generation=since, metric="variance")
        if diversity is None:
            return state["last_signal"] or None

        if state["ema"] is None:
            state["ema"] = diversity
            return None

        ema = state["ema"]
        # Negative feedback: error > 0 means diversity has dropped below
        # its recent trend by more than drop_threshold and needs
        # correcting; at/above trend relaxes back to baseline.
        error = (ema - diversity) / max(ema, 1e-9) - drop_threshold
        error = float(np.clip(error, -1.0, 3.0))
        corrective = max(error, 0.0)
        state["ema"] = (1.0 - ema_alpha) * ema + ema_alpha * diversity

        recent = history[-fragility_window:] if history else []
        mean_fragility = (
            float(np.mean([r.child_score_std for r in recent])) if recent else 0.0
        )
        fragility_boost = fragility_gain * mean_fragility * corrective

        step_scale = base_step_scale * (1.0 + k_p * corrective) + fragility_boost
        step_scale = float(np.clip(step_scale, min_step_scale, max_step_scale))

        migration_rate = base_migration_rate
        if corrective > 0.3:
            migration_rate = min(
                max_migration_rate, base_migration_rate * (1.0 + 2.0 * corrective)
            )

        signal = {"step_scale": step_scale, "migration_rate": migration_rate}
        state["last_signal"] = signal
        return signal

    return controller
