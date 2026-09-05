"""Simulation driver that exercises shinka's REAL ProgramDatabase (archive,
island, and parent-selection logic in shinka/database) with a synthetic,
free, deterministic mutation operator standing in for the LLM proposal step.

Why not just call shinka_run with a real LLM? The four hypotheses under test
(robustness-aware archiving, entropy-flux monitoring, homeostatic feedback,
motif conservation) are about the *population-management engine*, not about
LLM code-generation quality. Driving the real engine with a real LLM would
(a) cost real API budget per data point, (b) take far longer per generation
than we need for statistical power (we want hundreds of generations x many
seeds x several ablations), and (c) confound engine effects with LLM
proposal-quality noise. Using shinka.database directly means every
selection/archive/island decision is made by the exact same code a real run
would use; only the "did this patch make the code better" step is replaced
by a controlled synthetic landscape.
"""

import contextlib
import io
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from rich.console import Console as RichConsole

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shinka.database import DatabaseConfig, Program, ProgramDatabase  # noqa: E402

from landscape import Landscape, make_landscape, mutate  # noqa: E402

logging.getLogger("shinka").setLevel(logging.ERROR)

_NULL_CONSOLE = RichConsole(file=open(os.devnull, "w"))


@dataclass
class GenRecord:
    generation: int
    child_score: float
    child_true_fitness: float
    child_score_std: float
    best_score_so_far: float
    island_idx: Optional[int]
    patch_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimResult:
    history: List[GenRecord]
    db: ProgramDatabase
    landscape: Landscape


def build_db(
    archive_criteria: Dict[str, float],
    num_islands: int = 3,
    archive_size: int = 20,
    migration_interval: int = 15,
    migration_rate: float = 0.1,
    parent_selection_strategy: str = "weighted",
    exploitation_ratio: float = 0.2,
    enable_dynamic_islands: bool = False,
    stagnation_threshold: int = 60,
    island_selection_strategy: str = "uniform",
) -> ProgramDatabase:
    config = DatabaseConfig(
        db_path=None,
        num_islands=num_islands,
        archive_size=archive_size,
        archive_criteria=archive_criteria,
        migration_interval=migration_interval,
        migration_rate=migration_rate,
        parent_selection_strategy=parent_selection_strategy,
        exploitation_ratio=exploitation_ratio,
        enable_dynamic_islands=enable_dynamic_islands,
        stagnation_threshold=stagnation_threshold,
        island_selection_strategy=island_selection_strategy,
    )
    db = ProgramDatabase(config, embedding_model=None)
    db.set_display_console(_NULL_CONSOLE)
    return db


ControllerFn = Callable[[int, ProgramDatabase, List[GenRecord]], Optional[Dict[str, Any]]]


def run_simulation(
    archive_criteria: Dict[str, float],
    num_generations: int = 400,
    num_islands: int = 3,
    num_runs_eval: int = 3,
    eval_env_sigma: float = 0.05,
    eval_meas_sigma: float = 0.015,
    dim: int = 6,
    seed: int = 0,
    step_scale: float = 0.07,
    patch_type_probs: Optional[Dict[str, float]] = None,
    db_kwargs: Optional[Dict[str, Any]] = None,
    on_generation: Optional[Callable[[int, ProgramDatabase, GenRecord], None]] = None,
    controller: Optional[ControllerFn] = None,
    landscape_factory: Callable[[int, int], Landscape] = None,
    mutate_fn: Optional[Callable[..., np.ndarray]] = None,
) -> SimResult:
    """
    controller: called at the START of each generation with
        (gen, db, history_so_far). May return a dict of overrides applied
        before that generation's parent/mutation step:
          - "step_scale": float, new mutation magnitude
          - "patch_type_probs": Dict[str, float], new patch-type mix
          - "migration_rate": float, live-patched onto db.config
          - "exploitation_ratio": float, live-patched onto db.config
        This is how Step 3's homeostatic feedback loop is wired in without
        hardcoding any control logic into the generic simulation driver.
    """
    rng = np.random.default_rng(seed)
    landscape_factory = landscape_factory or (lambda d, s: make_landscape(dim=d, seed=s))
    landscape = landscape_factory(dim, seed)
    mutate_impl = mutate_fn or mutate
    db_kwargs = dict(db_kwargs or {})
    db_kwargs.setdefault("num_islands", num_islands)
    db = build_db(archive_criteria=archive_criteria, **db_kwargs)

    patch_type_probs = patch_type_probs or {"diff": 0.6, "full": 0.3, "cross": 0.1}
    patch_types = list(patch_type_probs.keys())
    patch_probs = [patch_type_probs[k] for k in patch_types]

    x0 = np.zeros(dim)
    init_samples = landscape.noisy_eval(
        x0, num_runs_eval, rng, env_sigma=eval_env_sigma, meas_sigma=eval_meas_sigma
    )
    init_program = Program(
        id=str(uuid.uuid4()),
        code=f"# genome\nx = {x0.tolist()}\n",
        generation=0,
        correct=True,
        combined_score=float(np.mean(init_samples)),
        embedding=x0.tolist(),
        metadata={
            "eval_score_std": float(np.std(init_samples)),
            "true_fitness": landscape.true_fitness(x0),
        },
    )
    with contextlib.redirect_stdout(io.StringIO()):
        db.add(init_program)

    history: List[GenRecord] = []
    best_so_far = init_program.combined_score
    current_step_scale = step_scale

    for gen in range(1, num_generations + 1):
        if controller is not None:
            signal = controller(gen, db, history) or {}
            if "step_scale" in signal:
                current_step_scale = float(signal["step_scale"])
            if "patch_type_probs" in signal:
                patch_types = list(signal["patch_type_probs"].keys())
                patch_probs = [signal["patch_type_probs"][k] for k in patch_types]
            if "migration_rate" in signal:
                db.config.migration_rate = float(signal["migration_rate"])
            if "exploitation_ratio" in signal:
                db.config.exploitation_ratio = float(signal["exploitation_ratio"])

        parent, archive_insp, top_k_insp = db.sample()
        parent_x = np.array(parent.embedding) if parent.embedding else x0

        patch_type = str(rng.choice(patch_types, p=patch_probs))
        cross_partner = None
        if patch_type == "cross" and archive_insp:
            candidate = archive_insp[int(rng.integers(0, len(archive_insp)))]
            if candidate.embedding:
                cross_partner = np.array(candidate.embedding)

        child_x = mutate_impl(
            parent_x,
            rng=rng,
            dim=dim,
            patch_type=patch_type,
            cross_partner=cross_partner,
            step_scale=current_step_scale,
            db=db,
        )

        samples = landscape.noisy_eval(
            child_x,
            num_runs_eval,
            rng,
            env_sigma=eval_env_sigma,
            meas_sigma=eval_meas_sigma,
        )
        mean_score = float(np.mean(samples))
        std_score = float(np.std(samples))
        true_fit = landscape.true_fitness(child_x)

        child = Program(
            id=str(uuid.uuid4()),
            code=f"# genome\nx = {child_x.tolist()}\n",
            generation=gen,
            correct=True,
            combined_score=mean_score,
            parent_id=parent.id,
            embedding=child_x.tolist(),
            metadata={
                "eval_score_std": std_score,
                "true_fitness": true_fit,
                "patch_type": patch_type,
            },
        )
        with contextlib.redirect_stdout(io.StringIO()):
            db.add(child)
        best_so_far = max(best_so_far, mean_score)

        record = GenRecord(
            generation=gen,
            child_score=mean_score,
            child_true_fitness=true_fit,
            child_score_std=std_score,
            best_score_so_far=best_so_far,
            island_idx=child.island_idx,
            patch_type=patch_type,
            metadata={
                "step_scale": current_step_scale,
                "migration_rate": db.config.migration_rate,
                "exploitation_ratio": db.config.exploitation_ratio,
            },
        )
        history.append(record)
        if on_generation is not None:
            on_generation(gen, db, record)

    return SimResult(history=history, db=db, landscape=landscape)
