"""Real-task validation of Step 1 (robustness-aware archive) against an
actual LLM-driven evolution run (via ShinkaEvolve's headless/claude
provider, no separate API key needed -- it drives the locally
authenticated Claude Code CLI).

Adapted from examples/sine_approx_headless/evaluate.py. The original
evaluator scores against the SAME fixed 161 points every time, so
"robustness" has no meaning there -- there is nothing to be inconsistent
across. Here each of `NUM_RUNS` runs scores against a different random
80-point subset of the domain (fixed per run-index across the whole
evolution run, so different candidate programs are compared fairly). This
gives "robustness" a real, honest meaning for this task: an approximation
that generalizes across the domain scores consistently across subsets
(low combined_score_std); one that's overfit to quirks of a few points --
e.g. a lookup-table/interpolation hack tuned to specific x values -- will
score well on some subsets and poorly on others (high combined_score_std).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from pathlib import Path

NUM_RUNS = 5
POINTS_PER_RUN = 80
TOTAL_POINTS = 161

BLOCKED_SOURCE_SNIPPETS = (
    "math.sin",
    "cmath.sin",
    "numpy.sin",
    "np.sin",
    "from math import sin",
    "import sin",
)


def _load_program(program_path: str):
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load program: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _all_points() -> list[float]:
    points = []
    for index in range(TOTAL_POINTS):
        base = -math.pi + (2.0 * math.pi * index / (TOTAL_POINTS - 1))
        wobble = 0.007 * math.sin(index * 1.61803398875)
        points.append(max(-math.pi, min(math.pi, base + wobble)))
    return points


def _run_subset(run_idx: int, all_points: list[float]) -> list[float]:
    # Fixed per run-index so every candidate program in this evolution
    # run is compared against the SAME 5 subsets -- fair, reproducible.
    rng = random.Random(1000 + run_idx)
    return rng.sample(all_points, POINTS_PER_RUN)


def _source_violation(source: str) -> str | None:
    lowered = source.lower()
    for blocked in BLOCKED_SOURCE_SNIPPETS:
        if blocked in lowered:
            return f"blocked direct sine implementation: {blocked}"
    return None


def _score_on_points(approximate, points: list[float]) -> tuple[float, dict]:
    abs_errors = []
    for x in points:
        estimate = float(approximate(x))
        if not math.isfinite(estimate) or abs(estimate) > 10.0:
            raise ValueError("non-finite or out-of-range output")
        abs_errors.append(abs(estimate - math.sin(x)))
    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(e * e for e in abs_errors) / len(abs_errors))
    max_error = max(abs_errors)
    score = 1.0 / (1.0 + 4.0 * rmse + max_error)
    return score, {"mean_abs_error": mae, "rmse": rmse, "max_abs_error": max_error}


def _evaluate(program_path: str) -> tuple[dict, bool, str]:
    source = Path(program_path).read_text(encoding="utf-8")
    violation = _source_violation(source)
    if violation:
        return _failure_metrics(violation), False, violation

    module = _load_program(program_path)
    approximate = getattr(module, "approximate", None)
    if not callable(approximate):
        msg = "missing callable approximate(x)"
        return _failure_metrics(msg), False, msg

    all_points = _all_points()
    try:
        run_scores = []
        run_details = []
        for run_idx in range(NUM_RUNS):
            subset = _run_subset(run_idx, all_points)
            score, detail = _score_on_points(approximate, subset)
            run_scores.append(score)
            run_details.append(detail)
    except Exception as exc:  # noqa: BLE001
        return _failure_metrics(str(exc)), False, str(exc)

    mean_score = sum(run_scores) / len(run_scores)
    variance = sum((s - mean_score) ** 2 for s in run_scores) / len(run_scores)
    std_score = math.sqrt(variance)

    return (
        {
            "combined_score": mean_score,
            "combined_score_std": std_score,
            "public": {
                "score": mean_score,
                "score_std": std_score,
                "run_scores": run_scores,
                "mean_abs_error": sum(d["mean_abs_error"] for d in run_details) / NUM_RUNS,
            },
            "private": {"num_runs": NUM_RUNS, "points_per_run": POINTS_PER_RUN},
        },
        True,
        "",
    )


def _failure_metrics(error: str) -> dict:
    return {
        "combined_score": 0.0,
        "combined_score_std": 0.0,
        "public": {"score": 0.0, "score_std": 0.0},
        "private": {"error": error},
    }


def main(program_path: str, results_dir: str):
    metrics, correct, error = _evaluate(program_path)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    (results_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (results_path / "correct.json").write_text(
        json.dumps({"correct": correct, "error": error}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
