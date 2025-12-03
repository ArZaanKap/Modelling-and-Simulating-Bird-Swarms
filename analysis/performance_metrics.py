"""Performance measurement script for Boids and Cucker-Smale models.

This script runs N simulations for each model (default 5), measures:
- total runtime per run
- average time per simulation step
- formation time (based on polarization threshold)
- final polarization
- mean nearest-neighbour distance (NND)
- mean speed and speed std

It writes a JSON summary to `analysis/performance_summary.json` and prints
the averaged metrics.

Usage: run from repository root with the project's Python interpreter:
  python analysis/performance_metrics.py
"""

import os
import time
import json
import numpy as np
import importlib.util


def load_class_from_path(path, class_name):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location(os.path.basename(path).replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def polarization_metric(velocities):
    """Polarization: ||sum_i v_i|| / (N * mean_speed)

    velocities: (N, d) array
    """
    velocities = np.asarray(velocities)
    if velocities.size == 0:
        return 0.0
    N = velocities.shape[0]
    sum_v_norm = np.linalg.norm(np.sum(velocities, axis=0))
    mean_speed = np.mean(np.linalg.norm(velocities, axis=1)) + 1e-12
    return float(sum_v_norm / (N * mean_speed))


def mean_nearest_neighbor_distance(positions):
    positions = np.asarray(positions)
    if positions.size == 0:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dists = np.sqrt(np.min(dist_sq, axis=1))
    return float(np.mean(min_dists))


def run_simulation_model(model_name, class_path, class_name, num_runs=5, sim_time=60.0, dt=0.08, num_boids=35):
    """Run `num_runs` simulations and collect metrics. Returns averaged dict."""
    ModelClass = load_class_from_path(class_path, class_name)
    results = []

    steps = int(sim_time / dt)
    formation_threshold = 0.95
    sustain_steps = max(1, int(0.5 / dt))  # require threshold hold for ~0.5s

    for run in range(num_runs):
        # Try to instantiate with common constructor signatures, fall back to no-arg
        try:
            model = ModelClass(num_boids=num_boids, dt=dt)
        except TypeError:
            try:
                # some scripts use positional args (N, dt/ts)
                model = ModelClass(num_boids, dt)
            except TypeError:
                try:
                    model = ModelClass(num_boids, ts=dt)
                except TypeError:
                    model = ModelClass()

        start = time.time()
        pol_history = []
        formation_time = None

        for step in range(steps):
            # Advance simulation (handle common method names)
            if hasattr(model, "step"):
                model.step()
            elif hasattr(model, "update"):
                model.update()
            elif hasattr(model, "boids_algorithm"):
                model.boids_algorithm()
            else:
                raise RuntimeError("Unknown model API; expected .step(), .update(), or .boids_algorithm()")

            # Read velocities
            if hasattr(model, "v"):
                velocities = np.asarray(model.v)
            elif hasattr(model, "velocities"):
                velocities = np.asarray(model.velocities)
            else:
                # fallback: try to infer N
                N = getattr(model, "N", getattr(model, "num_boids", num_boids))
                velocities = np.zeros((N, 3))

            pol = polarization_metric(velocities)
            pol_history.append(pol)

            # Detect formation: polarization above threshold sustained
            if formation_time is None and len(pol_history) >= sustain_steps:
                if all(p >= formation_threshold for p in pol_history[-sustain_steps:]):
                    formation_time = (step - sustain_steps + 1) * dt

        end = time.time()
        total_time = end - start
        avg_step_time = total_time / max(1, steps)

        # Final state metrics (positions & velocities)
        if hasattr(model, "x"):
            positions = np.asarray(model.x)
        elif hasattr(model, "positions"):
            positions = np.asarray(model.positions)
        else:
            N = getattr(model, "N", getattr(model, "num_boids", num_boids))
            positions = np.zeros((N, 3))

        if hasattr(model, "v"):
            velocities = np.asarray(model.v)
        elif hasattr(model, "velocities"):
            velocities = np.asarray(model.velocities)
        else:
            velocities = np.zeros((positions.shape[0], 3))

        final_pol = polarization_metric(velocities)
        mean_nnd = mean_nearest_neighbor_distance(positions)
        mean_speed = float(np.mean(np.linalg.norm(velocities, axis=1)))
        speed_std = float(np.std(np.linalg.norm(velocities, axis=1)))

        results.append({
            "run": run + 1,
            "total_time_s": total_time,
            "avg_step_time_s": avg_step_time,
            "steps": steps,
            "formation_time_s": formation_time,
            "final_polarization": final_pol,
            "mean_nearest_neighbor_distance": mean_nnd,
            "mean_speed": mean_speed,
            "speed_std": speed_std,
        })

        print(f"[{model_name}] run {run+1}/{num_runs}: total_time={total_time:.2f}s avg_step={avg_step_time*1e3:.3f}ms formation_time={formation_time}")

    # Aggregate averages
    avg = {}
    keys = ["total_time_s", "avg_step_time_s", "final_polarization",
            "mean_nearest_neighbor_distance", "mean_speed", "speed_std"]
    for k in keys:
        avg[k] = float(np.mean([r[k] for r in results]))

    # For formation_time, ignore None values
    formation_times = [r["formation_time_s"] for r in results if r["formation_time_s"] is not None]
    avg["formation_time_s"] = float(np.mean(formation_times)) if formation_times else None
    avg["per_run"] = results
    return avg


if __name__ == "__main__":
    # ensure analysis dir exists
    os.makedirs("analysis", exist_ok=True)

    # Config: match dt used in visual scripts if desired
    sim_time = 60.0
    dt = 0.08
    num_runs = 5
    num_boids = 35

    models = [
        ("Boids (boids4)", "boids/boids4.py", "BoidSwarm"),
        ("Cucker-Smale (boids5)", "cucker-smale/boids5_cucker_smale.py", "CuckerSmaleSwarm"),
    ]

    summary = {}
    for name, path, cls in models:
        print(f"Measuring model: {name}")
        try:
            avg_metrics = run_simulation_model(name, path, cls, num_runs=num_runs, sim_time=sim_time, dt=dt, num_boids=num_boids)
            summary[name] = avg_metrics
        except Exception as e:
            summary[name] = {"error": str(e)}
            print(f"Error running {name}: {e}")

    out_path = os.path.join("analysis", "performance_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAveraged performance summary (saved to analysis/performance_summary.json):")
    for model_name, metrics in summary.items():
        print(f"\nModel: {model_name}")
        if "error" in metrics:
            print(f"  error: {metrics['error']}")
            continue
        print(f"  avg total time: {metrics['total_time_s']:.2f}s")
        print(f"  avg step time: {metrics['avg_step_time_s']*1e3:.3f} ms")
        if metrics["formation_time_s"] is not None:
            print(f"  avg formation time: {metrics['formation_time_s']:.2f}s")
        else:
            print("  formation time: not reached in runs")
        print(f"  final polarization: {metrics['final_polarization']:.3f}")
        print(f"  mean NN distance: {metrics['mean_nearest_neighbor_distance']:.3f}")
        print(f"  mean speed: {metrics['mean_speed']:.3f} ± {metrics['speed_std']:.3f}")
