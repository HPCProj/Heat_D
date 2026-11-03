import os
import gc
import sys
import time
import json
import logging
import argparse
import platform
from datetime import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(description="HPC 2D Heat Diffusion Benchmark (Version 0 & Version 1)")
    parser.add_argument("--experiments_name", "-experiments_name", type=str, required=True)
    parser.add_argument("--results_folder", "-results_folder", type=str, default="results")
    parser.add_argument("--matrix_size", "-matrix_size", "--grid_size", type=str, default="512,1024")
    parser.add_argument("--p_list", "-p_list", type=str, default="1,2,4,8")
    parser.add_argument("--run_repetition", "-run_repetition", type=int, default=4)
    parser.add_argument("--steps", "-steps", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--check_interval", type=int, default=50)
    parser.add_argument("--frame_interval", type=int, default=40)
    parser.add_argument("--anim_grid_size", type=int, default=512)
    parser.add_argument("--no_hotspot", action="store_true")
    parser.add_argument("--threading_layer", type=str, default="omp", choices=["omp", "default", "workqueue", "tbb"])
    parser.add_argument("--config_file", type=str, default=None)
    return parser.parse_args()


def load_or_build_preliminary_config(args):
 
        grid_sizes = [int(x.strip()) for x in args.matrix_size.split(",") if x.strip()]
        p_list = [int(x.strip()) for x in args.p_list.split(",") if x.strip()]
        repetitions = args.run_repetition
        steps = args.steps
        tolerance = args.tolerance
        check_interval = args.check_interval
        frame_interval = args.frame_interval
        anim_grid_size = args.anim_grid_size if args.anim_grid_size in grid_sizes else grid_sizes[0]
        use_hotspot = not args.no_hotspot
        threading_layer = args.threading_layer

        exp_dir = os.path.join("results", args.experiments_name)
        return {
        "experiment_name": args.experiments_name,
        "base_results_folder": "results",
        "experiment_dir": exp_dir,
        "frames_dir": os.path.join(exp_dir, "frames"),
        "grid_sizes": grid_sizes,
        "p_list": p_list,
        "run_repetition": repetitions,
        "steps": steps,
        "tolerance": tolerance,
        "check_interval": check_interval,
        "frame_interval": frame_interval,
        "anim_grid_size": anim_grid_size,
        "use_hotspot": use_hotspot,
        "boundary_conditions": {"top": 100.0, "left": 60.0, "right": 0.0, "bottom": 0.0},
        "requested_threading_layer": threading_layer,
    }


args = parse_arguments()
prelim_config = load_or_build_preliminary_config(args)

max_threads_requested = max(prelim_config["p_list"])
logical_cores = os.cpu_count() or 1
os.environ["NUMBA_NUM_THREADS"] = str(max(max_threads_requested, logical_cores))
if prelim_config["requested_threading_layer"] != "default":
    os.environ["NUMBA_THREADING_LAYER"] = prelim_config["requested_threading_layer"]

import numpy as np
import pandas as pd
import numba
from numba import njit, prange
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def setup_logger(experiment_dir):
    os.makedirs(experiment_dir, exist_ok=True)
    log_file = os.path.join(experiment_dir, "experiment.log")

    logger = logging.getLogger("HPC_Heat_Diffusion")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


@njit(parallel=False, cache=True)
def jacobi_step_serial(u, v):
    n, m = u.shape
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            v[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])


@njit(parallel=True, cache=True)
def jacobi_step_parallel(u, v):
    n, m = u.shape
    for i in prange(1, n - 1):
        for j in range(1, m - 1):
            v[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])


@njit(parallel=False, cache=True)
def enforce_hotspot_serial(grid, cx, cy, r2):
    n, m = grid.shape
    r = int(np.sqrt(r2)) + 1
    i_start = max(1, cx - r)
    i_end = min(n - 1, cx + r + 1)
    j_start = max(1, cy - r)
    j_end = min(m - 1, cy + r + 1)
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            if (i - cx) * (i - cx) + (j - cy) * (j - cy) <= r2:
                grid[i, j] = 100.0


@njit(parallel=False, cache=True)
def compute_max_diff_serial(u, v):
    n, m = u.shape
    max_diff = 0.0
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            diff = abs(v[i, j] - u[i, j])
            if diff > max_diff:
                max_diff = diff
    return max_diff


def initialize_grids(n, use_hotspot):
    u = np.zeros((n, n), dtype=np.float64)
    u[0, :] = 100.0
    u[-1, :] = 0.0
    u[:, 0] = 60.0
    u[:, -1] = 0.0

    cx, cy = n // 2, n // 2
    r2 = (n // 12) ** 2
    if use_hotspot:
        enforce_hotspot_serial(u, cx, cy, r2)

    v = u.copy()
    return u, v, cx, cy, r2


def run_solver_serial(u, v, steps, use_hotspot, cx, cy, r2):
    for _ in range(steps):
        jacobi_step_serial(u, v)
        if use_hotspot:
            enforce_hotspot_serial(v, cx, cy, r2)
        u, v = v, u
    return u


def run_solver_parallel(u, v, steps, use_hotspot, cx, cy, r2):
    for _ in range(steps):
        jacobi_step_parallel(u, v)
        if use_hotspot:
            enforce_hotspot_serial(v, cx, cy, r2)
        u, v = v, u
    return u


def warmup_kernels(logger, requested_layer):
    logger.info("Starting Numba JIT warm-up for Jacobi stencil kernels...")
    u_dummy, v_dummy, cx, cy, r2 = initialize_grids(32, True)
    jacobi_step_serial(u_dummy, v_dummy)
    enforce_hotspot_serial(v_dummy, cx, cy, r2)
    compute_max_diff_serial(u_dummy, v_dummy)

    try:
        numba.set_num_threads(1)
        jacobi_step_parallel(u_dummy, v_dummy)
        active_layer = numba.threading_layer()
        logger.info(f"Numba warm-up complete. Active threading layer: {active_layer}")
    except ValueError as err:
        logger.warning(f"Requested threading layer '{requested_layer}' unavailable ({err}). Falling back to 'default'.")
        os.environ["NUMBA_THREADING_LAYER"] = "default"
        numba.config.THREADING_LAYER = "default"
        jacobi_step_parallel(u_dummy, v_dummy)
        active_layer = numba.threading_layer()
        logger.info(f"Fallback warm-up complete. Active threading layer: {active_layer}")

    return active_layer


def save_configuration(prelim_cfg, active_layer, logger, viz_metadata=None):
    config_data = {
        "experiment_name": prelim_cfg["experiment_name"],
        "timestamp": datetime.now().isoformat(),
        "experiment_dir": prelim_cfg["experiment_dir"],
        "frames_dir": prelim_cfg["frames_dir"],
        "grid_sizes": prelim_cfg["grid_sizes"],
        "p_list": prelim_cfg["p_list"],
        "run_repetition": prelim_cfg["run_repetition"],
        "steps": prelim_cfg["steps"],
        "tolerance": prelim_cfg["tolerance"],
        "check_interval": prelim_cfg["check_interval"],
        "frame_interval": prelim_cfg["frame_interval"],
        "anim_grid_size": prelim_cfg["anim_grid_size"],
        "use_hotspot": prelim_cfg["use_hotspot"],
        "boundary_conditions": prelim_cfg["boundary_conditions"],
        "visualization_metrics": viz_metadata or {},
        "hardware_and_env": {
            "logical_cpu_cores": os.cpu_count() or 1,
            "max_allocated_numba_threads": int(os.environ["NUMBA_NUM_THREADS"]),
            "requested_threading_layer": prelim_cfg["requested_threading_layer"],
            "active_threading_layer": active_layer,
            "os_system": platform.system(),
            "os_release": platform.release(),
            "machine_arch": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "numba_version": numba.__version__,
            "pandas_version": pd.__version__,
        }
    }

    config_path = os.path.join(prelim_cfg["experiment_dir"], "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)

    logger.info(f"Configuration saved to {config_path}")
    return config_data


def render_single_frame(grid, step, max_diff, frame_path, num_subdomains=4):
    n = grid.shape[0]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(grid, cmap="inferno", vmin=0.0, vmax=100.0, origin="upper")

    for r in range(1, num_subdomains):
        y_line = (n * r) / num_subdomains
        ax.axhline(y=y_line, color="cyan", linestyle="--", linewidth=1.0, alpha=0.8)

    for r in range(num_subdomains):
        y_pos = (n * (r + 0.5)) / num_subdomains
        ax.text(n * 0.96, y_pos, f"strip {r}", color="cyan", fontsize=8, ha="right", va="center")

    ax.set_title(f"2D Heat Diffusion (N={n}, Step={step}, Max $\\Delta u$={max_diff:.2e})")
    ax.set_xlabel("X Grid Index")
    ax.set_ylabel("Y Grid Index")
    fig.colorbar(im, ax=ax, label="Temperature (°C)")
    fig.tight_layout()
    fig.savefig(frame_path, dpi=120)
    plt.close(fig)


def generate_animation_and_convergence(config, logger):
    n = config["anim_grid_size"]
    steps = config["steps"]
    frame_interval = config["frame_interval"]
    check_interval = config["check_interval"]
    use_hotspot = config["use_hotspot"]
    frames_dir = config["frames_dir"]
    exp_dir = config["experiment_dir"]

    os.makedirs(frames_dir, exist_ok=True)
    logger.info(f"Starting dedicated visualization & convergence pass (N={n}, steps={steps})...")

    t_viz_start = time.perf_counter()
    u, v, cx, cy, r2 = initialize_grids(n, use_hotspot)

    frame_paths = []
    convergence_records = []
    current_diff = 100.0

    initial_frame = os.path.join(frames_dir, f"frame_{0:05d}.png")
    render_single_frame(u, 0, current_diff, initial_frame)
    frame_paths.append(initial_frame)

    for step in range(1, steps + 1):
        jacobi_step_serial(u, v)
        if use_hotspot:
            enforce_hotspot_serial(v, cx, cy, r2)

        if step % check_interval == 0 or step % frame_interval == 0 or step == steps:
            current_diff = float(compute_max_diff_serial(u, v))
            convergence_records.append({"step": step, "max_diff": current_diff})

        u, v = v, u

        if step % frame_interval == 0 or step == steps:
            frame_file = os.path.join(frames_dir, f"frame_{step:05d}.png")
            render_single_frame(u, step, current_diff, frame_file)
            frame_paths.append(frame_file)

    gif_path = os.path.join(exp_dir, "heat_diffusion.gif")
    frames_data = [imageio.imread(fp) for fp in frame_paths]
    imageio.mimsave(gif_path, frames_data, fps=10, loop=0)

    final_heatmap_path = os.path.join(exp_dir, "heatmap_final.png")
    render_single_frame(u, steps, current_diff, final_heatmap_path)

    conv_df = pd.DataFrame(convergence_records)
    conv_csv_path = os.path.join(exp_dir, "convergence.csv")
    conv_df.to_csv(conv_csv_path, index=False)

    plt.figure(figsize=(8, 5.5))
    plt.plot(conv_df["step"], conv_df["max_diff"], marker="o", markersize=4, linewidth=2, color="#d95f02")
    plt.axhline(y=config["tolerance"], color="red", linestyle="--", alpha=0.7, label=f"Tolerance ({config['tolerance']:.0e})")
    plt.yscale("log")
    plt.title(f"Jacobi Convergence Curve (N = {n})")
    plt.xlabel("Iteration (Step)")
    plt.ylabel("Max Change ($\\max |u_{new} - u|$) [Log Scale]")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "convergence.png"), dpi=300)
    plt.close()

    viz_time_s = time.perf_counter() - t_viz_start
    logger.info(
        f"Visualization pass complete in {viz_time_s:.2f}s (isolated from benchmark timings). "
        f"Saved {len(frame_paths)} frames in '{frames_dir}' and GIF at '{gif_path}'."
    )

    return {
        "visualization_total_time_s": round(viz_time_s, 4),
        "frames_saved_count": len(frame_paths),
        "gif_path": gif_path,
        "final_max_diff": round(current_diff, 8)
    }


def run_benchmarks(config, logger):
    records = []
    repetitions = config["run_repetition"]
    steps = config["steps"]
    use_hotspot = config["use_hotspot"]
    csv_path = os.path.join(config["experiment_dir"], "results.csv")

    for n in config["grid_sizes"]:
        logger.info(f"Running Jacobi benchmarks for Grid Size N = {n} x {n} ({steps} steps)")

        serial_runs = []
        u_serial_final = None
        for r in range(repetitions):
            u, v, cx, cy, r2 = initialize_grids(n, use_hotspot)
            t0 = time.perf_counter()
            u_out = run_solver_serial(u, v, steps, use_hotspot, cx, cy, r2)
            serial_runs.append(time.perf_counter() - t0)
            if r == 0:
                u_serial_final = u_out.copy()

        t_serial = float(np.median(serial_runs))
        cells_updated_per_sec = ((n - 2) * (n - 2) * steps) / t_serial
        logger.info(
            f"[N={n}] Version 0 (Serial)      | Median Time: {t_serial:.6f}s | "
            f"Throughput: {cells_updated_per_sec / 1e6:.2f} MCells/s"
        )

        records.append({
            "experiment_name": config["experiment_name"],
            "version": "Version 0 (Serial)",
            "n": n,
            "p": 1,
            "steps": steps,
            "repetitions": repetitions,
            "median_time_s": round(t_serial, 6),
            "min_time_s": round(float(np.min(serial_runs)), 6),
            "max_time_s": round(float(np.max(serial_runs)), 6),
            "std_time_s": round(float(np.std(serial_runs)), 6),
            "speedup": 1.0,
            "efficiency": 1.0,
            "mcells_per_s": round(cells_updated_per_sec / 1e6, 4),
            "correct_allclose": True,
            "raw_times_s": ";".join(f"{t:.6f}" for t in serial_runs),
        })
        pd.DataFrame(records).to_csv(csv_path, index=False)

        for p in config["p_list"]:
            numba.set_num_threads(p)
            parallel_runs = []
            is_correct = False

            for r in range(repetitions):
                u, v, cx, cy, r2 = initialize_grids(n, use_hotspot)
                t0 = time.perf_counter()
                u_par = run_solver_parallel(u, v, steps, use_hotspot, cx, cy, r2)
                parallel_runs.append(time.perf_counter() - t0)
                if r == 0:
                    is_correct = bool(np.allclose(u_serial_final, u_par))

            t_p = float(np.median(parallel_runs))
            speedup = t_serial / t_p
            efficiency = speedup / p
            mcells_s = (((n - 2) * (n - 2) * steps) / t_p) / 1e6

            logger.info(
                f"[N={n}] Version 1 (Parallel)    | p={p:<3d} | Median Time: {t_p:.6f}s | "
                f"Speedup: {speedup:.4f}x | Efficiency: {efficiency:.4f} | Correct: {is_correct}"
            )

            records.append({
                "experiment_name": config["experiment_name"],
                "version": "Version 1 (Parallel)",
                "n": n,
                "p": p,
                "steps": steps,
                "repetitions": repetitions,
                "median_time_s": round(t_p, 6),
                "min_time_s": round(float(np.min(parallel_runs)), 6),
                "max_time_s": round(float(np.max(parallel_runs)), 6),
                "std_time_s": round(float(np.std(parallel_runs)), 6),
                "speedup": round(speedup, 4),
                "efficiency": round(efficiency, 4),
                "mcells_per_s": round(mcells_s, 4),
                "correct_allclose": is_correct,
                "raw_times_s": ";".join(f"{t:.6f}" for t in parallel_runs),
            })
            pd.DataFrame(records).to_csv(csv_path, index=False)

        del u_serial_final
        gc.collect()

    return pd.DataFrame(records)


def generate_scaling_plots(df, config, logger):
    parallel_df = df[df["version"] == "Version 1 (Parallel)"].copy()
    if parallel_df.empty:
        logger.warning("No parallel results available to plot.")
        return

    exp_dir = config["experiment_dir"]
    exp_name = config["experiment_name"]
    p_values = config["p_list"]
    cpu_cores = config["hardware_and_env"]["logical_cpu_cores"]

    plt.figure(figsize=(8, 6))
    for n, group in parallel_df.groupby("n"):
        plt.plot(group["p"], group["speedup"], marker="o", linewidth=2, label=f"N = {n}")
    plt.axvline(x=cpu_cores, color="red", linestyle=":", alpha=0.7, label=f"Logical Cores ({cpu_cores})")
    plt.title(f"2D Heat Diffusion Speedup vs. Threads ({exp_name})")
    plt.xlabel("Number of Threads (p)")
    plt.ylabel("Speedup (S = T_serial / T_p)")
    plt.xticks(p_values)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "speedup.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for n, group in parallel_df.groupby("n"):
        plt.plot(group["p"], group["efficiency"], marker="s", linewidth=2, label=f"N = {n}")
    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Ideal Efficiency (1.0)")
    plt.axvline(x=cpu_cores, color="red", linestyle=":", alpha=0.7, label=f"Logical Cores ({cpu_cores})")
    plt.title(f"2D Heat Diffusion Efficiency vs. Threads ({exp_name})")
    plt.xlabel("Number of Threads (p)")
    plt.ylabel("Efficiency (E = S / p)")
    plt.xticks(p_values)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "efficiency.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for n, group in parallel_df.groupby("n"):
        plt.plot(group["p"], group["median_time_s"], marker="^", linewidth=2, label=f"N = {n}")
    plt.axvline(x=cpu_cores, color="red", linestyle=":", alpha=0.7, label=f"Logical Cores ({cpu_cores})")
    plt.title(f"2D Heat Diffusion Runtime vs. Threads ({exp_name})")
    plt.xlabel("Number of Threads (p)")
    plt.ylabel("Median Time (s) [Log Scale]")
    plt.yscale("log")
    plt.xticks(p_values)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "runtime.png"), dpi=300)
    plt.close()

    logger.info(f"Scaling plots saved in {exp_dir} (speedup.png, efficiency.png, runtime.png)")


def main():
    logger = setup_logger(prelim_config["experiment_dir"])
    logger.info(f"Starting 2D Heat Diffusion experiment '{prelim_config['experiment_name']}'")

    active_layer = warmup_kernels(logger, prelim_config["requested_threading_layer"])
    viz_meta = generate_animation_and_convergence(prelim_config, logger)
    full_config = save_configuration(prelim_config, active_layer, logger, viz_meta)

    results_df = run_benchmarks(full_config, logger)

    csv_path = os.path.join(full_config["experiment_dir"], "results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Benchmark results saved to {csv_path}")

    summary_cols = ["version", "n", "p", "steps", "median_time_s", "speedup", "efficiency", "correct_allclose"]
    logger.info("Experiment Summary Table:\n" + results_df[summary_cols].to_string(index=False))

    generate_scaling_plots(results_df, full_config, logger)
    logger.info("Experiment finished successfully.")


if __name__ == "__main__":
    main()
