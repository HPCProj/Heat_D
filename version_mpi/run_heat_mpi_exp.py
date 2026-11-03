import os
import gc
import sys
import time
import glob
import json
import logging
import platform
import argparse
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def parse_args():
    parser = argparse.ArgumentParser(description="HPC MPI 2D Heat Diffusion Benchmark")
    parser.add_argument("--experiments_name", "-experiments_name", type=str, required=True)
    parser.add_argument("--results_folder", "-results_folder", type=str, default="results")
    parser.add_argument("--matrix_size", "-matrix_size", "--grid_size", type=str, default="512,1024,2048")
    parser.add_argument("--p_list", "-p_list", type=str, default="1,2,4,8")
    parser.add_argument("--run_repetition", "-run_repetition", type=int, default=3)
    parser.add_argument("--steps", "-steps", type=int, default=2000)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--check_interval", type=int, default=50)
    parser.add_argument("--frame_interval", type=int, default=40)
    parser.add_argument("--anim_grid_size", type=int, default=512)
    parser.add_argument("--anim_p", type=int, default=4)
    parser.add_argument("--no_hotspot", action="store_true")
    parser.add_argument("--hostfile", type=str, default=None)
    return parser.parse_args()


args = parse_args()

p_values = [int(p.strip()) for p in args.p_list.split(",") if p.strip()]
grid_sizes = [int(n.strip()) for n in args.matrix_size.split(",") if n.strip()]

target_dir = os.path.join("results", args.experiments_name)
frames_dir = os.path.join(target_dir, "frames")
os.makedirs(target_dir, exist_ok=True)

logger = logging.getLogger("MPI_Heat_Diffusion")
logger.setLevel(logging.INFO)
logger.handlers.clear()
formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(os.path.join(target_dir, "experiment.log"), mode="w", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


@njit(parallel=False, cache=True)
def serial_jacobi_step(u, v):
    n, m = u.shape
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            v[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])


@njit(parallel=False, cache=True)
def serial_enforce_hotspot(grid, cx, cy, r2):
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


def init_serial_grid(n, use_hotspot):
    u = np.zeros((n, n), dtype=np.float64)
    u[:, 0] = 60.0
    u[:, -1] = 0.0
    u[0, :] = 100.0
    u[0, 0] = 60.0
    u[0, -1] = 0.0
    u[-1, :] = 0.0
    u[-1, 0] = 60.0
    u[-1, -1] = 0.0

    cx, cy = n // 2, n // 2
    r2 = (n // 12) ** 2
    if use_hotspot:
        serial_enforce_hotspot(u, cx, cy, r2)
    v = u.copy()
    return u, v, cx, cy, r2


def compute_serial_reference(n, steps, repetitions, use_hotspot):
    u_w, v_w, cx_w, cy_w, r2_w = init_serial_grid(32, use_hotspot)
    serial_jacobi_step(u_w, v_w)

    times = []
    u_ref = None
    for r in range(repetitions):
        u, v, cx, cy, r2 = init_serial_grid(n, use_hotspot)
        t0 = time.perf_counter()
        for _ in range(steps):
            serial_jacobi_step(u, v)
            if use_hotspot:
                serial_enforce_hotspot(v, cx, cy, r2)
            u, v = v, u
        times.append(time.perf_counter() - t0)
        if r == 0:
            u_ref = u.copy()

    return float(np.median(times)), u_ref


def get_available_memory_gb():
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemAvailable:" in line:
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        return 999.0
    return 999.0


def build_mpirun_base(p):
    cmd = [
        "mpirun",
        "--oversubscribe",
        "--mca", "btl_vader_single_copy_mechanism", "none",
        "-n", str(p)
    ]
    if args.hostfile:
        cmd.extend(["--hostfile", args.hostfile])
    return cmd


def run_mpi_animation_pass():
    anim_n = args.anim_grid_size if args.anim_grid_size in grid_sizes else grid_sizes[0]
    anim_p = args.anim_p if anim_n % args.anim_p == 0 else p_values[0]
    os.makedirs(frames_dir, exist_ok=True)

    logger.info(f"Running dedicated MPI animation & convergence pass (N={anim_n}, p={anim_p}, steps={args.steps})...")
    t0 = time.perf_counter()

    cmd = build_mpirun_base(anim_p) + [
        sys.executable, "heat_mpi_worker.py",
        "--size", str(anim_n),
        "--steps", str(args.steps),
        "--check_interval", str(args.check_interval),
        "--frame_interval", str(args.frame_interval),
        "--frames_dir", frames_dir,
        "--animate"
    ]
    if args.no_hotspot:
        cmd.append("--no_hotspot")

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        logger.error(f"Animation pass failed: {res.stderr.strip()}")
        return {}

    anim_meta = {}
    for line in res.stdout.splitlines():
        if line.startswith("JSON_ANIM:"):
            anim_meta = json.loads(line.replace("JSON_ANIM:", ""))

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    gif_path = os.path.join(target_dir, "heat_diffusion.gif")
    if frame_files:
        frames_imgs = [imageio.imread(fp) for fp in frame_files]
        imageio.mimsave(gif_path, frames_imgs, fps=10, loop=0)
        final_img = imageio.imread(frame_files[-1])
        imageio.imwrite(os.path.join(target_dir, "heatmap_final.png"), final_img)

    if "convergence" in anim_meta and anim_meta["convergence"]:
        conv_df = pd.DataFrame(anim_meta["convergence"])
        conv_df.to_csv(os.path.join(target_dir, "convergence.csv"), index=False)

        plt.figure(figsize=(8, 5.5))
        plt.plot(conv_df["step"], conv_df["max_diff"], marker="o", markersize=4, linewidth=2, color="#d95f02")
        plt.axhline(y=args.tolerance, color="red", linestyle="--", alpha=0.7, label=f"Tolerance ({args.tolerance:.0e})")
        plt.yscale("log")
        plt.title(f"MPI Jacobi Convergence Curve (N = {anim_n}, p = {anim_p})")
        plt.xlabel("Iteration (Step)")
        plt.ylabel("Max Change ($\\max |u_{new} - u|$) [Log Scale]")
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, "convergence.png"), dpi=300)
        plt.close()

    viz_time = time.perf_counter() - t0
    logger.info(f"MPI Animation & GIF saved in {viz_time:.2f}s ({len(frame_files)} frames in {frames_dir})")
    return {
        "anim_grid_size": anim_n,
        "anim_ranks": anim_p,
        "frames_count": len(frame_files),
        "gif_path": gif_path,
        "visualization_time_s": round(viz_time, 4)
    }


def run_mpi_benchmark_trial(p, n, check_grid_path=None):
    avail_gb = get_available_memory_gb()
    est_needed_gb = p * 0.085
    if est_needed_gb > avail_gb * 0.90:
        logger.warning(f"Skipping p={p} for N={n}: Estimated memory ({est_needed_gb:.1f} GB) exceeds available RAM ({avail_gb:.1f} GB).")
        return "OOM_SKIP"

    cmd = build_mpirun_base(p) + [
        sys.executable, "heat_mpi_worker.py",
        "--size", str(n),
        "--steps", str(args.steps),
        "--check_interval", str(args.check_interval)
    ]
    if args.no_hotspot:
        cmd.append("--no_hotspot")
    if check_grid_path:
        cmd.extend(["--save_final_grid", check_grid_path])

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode in (-9, 137) or "Killed" in result.stderr:
        logger.error(f"OOM / Resource limit reached for p={p}, N={n} (Exit code {result.returncode}). Skipping.")
        return "OOM_SKIP"

    if result.returncode != 0:
        logger.error(f"mpirun error (p={p}, N={n}): {result.stderr.strip()}")
        return None

    for line in result.stdout.splitlines():
        if line.startswith("JSON_RESULT:"):
            return json.loads(line.replace("JSON_RESULT:", ""))
    return None


def save_config(viz_info):
    config = {
        "experiment_name": args.experiments_name,
        "timestamp": datetime.now().isoformat(),
        "experiment_dir": target_dir,
        "frames_dir": frames_dir,
        "grid_sizes": grid_sizes,
        "p_list": p_values,
        "run_repetition": args.run_repetition,
        "steps": args.steps,
        "tolerance": args.tolerance,
        "check_interval": args.check_interval,
        "frame_interval": args.frame_interval,
        "use_hotspot": not args.no_hotspot,
        "visualization": viz_info,
        "system": {
            "os": platform.system(),
            "os_release": platform.release(),
            "machine_arch": platform.machine(),
            "cpu_cores_logical": os.cpu_count(),
            "available_ram_gb": round(get_available_memory_gb(), 2),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__
        }
    }
    cfg_path = os.path.join(target_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {cfg_path}")


def generate_plots(df):
    if df.empty:
        logger.warning("DataFrame is empty. Skipping plots.")
        return

    logical_cores = os.cpu_count() or 1
    tested_p = sorted(df["p"].unique())

    plt.figure(figsize=(8, 6))
    for n, grp in df.groupby("n"):
        plt.plot(grp["p"], grp["speedup"], marker="o", linewidth=2, label=f"N = {n}")
    plt.axvline(x=logical_cores, color="red", linestyle="--", alpha=0.7, label=f"Logical Cores ({logical_cores})")
    plt.title(f"MPI 2D Heat Diffusion Speedup vs. Processes ({args.experiments_name})")
    plt.xlabel("Processes (p)")
    plt.ylabel("Speedup (S = T_serial / T_p)")
    plt.xticks(tested_p)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "speedup.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for n, grp in df.groupby("n"):
        plt.plot(grp["p"], grp["efficiency"], marker="s", linewidth=2, label=f"N = {n}")
    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Ideal Efficiency (1.0)")
    plt.axvline(x=logical_cores, color="red", linestyle="--", alpha=0.7, label=f"Logical Cores ({logical_cores})")
    plt.title(f"MPI 2D Heat Diffusion Efficiency vs. Processes ({args.experiments_name})")
    plt.xlabel("Processes (p)")
    plt.ylabel("Efficiency (E = S / p)")
    plt.xticks(tested_p)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "efficiency.png"), dpi=300)
    plt.close()

    max_n = df["n"].max()
    df_max = df[df["n"] == max_n]
    if not df_max.empty:
        plt.figure(figsize=(10, 6))
        p_str = [str(p) for p in df_max["p"]]
        comp = df_max["compute_time_s"].values
        halo = df_max["halo_time_s"].values
        allred = df_max["allreduce_time_s"].values

        plt.bar(p_str, comp, label="Computation (Jacobi Stencil)", color="#2b5c8f")
        plt.bar(p_str, halo, bottom=comp, label="Halo Exchange (Sendrecv)", color="#d95f02")
        plt.bar(p_str, allred, bottom=comp + halo, label="Global Convergence (Allreduce)", color="#7570b3")
        plt.title(f"MPI Time Breakdown: Compute vs. Communication (N = {max_n})")
        plt.xlabel("Processes (p)")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid(True, axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, "comm_comp_breakdown.png"), dpi=300)
        plt.close()

    plt.figure(figsize=(8, 6))
    for n, grp in df.groupby("n"):
        plt.plot(grp["p"], grp["halo_share_pct"], marker="^", linewidth=2, label=f"Measured Halo Share (N = {n})")
    plt.title(f"Share of Time Spent in Halo Exchange vs. N and p ({args.experiments_name})")
    plt.xlabel("Processes (p)")
    plt.ylabel("Halo Exchange Share of Total Time (%)")
    plt.xticks(tested_p)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "halo_share.png"), dpi=300)
    plt.close()

    logger.info(f"Plots saved in {target_dir} (speedup.png, efficiency.png, comm_comp_breakdown.png, halo_share.png)")


def main():
    logger.info(f"Starting MPI 2D Heat Diffusion experiment '{args.experiments_name}'")
    logger.info("Pre-compiling and caching Numba stencil kernels...")
    subprocess.run([sys.executable, "heat_mpi_worker.py", "--size", "32", "--warmup"], check=True)

    viz_info = run_mpi_animation_pass()
    save_config(viz_info)

    records = []
    csv_path = os.path.join(target_dir, "results.csv")
    temp_grid_path = os.path.join(target_dir, "_temp_mpi_grid.npy")

    for n in grid_sizes:
        logger.info(f"Computing Serial Baseline for N = {n} ({args.steps} steps)...")
        t_serial, u_serial_ref = compute_serial_reference(n, args.steps, args.run_repetition, not args.no_hotspot)
        logger.info(f"[Baseline] N={n} | Serial median runtime: {t_serial:.4f}s")

        for p in p_values:
            if n % p != 0:
                logger.warning(f"Skipping N={n}, p={p} because {n} is not divisible by {p}.")
                continue

            runs = []
            is_correct = False
            oom_triggered = False

            for r in range(args.run_repetition):
                save_path = temp_grid_path if r == 0 else None
                res = run_mpi_benchmark_trial(p, n, check_grid_path=save_path)
                if res == "OOM_SKIP":
                    oom_triggered = True
                    break
                elif res is not None:
                    runs.append(res)
                    if r == 0 and os.path.exists(temp_grid_path):
                        u_mpi = np.load(temp_grid_path)
                        is_correct = bool(np.allclose(u_serial_ref, u_mpi, atol=1e-6))
                        os.remove(temp_grid_path)

            if oom_triggered:
                break

            if not runs:
                logger.error(f"Failed executions for N={n}, p={p}")
                continue

            med_total = float(np.median([r["total_time_s"] for r in runs]))
            med_comp = float(np.median([r["compute_time_s"] for r in runs]))
            med_halo = float(np.median([r["halo_time_s"] for r in runs]))
            med_allred = float(np.median([r["allreduce_time_s"] for r in runs]))
            med_comm = med_halo + med_allred
            med_halo_pct = (med_halo / med_total) * 100.0 if med_total > 0 else 0.0
            final_diff = runs[0]["final_max_diff"]

            speedup = t_serial / med_total
            efficiency = speedup / p

            logger.info(
                f"[MPI] N={n:<4d} | p={p:<3d} | Total: {med_total:.4f}s | "
                f"Comp: {med_comp:.4f}s | Halo: {med_halo:.4f}s ({med_halo_pct:.1f}%) | "
                f"Allred: {med_allred:.4f}s | Speedup: {speedup:.2f}x | Eff: {efficiency:.2f} | Correct: {is_correct}"
            )

            records.append({
                "experiment_name": args.experiments_name,
                "n": n,
                "p": p,
                "steps": args.steps,
                "serial_time_s": round(t_serial, 6),
                "total_time_s": round(med_total, 6),
                "compute_time_s": round(med_comp, 6),
                "halo_time_s": round(med_halo, 6),
                "allreduce_time_s": round(med_allred, 6),
                "comm_time_s": round(med_comm, 6),
                "halo_share_pct": round(med_halo_pct, 2),
                "cost_model_p_over_n": round(p / n, 6),
                "speedup": round(speedup, 4),
                "efficiency": round(efficiency, 4),
                "final_max_diff": round(final_diff, 8),
                "correct_allclose": is_correct
            })

            pd.DataFrame(records).to_csv(csv_path, index=False)

        del u_serial_ref
        gc.collect()

    df = pd.DataFrame(records)
    if not df.empty:
        summary_cols = [
            "n", "p", "total_time_s", "compute_time_s", "halo_time_s",
            "halo_share_pct", "speedup", "efficiency", "correct_allclose"
        ]
        logger.info("MPI Experiment Summary Table:\n" + df[summary_cols].to_string(index=False))
        generate_plots(df)

    logger.info("MPI 2D Heat Diffusion Experiment completed successfully.")


if __name__ == "__main__":
    main()
