import os
import sys
import json
import argparse

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
from numba import njit


@njit(parallel=False, cache=True)
def jacobi_strip_step(u, v, i_start, i_end):
    _, n = u.shape
    for i in range(i_start, i_end):
        for j in range(1, n - 1):
            v[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])


@njit(parallel=False, cache=True)
def enforce_strip_hotspot(grid, global_row_offset, rows, cx, cy, r2):
    n = grid.shape[1]
    for loc_i in range(1, rows + 1):
        glob_i = global_row_offset + (loc_i - 1)
        di = glob_i - cx
        if di * di <= r2:
            for j in range(1, n - 1):
                dj = j - cy
                if di * di + dj * dj <= r2:
                    grid[loc_i, j] = 100.0


@njit(parallel=False, cache=True)
def compute_strip_max_diff(u, v, i_start, i_end):
    _, n = u.shape
    max_diff = 0.0
    for i in range(i_start, i_end):
        for j in range(1, n - 1):
            diff = abs(v[i, j] - u[i, j])
            if diff > max_diff:
                max_diff = diff
    return max_diff


def initialize_local_strip(n, rows, rank, size, use_hotspot):
    u = np.zeros((rows + 2, n), dtype=np.float64)
    u[:, 0] = 60.0
    u[:, -1] = 0.0
    if rank == 0:
        u[1, :] = 100.0
        u[1, 0] = 60.0
        u[1, -1] = 0.0
    if rank == size - 1:
        u[-2, :] = 0.0
        u[-2, 0] = 60.0
        u[-2, -1] = 0.0

    cx, cy = n // 2, n // 2
    r2 = (n // 12) ** 2
    global_row_offset = rank * rows

    if use_hotspot:
        enforce_strip_hotspot(u, global_row_offset, rows, cx, cy, r2)

    v = u.copy()
    return u, v, global_row_offset, cx, cy, r2


def render_rank_frame(grid, step, max_diff, frame_path, num_ranks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = grid.shape[0]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(grid, cmap="inferno", vmin=0.0, vmax=100.0, origin="upper")

    for r in range(1, num_ranks):
        y_line = (n * r) / num_ranks
        ax.axhline(y=y_line, color="cyan", linestyle="--", linewidth=1.1, alpha=0.85)

    for r in range(num_ranks):
        y_pos = (n * (r + 0.5)) / num_ranks
        ax.text(n * 0.96, y_pos, f"rank {r}", color="cyan", fontsize=8, ha="right", va="center")

    ax.set_title(f"MPI 2D Heat Diffusion (N={n}, p={num_ranks}, Step={step}, Max $\\Delta u$={max_diff:.2e})")
    ax.set_xlabel("X Grid Index")
    ax.set_ylabel("Y Grid Index")
    fig.colorbar(im, ax=ax, label="Temperature (°C)")
    fig.tight_layout()
    fig.savefig(frame_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--check_interval", type=int, default=50)
    parser.add_argument("--frame_interval", type=int, default=40)
    parser.add_argument("--frames_dir", type=str, default=None)
    parser.add_argument("--no_hotspot", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--save_final_grid", type=str, default=None)
    args = parser.parse_args()

    if args.warmup:
        u_d, v_d, off, cx, cy, r2 = initialize_local_strip(32, 8, 0, 4, True)
        jacobi_strip_step(u_d, v_d, 2, 9)
        enforce_strip_hotspot(v_d, off, 8, cx, cy, r2)
        compute_strip_max_diff(u_d, v_d, 2, 9)
        sys.exit(0)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n = args.size
    steps = args.steps
    use_hotspot = not args.no_hotspot

    if n % size != 0:
        if rank == 0:
            print(json.dumps({"error": f"Grid size {n} is not divisible by rank count {size}"}))
        sys.exit(1)

    rows = n // size
    up = rank - 1 if rank > 0 else MPI.PROC_NULL
    down = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    i_start = 2 if rank == 0 else 1
    i_end = rows if rank == size - 1 else rows + 1

    u, v, global_row_offset, cx, cy, r2 = initialize_local_strip(n, rows, rank, size, use_hotspot)
    jacobi_strip_step(u[:4, :4], v[:4, :4], 1, 3)
    u, v, global_row_offset, cx, cy, r2 = initialize_local_strip(n, rows, rank, size, use_hotspot)

    if args.animate:
        full_grid = np.empty((n, n), dtype=np.float64) if rank == 0 else None
        convergence_history = []
        current_diff = 100.0

        comm.Gather(np.ascontiguousarray(u[1:-1, :]), full_grid, root=0)
        if rank == 0 and args.frames_dir:
            os.makedirs(args.frames_dir, exist_ok=True)
            render_rank_frame(full_grid, 0, current_diff, os.path.join(args.frames_dir, f"frame_{0:05d}.png"), size)

        for step in range(1, steps + 1):
            comm.Sendrecv(u[1, :], dest=up, sendtag=10, recvbuf=u[-1, :], source=down, recvtag=10)
            comm.Sendrecv(u[-2, :], dest=down, sendtag=20, recvbuf=u[0, :], source=up, recvtag=20)

            jacobi_strip_step(u, v, i_start, i_end)
            if use_hotspot:
                enforce_strip_hotspot(v, global_row_offset, rows, cx, cy, r2)

            if step % args.check_interval == 0 or step % args.frame_interval == 0 or step == steps:
                loc_diff = compute_strip_max_diff(u, v, i_start, i_end)
                current_diff = float(comm.allreduce(loc_diff, op=MPI.MAX))
                if rank == 0:
                    convergence_history.append({"step": step, "max_diff": current_diff})

            u, v = v, u

            if step % args.frame_interval == 0 or step == steps:
                comm.Gather(np.ascontiguousarray(u[1:-1, :]), full_grid, root=0)
                if rank == 0 and args.frames_dir:
                    frame_file = os.path.join(args.frames_dir, f"frame_{step:05d}.png")
                    render_rank_frame(full_grid, step, current_diff, frame_file, size)

        if rank == 0:
            print(f"JSON_ANIM:{json.dumps({'convergence': convergence_history, 'final_diff': current_diff})}")
        sys.exit(0)

    t_halo = 0.0
    t_comp = 0.0
    t_allreduce = 0.0
    final_diff = 0.0

    comm.Barrier()
    t_total_start = MPI.Wtime()

    for step in range(1, steps + 1):
        t0 = MPI.Wtime()
        comm.Sendrecv(u[1, :], dest=up, sendtag=10, recvbuf=u[-1, :], source=down, recvtag=10)
        comm.Sendrecv(u[-2, :], dest=down, sendtag=20, recvbuf=u[0, :], source=up, recvtag=20)
        t_halo += MPI.Wtime() - t0

        t1 = MPI.Wtime()
        jacobi_strip_step(u, v, i_start, i_end)
        if use_hotspot:
            enforce_strip_hotspot(v, global_row_offset, rows, cx, cy, r2)
        t_comp += MPI.Wtime() - t1

        if step % args.check_interval == 0 or step == steps:
            t2 = MPI.Wtime()
            loc_diff = compute_strip_max_diff(u, v, i_start, i_end)
            final_diff = float(comm.allreduce(loc_diff, op=MPI.MAX))
            t_allreduce += MPI.Wtime() - t2

        u, v = v, u

    comm.Barrier()
    t_total = MPI.Wtime() - t_total_start

    max_halo = comm.reduce(t_halo, op=MPI.MAX, root=0)
    max_comp = comm.reduce(t_comp, op=MPI.MAX, root=0)
    max_allred = comm.reduce(t_allreduce, op=MPI.MAX, root=0)
    max_total = comm.reduce(t_total, op=MPI.MAX, root=0)

    full_grid = np.empty((n, n), dtype=np.float64) if rank == 0 else None
    comm.Gather(np.ascontiguousarray(u[1:-1, :]), full_grid, root=0)

    if rank == 0:
        if args.save_final_grid:
            np.save(args.save_final_grid, full_grid)
        comm_total = max_halo + max_allred
        payload = {
            "total_time_s": max_total,
            "compute_time_s": max_comp,
            "halo_time_s": max_halo,
            "allreduce_time_s": max_allred,
            "comm_time_s": comm_total,
            "halo_share_pct": (max_halo / max_total) * 100.0 if max_total > 0 else 0.0,
            "final_max_diff": final_diff
        }
        print(f"JSON_RESULT:{json.dumps(payload)}")


if __name__ == "__main__":
    main()
