from mpi4py import MPI
import numpy as np
import time
import os
from numba import njit

@njit
def compute_stripe(u_local, v_local, rows, N):
    for i in range(1, rows + 1):
        for j in range(1, N - 1):
            v_local[i, j] = 0.25 * (u_local[i - 1, j] + u_local[i + 1, j] + u_local[i, j - 1] + u_local[i, j + 1])

def apply_boundaries(u_local, rows, N, rank, size):
    """Applique les conditions aux bords UNIQUEMENT sur les lignes utiles."""
    # 1. Bords haut et bas d'abord (sur toute la largeur)
    if rank == 0:
        u_local[1, :] = 100.0
    if rank == size - 1:
        u_local[rows, :] = 0.0
        
    # 2. Bords gauche et droit ensuite (sur les lignes utiles)
    #    Le gauche écrase le coin haut-gauche et bas-gauche (comme dans la V0)
    u_local[1:rows+1, 0] = 60.0
    #    Le droit écrase le coin bas-droit mais PAS le haut-droit (pour garder 100)
    if rank == 0:
        u_local[2:rows+1, -1] = 0.0  # On garde u_local[1, -1] à 100
    else:
        u_local[1:rows+1, -1] = 0.0
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 512
    MAX_STEPS = 500
    TOL = 1e-4
    
    if N % size != 0:
        if rank == 0:
            print(f"Erreur : N={N} doit être divisible par le nombre de processus ({size}).")
        return

    rows = N // size
    u_local = np.zeros((rows + 2, N), dtype=np.float64)
    v_local = np.zeros((rows + 2, N), dtype=np.float64)

    # Initialisation des bords
    apply_boundaries(u_local, rows, N, rank, size)
    apply_boundaries(v_local, rows, N, rank, size)

    north = rank - 1 if rank > 0 else MPI.PROC_NULL
    south = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    t_start = time.perf_counter()
    time_compute = 0.0
    time_comm = 0.0

    for step in range(MAX_STEPS):
        # Échange des halos
        t_comm_0 = time.perf_counter()
        comm.Sendrecv(sendbuf=u_local[1, :], dest=north, sendtag=1,
                      recvbuf=u_local[0, :], source=north, recvtag=2)
        comm.Sendrecv(sendbuf=u_local[rows, :], dest=south, sendtag=2,
                      recvbuf=u_local[rows + 1, :], source=south, recvtag=1)
        t_comm_1 = time.perf_counter()
        time_comm += (t_comm_1 - t_comm_0)

        # Calcul local
        t_comp_0 = time.perf_counter()
        compute_stripe(u_local, v_local, rows, N)
        t_comp_1 = time.perf_counter()
        time_compute += (t_comp_1 - t_comp_0)

        # Différence locale
        diff_local = np.max(np.abs(u_local[1:rows+1, 1:N-1] - v_local[1:rows+1, 1:N-1]))
        diff_global = comm.allreduce(diff_local, op=MPI.MAX)

        # Échange des grilles
        u_local, v_local = v_local, u_local
        
        # Remise des conditions aux bords
        apply_boundaries(u_local, rows, N, rank, size)

        if diff_global < TOL:
            break

    t_end = time.perf_counter()
    
    # Sauvegarde
    local_interior = u_local[1:rows+1, :].copy()
    gathered_parts = comm.gather(local_interior, root=0)

    if rank == 0:
        full_grid = np.vstack(gathered_parts)
        print(f"Temps total MPI : {t_end - t_start:.4f} s")
        print(f"Temps de calcul (Compute) : {time_compute:.4f} s")
        print(f"Temps de communication (Halo) : {time_comm:.4f} s")
        print(f"Part de la communication : {(time_comm / (t_end - t_start)) * 100:.2f}%")
        os.makedirs("results", exist_ok=True)
        np.save("results/heat_mpi_final.npy", full_grid)
        print(f"Grille finale sauvegardée (taille: {full_grid.shape})")

if __name__ == "__main__":
    main()