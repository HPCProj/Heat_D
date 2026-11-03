from mpi4py import MPI
import numpy as np
import matplotlib
matplotlib.use('Agg') # Pour sauvegarder sans ouvrir de fenêtre
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from numba import njit

@njit
def compute_stripe(u_local, v_local, rows, N):
    for i in range(1, rows + 1):
        for j in range(1, N - 1):
            v_local[i, j] = 0.25 * (u_local[i - 1, j] + u_local[i + 1, j] + u_local[i, j - 1] + u_local[i, j + 1])

def apply_boundaries(u_local, rows, N, rank, size):
    if rank == 0:
        u_local[1, :] = 100.0
    if rank == size - 1:
        u_local[rows, :] = 0.0
        
    u_local[1:rows+1, 0] = 60.0
    
    if rank == 0:
        u_local[2:rows+1, -1] = 0.0
        u_local[1, -1] = 100.0 # On garde le coin haut-droit à 100
    else:
        u_local[1:rows+1, -1] = 0.0

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 256 # On réduit un peu pour que l'animation soit rapide
    MAX_STEPS = 500
    FRAME_EVERY = 25
    
    rows = N // size
    u_local = np.zeros((rows + 2, N), dtype=np.float64)
    v_local = np.zeros((rows + 2, N), dtype=np.float64)

    apply_boundaries(u_local, rows, N, rank, size)
    apply_boundaries(v_local, rows, N, rank, size)

    north = rank - 1 if rank > 0 else MPI.PROC_NULL
    south = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    os.makedirs("results/frames_mpi", exist_ok=True)
    filenames = []

    for step in range(MAX_STEPS):
        # Échange des halos
        comm.Sendrecv(sendbuf=u_local[1, :], dest=north, sendtag=1,
                      recvbuf=u_local[0, :], source=north, recvtag=2)
        comm.Sendrecv(sendbuf=u_local[rows, :], dest=south, sendtag=2,
                      recvbuf=u_local[rows + 1, :], source=south, recvtag=1)

        # Calcul
        compute_stripe(u_local, v_local, rows, N)
        u_local, v_local = v_local, u_local
        
        apply_boundaries(u_local, rows, N, rank, size)

        # Sauvegarde de l'image toutes les FRAME_EVERY étapes
        if step % FRAME_EVERY == 0:
            local_interior = u_local[1:rows+1, :].copy()
            gathered_parts = comm.gather(local_interior, root=0)
            
            if rank == 0:
                full_grid = np.vstack(gathered_parts)
                
                plt.figure(figsize=(6, 6))
                plt.imshow(full_grid, cmap='inferno', origin='upper')
                plt.colorbar(label='Température (°C)')
                plt.title(f"Étape {step} - MPI ({size} processus)")
                
                # Tracer les lignes pointillées aux frontières des processus
                for r in range(1, size):
                    y_pos = r * rows
                    plt.axhline(y=y_pos, color='cyan', linestyle='--', linewidth=1.5)

                filename = f"results/frames_mpi/frame_{step:03d}.png"
                plt.savefig(filename, dpi=100)
                plt.close()
                filenames.append(filename)

    # Création du GIF sur le rang 0
    if rank == 0:
        print("Assemblage du GIF MPI...")
        with imageio.get_writer('results/heat_animation_mpi.gif', mode='I', duration=0.2) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        print("Animation MPI sauvegardée sous results/heat_animation_mpi.gif !")

if __name__ == "__main__":
    main()