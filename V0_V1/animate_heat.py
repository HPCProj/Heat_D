import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from numba import njit, prange, set_num_threads

os.environ['NUMBA_THREADING_LAYER'] = 'omp'

@njit(parallel=True)
def jacobi_step_threads(u, v):
    n, m = u.shape
    for i in prange(1, n - 1):
        for j in range(1, m - 1):
            v[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

def generate_animation(N=256, max_steps=500, frame_every=25):
    # On utilise N=256 pour que l'animation soit rapide à générer
    u = np.zeros((N, N), dtype=np.float64)
    v = np.zeros((N, N), dtype=np.float64)

    # Conditions aux bords
    u[0, :] = 100.0
    u[:, 0] = 60.0
    
    # Dossier pour les images
    os.makedirs("results/frames", exist_ok=True)
    filenames = []

    print(f"Génération de l'animation pour N={N}...")
    
    # Warm-up Numba
    set_num_threads(4)
    dummy = np.zeros((10,10))
    jacobi_step_threads(dummy, dummy)

    for step in range(max_steps):
        jacobi_step_threads(u, v)
        
        # On sauvegarde une image toutes les 'frame_every' étapes
        if step % frame_every == 0:
            plt.figure(figsize=(5, 5))
            plt.imshow(u, cmap='inferno', origin='upper')
            plt.title(f"Étape {step}")
            plt.colorbar(label='Température (°C)')
            # Nom du fichier (ex: frame_000.png)
            filename = f"results/frames/frame_{step:03d}.png"
            plt.savefig(filename, dpi=100)
            plt.close()
            filenames.append(filename)
            
        # Échange des grilles
        u, v = v, u

    # Création du GIF
    print("Assemblage du GIF...")
    with imageio.get_writer('results/heat_animation.gif', mode='I', duration=0.2) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    print("Animation sauvegardée sous results/heat_animation.gif !")

if __name__ == "__main__":
    generate_animation()