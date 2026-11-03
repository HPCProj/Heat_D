import numpy as np
from numba import njit, prange, set_num_threads
import time
import matplotlib.pyplot as plt
import os

# On force Numba à utiliser OpenMP (comme demandé par le prof)
os.environ['NUMBA_THREADING_LAYER'] = 'omp'

# Le kernel parallèle
@njit(parallel=True)
def jacobi_step_threads(u, v):
    n, m = u.shape
    # On utilise prange pour paralléliser la boucle des lignes
    for i in prange(1, n - 1):
        for j in range(1, m - 1):
            v[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

def run_threads(N=512, max_steps=2000, tolerance=1e-4, num_threads=4):
    u = np.zeros((N, N), dtype=np.float64)
    v = np.zeros((N, N), dtype=np.float64)

    # Conditions aux bords
    u[0, :] = 100.0
    u[:, 0] = 60.0
    v[0, :] = 100.0
    v[:, 0] = 60.0

    # Warm-up Numba (avec 1 thread pour compiler)
    set_num_threads(1)
    dummy_u = np.zeros((10, 10))
    dummy_v = np.zeros((10, 10))
    jacobi_step_threads(dummy_u, dummy_v)
    
    # On règle le nombre de threads demandé
    set_num_threads(num_threads)

    print(f"Simulation Threads ({num_threads} threads, Grille {N}x{N})...")
    
    t0 = time.perf_counter()
    
    for step in range(max_steps):
        jacobi_step_threads(u, v)
        
        # Calcul de la différence max
        diff = np.max(np.abs(u - v))
        
        # Échange des grilles
        u, v = v, u
        
        if diff < tolerance:
            print(f"  -> Convergence atteinte à l'étape {step}")
            break
            
    t1 = time.perf_counter()
    temps = t1 - t0
    print(f"  -> Temps de calcul : {temps:.4f} secondes")
    
    return u, temps

if __name__ == "__main__":
    # Exécution simple pour vérifier que ça marche et générer l'image
    final_grid, t = run_threads(N=512, max_steps=2000, num_threads=4)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(final_grid, cmap='inferno', origin='upper')
    plt.colorbar(label='Température (°C)')
    plt.title("Propagation de la chaleur (Version 1 - Threads)")
    plt.savefig("heat_threads.png", dpi=150)
    print("Image sauvegardée sous heat_threads.png")
    plt.show()