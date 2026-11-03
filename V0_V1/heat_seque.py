import numpy as np
from numba import njit
import time
import matplotlib.pyplot as plt

# Le cœur du calcul (Kernel Numba)
# u est la grille actuelle, v est la grille future
@njit
def jacobi_step(u, v):
    n, m = u.shape
    # On parcourt l'intérieur de la grille (on ignore les bords)
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            # La case vaut la moyenne des 4 voisins (haut, bas, gauche, droite)
            v[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])

def run_serial(N=512, max_steps=2000, tolerance=1e-4):
    # 1. Initialisation de la grille
    # On crée deux grilles : u (actuelle) et v (future)
    u = np.zeros((N, N), dtype=np.float64)
    v = np.zeros((N, N), dtype=np.float64)

    # 2. Conditions aux bords (Boundaries)
    # Haut = 100, Gauche = 60, Droite et Bas = 0 (déjà à 0 par défaut)
    u[0, :] = 100.0
    u[:, 0] = 60.0
    v[0, :] = 100.0
    v[:, 0] = 60.0

    # 3. Warm-up Numba (pour compiler la fonction avant de mesurer)
    dummy_u = np.zeros((10, 10))
    dummy_v = np.zeros((10, 10))
    jacobi_step(dummy_u, dummy_v)

    print(f"Démarrage de la simulation séquentielle (Grille {N}x{N})...")
    
    # 4. Boucle temporelle
    t0 = time.perf_counter()
    
    for step in range(max_steps):
        # On calcule la grille future
        jacobi_step(u, v)
        
        # On calcule le changement maximum pour vérifier la convergence
        diff = np.max(np.abs(u - v))
        
        # On échange les grilles : v devient la grille actuelle pour le prochain tour
        u, v = v, u
        
        # Vérification de la convergence
        if diff < tolerance:
            print(f"Convergence atteinte à l'étape {step} !")
            break
            
    t1 = time.perf_counter()
    temps = t1 - t0
    print(f"Temps de calcul (Séquentiel) : {t1 - t0:.4f} secondes")
    print(f"Changement final : {diff:.6f}")

    return u, temps 

if __name__ == "__main__":
    # On lance la simulation
    final_grid = run_serial(N=512, max_steps=2000)
    
    # Affichage du résultat avec une palette de couleurs de chaleur
    plt.figure(figsize=(6, 6))
    # 'inferno' va du noir (froid) au jaune/blanc (chaud)
    plt.imshow(final_grid, cmap='inferno', origin='upper')
    plt.colorbar(label='Température (°C)')
    plt.title("Propagation de la chaleur (Version 0 - Séquentiel)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("heat_sequentiel.png", dpi=150)
    print("Image sauvegardée sous heat_sequentiel.png")
    plt.show()