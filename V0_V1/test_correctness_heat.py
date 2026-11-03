import numpy as np
from heat_seque import run_serial
from heat_threads import run_threads

N = 512
STEPS = 2000
TOL = 1e-4

print("Calcul de la Version 0 (Séquentiel)...")
grid_serial, _ = run_serial(N=N, max_steps=STEPS, tolerance=TOL)

print("\nCalcul de la Version 1 (Threads)...")
grid_threads, _ = run_threads(N=N, max_steps=STEPS, tolerance=TOL, num_threads=4)

# Comparaison avec tolérance (np.allclose)
if np.allclose(grid_serial, grid_threads, rtol=1e-5, atol=1e-5):
    print("\n✅ SUCCÈS : La Version 1 (Threads) donne EXACTEMENT le même résultat que la Version 0 !")
else:
    diff_max = np.max(np.abs(grid_serial - grid_threads))
    print(f"\n❌ ERREUR : Différence maximale = {diff_max}")