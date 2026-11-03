import numpy as np
from heat_seque import run_serial

N = 512
STEPS = 500
TOL = 1e-4

print("Calcul de la Version 0 (Séquentielle) pour référence...")
grid_serial, _ = run_serial(N=N, max_steps=STEPS, tolerance=TOL)

print("Chargement de l'image MPI...")
grid_mpi = np.load("results/heat_mpi_final.npy")

# On utilise une tolérance plus large pour ignorer les micro-arrondis flottants
if np.allclose(grid_serial, grid_mpi, rtol=1e-3, atol=1e-3):
    print("\n✅ SUCCÈS : La Version 2 (MPI) est visuellement identique à la Version 0 !")
else:
    diff_max = np.max(np.abs(grid_serial - grid_mpi))
    print(f"\n❌ ERREUR : Différence maximale = {diff_max}")
    
    # Si ça échoue, on regarde où est le problème
    print("\n--- Vérification des coins ---")
    print("V0 coin haut-gauche :", grid_serial[0, 0], "| V2 :", grid_mpi[0, 0])
    print("V0 coin haut-droit  :", grid_serial[0, -1], "| V2 :", grid_mpi[0, -1])
    print("V0 coin bas-gauche  :", grid_serial[-1, 0], "| V2 :", grid_mpi[-1, 0])