import time
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from heat_seque import run_serial
from heat_threads import run_threads

# Tailles de grille demandées par le prof
GRID_SIZES = [512, 1024] # Ajoutez 2048 si vous avez le temps
THREAD_COUNTS = [1, 2, 4]
REPETITIONS = 3
MAX_STEPS = 500 # On réduit un peu pour que le benchmark ne prenne pas 1 heure

def measure_time(func, N, p=None):
    times = []
    for _ in range(REPETITIONS):
        if p is None:
            _, t = func(N=N, max_steps=MAX_STEPS)
        else:
            _, t = func(N=N, max_steps=MAX_STEPS, num_threads=p)
        times.append(t)
    return np.median(times)

def main():
    os.makedirs("results", exist_ok=True)
    results = {"serial": {}, "threads": {}}
    
    print("--- BENCHMARK MULTI-TAILLES EN COURS ---")
    
    for N in GRID_SIZES:
        print(f"\n=== TEST AVEC GRILLE {N}x{N} ===")
        # Séquentiel
        print(f"Test Séquentiel (N={N})...")
        t_serial = measure_time(run_serial, N)
        results["serial"][N] = t_serial
        print(f"  -> Temps : {t_serial:.4f} s")

        # Threads
        results["threads"][N] = {}
        for p in THREAD_COUNTS:
            print(f"Test Threads = {p} (N={N})...")
            t_threads = measure_time(run_threads, N, p)
            results["threads"][N][p] = t_threads
            print(f"  -> Temps : {t_threads:.4f} s")

    # --- SAUVEGARDE CSV ---
    with open("results/timings_multi.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["N", "Version", "Threads", "Time_s"])
        for N in GRID_SIZES:
            writer.writerow([N, "Serial", 1, results["serial"][N]])
            for p, t in results["threads"][N].items():
                writer.writerow([N, "V1_Threads", p, t])
    print("\nFichier results/timings_multi.csv sauvegardé !")

    # --- GRAPHIQUE 1 : Speedup pour chaque N ---
    plt.figure()
    for N in GRID_SIZES:
        t1 = results["serial"][N]
        speedup = [t1 / results["threads"][N][p] for p in THREAD_COUNTS]
        plt.plot(THREAD_COUNTS, speedup, marker='o', label=f'N={N}')
    
    plt.plot(THREAD_COUNTS, THREAD_COUNTS, linestyle='--', color='gray', label='Idéal')
    plt.xlabel("Nombre de Threads (p)")
    plt.ylabel("Speedup (S = T1 / Tp)")
    plt.title("Speedup selon la taille de la grille (Heat Diffusion)")
    plt.xticks(THREAD_COUNTS)
    plt.legend()
    plt.grid(True)
    plt.savefig("results/speedup_multi_heat.png", dpi=150)
    print("Graphique results/speedup_multi_heat.png sauvegardé !")
    plt.close()

    # --- GRAPHIQUE 2 : Temps total en fonction de N (avec 4 threads) ---
    plt.figure()
    serial_times = [results["serial"][N] for N in GRID_SIZES]
    threads_times = [results["threads"][N][4] for N in GRID_SIZES]
    
    x = np.arange(len(GRID_SIZES))
    width = 0.35
    plt.bar(x - width/2, serial_times, width, label='Séquentiel')
    plt.bar(x + width/2, threads_times, width, label='Threads (4)')
    
    plt.xlabel("Taille de la grille (N)")
    plt.ylabel("Temps (s)")
    plt.title("Comparaison des temps selon N")
    plt.xticks(x, [str(N) for N in GRID_SIZES])
    plt.legend()
    plt.grid(axis='y')
    plt.savefig("results/time_vs_N.png", dpi=150)
    print("Graphique results/time_vs_N.png sauvegardé !")
    plt.close()

if __name__ == "__main__":
    main()