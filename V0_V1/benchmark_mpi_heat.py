import subprocess
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import csv

def run_mpi(num_procs):
    cmd = ["mpiexec", "--oversubscribe", "-n", str(num_procs), "python3", "heat_mpi.py"]
    # On répète 3 fois
    times_total = []
    times_comm = []
    for _ in range(3):
        result = subprocess.run(cmd, capture_output=True, text=True)
        match_total = re.search(r"Temps total MPI : ([\d\.]+)", result.stdout)
        match_comm = re.search(r"Temps de communication \(Halo\) : ([\d\.]+)", result.stdout)
        if match_total and match_comm:
            times_total.append(float(match_total.group(1)))
            times_comm.append(float(match_comm.group(1)))
    return np.median(times_total), np.median(times_comm)

def main():
    procs = [1, 2, 4]
    results = {}
    
    for p in procs:
        print(f"Test avec {p} processus...")
        t_total, t_comm = run_mpi(p)
        results[p] = {"total": t_total, "comm": t_comm}
        
    # CSV
    with open("results/mpi_heat_timings.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Processus", "Temps_Total_s", "Temps_Comm_s"])
        for p in procs:
            writer.writerow([p, results[p]["total"], results[p]["comm"]])
            
    # Graphique 1 : Speedup
    plt.figure()
    t1 = results[1]["total"]
    speedups = [t1 / results[p]["total"] for p in procs]
    plt.plot(procs, speedups, marker='o', label='MPI')
    plt.plot(procs, procs, linestyle='--', color='gray', label='Idéal')
    plt.xlabel("Nombre de Processus (p)")
    plt.ylabel("Speedup")
    plt.title("Speedup MPI (Heat Diffusion)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/speedup_heat_mpi.png", dpi=150)
    
    # Graphique 2 : Part de communication
    plt.figure()
    comm_shares = [(results[p]["comm"] / results[p]["total"]) * 100 for p in procs]
    plt.plot(procs, comm_shares, marker='s', color='red')
    plt.xlabel("Nombre de Processus (p)")
    plt.ylabel("Temps de communication (%)")
    plt.title("Part de la communication (Halo Exchange)")
    plt.grid(True)
    plt.savefig("results/comm_share_heat.png", dpi=150)
    print("Graphiques générés !")

if __name__ == "__main__":
    main()