"""
===========================================================================
Λ³-DFT: Doping Dependence Analysis - γ(d) Fitting
===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Complete Data
# ============================================================

data = {
    # U/t: {Lx: {d: alpha}}
    0.5: {
        3: {0: 0.001640, 1: 0.001683, 2: 0.002052},
        4: {0: 0.001136, 1: 0.001141, 2: 0.001253},
        5: {0: 0.001208, 1: 0.000872, 2: 0.000912},
    },
    1.0: {
        3: {0: 0.003352, 1: 0.003339, 2: 0.004068},
        4: {0: 0.002328, 1: 0.002288, 2: 0.002591},
        5: {0: 0.002446, 1: 0.001760, 2: 0.001821},
    },
    1.5: {
        3: {0: 0.005351, 1: 0.005074, 2: 0.006198},
        4: {0: 0.003728, 1: 0.003536, 2: 0.004145},
        5: {0: 0.003834, 1: 0.002752, 2: 0.002797},
    },
    2.0: {
        3: {0: 0.007883, 1: 0.006982, 2: 0.008520},
        4: {0: 0.005518, 1: 0.004979, 2: 0.005977},
        5: {0: 0.005488, 1: 0.003935, 2: 0.003898},
    },
    2.5: {
        3: {0: 0.011214, 1: 0.009151, 2: 0.011066},
        4: {0: 0.007910, 1: 0.006709, 2: 0.008113},
        5: {0: 0.007519, 1: 0.005396, 2: 0.005178},
    },
}

# ============================================================
# γ fitting for each (U/t, d)
# ============================================================

print("="*70)
print("γ(U/t, d) Fitting Results")
print("="*70)

# N_sites for scaling
N_sites = {3: 6, 4: 8, 5: 10}

results = {}

for U in sorted(data.keys()):
    results[U] = {}
    print(f"\n--- U/t = {U} ---")
    print(f"{'d':>3} {'γ':>8} {'R²':>8} {'α(Lx=3)':>10} {'α(Lx=4)':>10} {'α(Lx=5)':>10}")
    print("-"*55)

    for d in [0, 1, 2]:
        # Get alpha values
        Lx_list = [3, 4, 5]
        Ns = np.array([N_sites[Lx] for Lx in Lx_list])
        alphas = np.array([data[U][Lx][d] for Lx in Lx_list])

        # Log-log fit: log(α) = -γ log(N) + const
        log_N = np.log(Ns)
        log_alpha = np.log(alphas)
        coeffs = np.polyfit(log_N, log_alpha, 1)
        gamma = -coeffs[0]

        # R²
        pred = np.polyval(coeffs, log_N)
        ss_res = np.sum((log_alpha - pred)**2)
        ss_tot = np.sum((log_alpha - np.mean(log_alpha))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        results[U][d] = {'gamma': gamma, 'r2': r2, 'alphas': alphas}

        print(f"{d:>3} {gamma:>8.3f} {r2:>8.3f} {alphas[0]:>10.6f} {alphas[1]:>10.6f} {alphas[2]:>10.6f}")

# ============================================================
# Summary: γ(d) for each U/t
# ============================================================

print("\n" + "="*70)
print("SUMMARY: γ vs Doping")
print("="*70)
print(f"{'U/t':>5} {'γ(d=0)':>8} {'γ(d=1)':>8} {'γ(d=2)':>8} {'min at':>8}")
print("-"*45)

for U in sorted(results.keys()):
    g0 = results[U][0]['gamma']
    g1 = results[U][1]['gamma']
    g2 = results[U][2]['gamma']

    # Find minimum
    gammas = [g0, g1, g2]
    min_idx = np.argmin(gammas)
    min_d = ['d=0', 'd=1', 'd=2'][min_idx]

    marker = '★' if min_idx == 1 else ''
    print(f"{U:>5.1f} {g0:>8.3f} {g1:>8.3f} {g2:>8.3f} {min_d:>6} {marker}")

# ============================================================
# Key finding: α at Lx=5 (largest system)
# ============================================================

print("\n" + "="*70)
print("α at Lx=5 (most reliable for trends)")
print("="*70)
print(f"{'U/t':>5} {'α(d=0)':>10} {'α(d=1)':>10} {'α(d=2)':>10} {'min at':>8}")
print("-"*50)

for U in sorted(data.keys()):
    a0 = data[U][5][0]
    a1 = data[U][5][1]
    a2 = data[U][5][2]

    alphas = [a0, a1, a2]
    min_idx = np.argmin(alphas)
    min_d = ['d=0', 'd=1', 'd=2'][min_idx]

    marker = '★' if min_idx == 1 else ''
    print(f"{U:>5.1f} {a0:>10.6f} {a1:>10.6f} {a2:>10.6f} {min_d:>6} {marker}")
