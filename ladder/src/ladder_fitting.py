"""
===========================================================================
Λ³-DFT: Ladder γ Fitting with 2×6 Data (5 points!)
===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# UPDATED Data: 2×2 to 2×6 (5 points per U/t)
# ============================================================

ladder_data = {
    # U/t: {N_sites: alpha}
    0.5: {4: 0.003045, 6: 0.001693, 8: 0.001136, 10: 0.001227, 12: 0.001253},
    1.0: {4: 0.006024, 6: 0.004239, 8: 0.002278, 10: 0.001727, 12: 0.002503},
    1.5: {4: 0.009067, 6: 0.006509, 8: 0.003503, 10: 0.003484, 12: 0.003832},
    2.0: {4: 0.012279, 6: 0.012050, 8: 0.004896, 10: 0.004362, 12: 0.005308},
    2.5: {4: 0.015747, 6: 0.009552, 8: 0.006563, 10: 0.008484, 12: 0.006988},
    3.0: {4: 0.019534, 6: 0.011066, 8: 0.008646, 10: 0.011465, 12: 0.008925},
    3.5: {4: 0.023684, 6: 0.014836, 8: 0.011323, 10: 0.015132, 12: 0.011159},
    4.0: {4: 0.028217, 6: 0.030007, 8: 0.014798, 10: 0.019460, 12: 0.013716},
    5.0: {4: 0.038410, 6: 0.073222, 8: 0.024739, 10: 0.029588, 12: 0.019789},
}

# 1D chain γ for comparison
gamma_chain = {
    0.5: 2.16, 1.0: 2.12, 1.5: 2.06, 2.0: 1.97, 2.5: 1.85,
    3.0: 1.70, 3.5: 1.52, 4.0: 1.35, 5.0: 1.03
}

# ============================================================
# γ fitting for each U/t (NOW WITH 5 POINTS!)
# ============================================================

print("="*70)
print("γ_ladder fitting with 2×6 data (5 points per U/t)")
print("="*70)

results = []

for U in sorted(ladder_data.keys()):
    d = ladder_data[U]

    # All 5 data points
    N_all = np.array(list(d.keys()))
    alpha_all = np.array(list(d.values()))

    log_N = np.log(N_all)
    log_alpha = np.log(alpha_all)

    # Fit: log(α) = -γ * log(N) + const
    coeffs_all = np.polyfit(log_N, log_alpha, 1)
    gamma_all = -coeffs_all[0]

    # R² calculation
    pred = np.polyval(coeffs_all, log_N)
    ss_res = np.sum((log_alpha - pred)**2)
    ss_tot = np.sum((log_alpha - np.mean(log_alpha))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Exclude outliers (N=4, N=6 for strong coupling)
    if U >= 2.5:
        N_large = np.array([8, 10, 12])
        alpha_large = np.array([d[8], d[10], d[12]])
        coeffs_large = np.polyfit(np.log(N_large), np.log(alpha_large), 1)
        gamma_large = -coeffs_large[0]
    else:
        gamma_large = gamma_all

    gamma_1d = gamma_chain.get(U, np.nan)
    delta_gamma = gamma_1d - gamma_all

    results.append({
        'U': U,
        'gamma_all': gamma_all,
        'gamma_large': gamma_large,
        'gamma_chain': gamma_1d,
        'delta': delta_gamma,
        'r2': r2
    })

    print(f"U/t={U:.1f}: γ_all={gamma_all:.3f}, γ_large={gamma_large:.3f}, "
          f"γ_chain={gamma_1d:.2f}, Δγ={delta_gamma:.2f}, R²={r2:.3f}")

# ============================================================
# Summary table
# ============================================================

print("\n" + "="*70)
print("SUMMARY: γ comparison (chain vs ladder) - UPDATED WITH 2×6")
print("="*70)
print(f"{'U/t':>5} {'γ_chain':>8} {'γ_ladder':>9} {'γ_large':>9} {'Δγ':>6} {'R²':>6}")
print("-"*50)

for r in results:
    print(f"{r['U']:>5.1f} {r['gamma_chain']:>8.2f} {r['gamma_all']:>9.3f} "
          f"{r['gamma_large']:>9.3f} {r['delta']:>6.2f} {r['r2']:>6.3f}")

# ============================================================
# Fit γ_ladder(U/t) relationship
# ============================================================

print("\n" + "="*70)
print("γ_ladder(U/t) fitting - UPDATED")
print("="*70)

# Use U/t ≤ 2.0 (weak correlation)
U_weak = np.array([r['U'] for r in results if r['U'] <= 2.0])
gamma_weak = np.array([r['gamma_all'] for r in results if r['U'] <= 2.0])

coeffs_gamma = np.polyfit(U_weak, gamma_weak, 1)
print(f"\nWeak correlation (U/t ≤ 2.0):")
print(f"  γ_ladder(U/t) = {coeffs_gamma[1]:.3f} + ({coeffs_gamma[0]:.3f}) × (U/t)")

# All data
U_all = np.array([r['U'] for r in results])
gamma_all_fit = np.array([r['gamma_all'] for r in results])
coeffs_all = np.polyfit(U_all, gamma_all_fit, 1)
print(f"\nAll U/t:")
print(f"  γ_ladder(U/t) = {coeffs_all[1]:.3f} + ({coeffs_all[0]:.3f}) × (U/t)")

print(f"\nComparison:")
print(f"  γ_chain(U/t)  = 2.43 - 0.27 × (U/t)")
print(f"  γ_ladder(U/t) = {coeffs_gamma[1]:.2f} + {coeffs_gamma[0]:.2f} × (U/t)")

# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (a) γ vs U/t
ax1 = axes[0]
U_plot = np.array([r['U'] for r in results])
gamma_chain_plot = np.array([r['gamma_chain'] for r in results])
gamma_ladder_plot = np.array([r['gamma_all'] for r in results])

ax1.plot(U_plot, gamma_chain_plot, 'bo-', markersize=10, linewidth=2, label='1D Chain')
ax1.plot(U_plot, gamma_ladder_plot, 'rs-', markersize=10, linewidth=2, label='2-Leg Ladder (2×6)')

# Fit lines
U_fit = np.linspace(0, 6, 100)
gamma_chain_fit = 2.43 - 0.27 * U_fit
gamma_ladder_fit = coeffs_gamma[1] + coeffs_gamma[0] * U_fit

ax1.plot(U_fit, gamma_chain_fit, 'b--', linewidth=2, alpha=0.7)
ax1.plot(U_fit, gamma_ladder_fit, 'r--', linewidth=2, alpha=0.7)

ax1.axvspan(0, 2.0, alpha=0.1, color='green', label='Weak correlation')
ax1.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('U/t', fontsize=14)
ax1.set_ylabel('γ (correlation dimension)', fontsize=14)
ax1.set_title('(a) γ vs U/t: Chain vs Ladder (Updated with 2×6)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_xlim(0, 5.5)
ax1.set_ylim(0, 2.5)
ax1.grid(True, alpha=0.3)

# (b) R² vs U/t
ax2 = axes[1]
r2_plot = np.array([r['r2'] for r in results])
ax2.plot(U_plot, r2_plot, 'mo-', markersize=12, linewidth=2)
ax2.axhline(0.8, color='green', linestyle='--', linewidth=2, label='R² = 0.8 threshold')
ax2.axvspan(2.0, 3.0, alpha=0.3, color='red', label='Critical region')
ax2.set_xlabel('U/t', fontsize=14)
ax2.set_ylabel('R² (scaling quality)', fontsize=14)
ax2.set_title('(b) Scaling Quality vs U/t', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_xlim(0, 5.5)
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3)

plt.suptitle('Λ³-DFT: Ladder Analysis with 2×6 Data (5 points per fit)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('ladder_gamma_updated.png', dpi=150, bbox_inches='tight')
print("\n[Saved] ladder_gamma_updated.png")
plt.show()
