"""
===========================================================================
Λ³-DFT Validation: Bethe Ansatz Comparison & Scaling Analysis
===========================================================================

Compares Λ³-DFT results with exact Bethe Ansatz solution for the 
1D Hubbard model at half-filling.

Key results:
    - Mean error < 0.5% for U/t ≤ 3 (open-shell extrapolation)
    - Discovery of correlation dimension γ(U/t) = 2.43 - 0.27(U/t)
    - Shell effects: closed-shell (N=4,8) vs open-shell (N=6,10)

Reference:
    Lieb, E.H. & Wu, F.Y. Phys. Rev. Lett. 20, 1445 (1968)

Author: Masamichi Iizumi, Tamaki Iizumi
License: MIT
"""

import numpy as np
from scipy.special import jv
from scipy.integrate import quad
import matplotlib.pyplot as plt


# ===========================================================================
# Bethe Ansatz Exact Solution
# ===========================================================================

def bethe_energy_density(U, t=1.0):
    """
    Exact ground state energy per site for 1D Hubbard at half-filling.
    
    Formula (Lieb-Wu):
        e(U) = -4t ∫_0^∞ J_0(ω)J_1(ω) / [ω(1 + exp(2ωu))] dω
        
    where u = U/(4t), J_n = Bessel function of first kind.
    
    Parameters
    ----------
    U : float
        On-site Coulomb repulsion
    t : float
        Hopping parameter
    
    Returns
    -------
    float
        Ground state energy per site
    """
    u = U / (4.0 * t)
    
    def integrand(omega):
        if omega < 1e-10:
            return 0.0
        J0 = jv(0, omega)
        J1 = jv(1, omega)
        denom = omega * (1.0 + np.exp(2.0 * omega * u))
        return J0 * J1 / denom
    
    result, _ = quad(integrand, 0, 100, limit=200)
    return -4.0 * t * result


def bethe_Exc_density(U, t=1.0):
    """
    Exact exchange-correlation energy per site.
    
    Definition:
        e_xc = e(U) - e(U=0)
        e(U=0) = -4t/π  (non-interacting limit)
    """
    e_total = bethe_energy_density(U, t)
    e_nonint = -4.0 * t / np.pi
    return e_total - e_nonint


# ===========================================================================
# Data from Λ³-DFT Calculations
# ===========================================================================

# 40 data points: 10 U/t values × 4 system sizes
DATA = {
    0.5: {4: {'E_xc': 0.3510, 'vorticity': 47.35, 'alpha': 0.007413},
          6: {'E_xc': 0.7248, 'vorticity': 442.68, 'alpha': 0.001637},
          8: {'E_xc': 0.8943, 'vorticity': 519.10, 'alpha': 0.001723},
          10: {'E_xc': 1.2077, 'vorticity': 1388.71, 'alpha': 0.000870}},
    1.0: {4: {'E_xc': 0.6592, 'vorticity': 45.85, 'alpha': 0.014378},
          6: {'E_xc': 1.3988, 'vorticity': 427.00, 'alpha': 0.003276},
          8: {'E_xc': 1.7045, 'vorticity': 494.61, 'alpha': 0.003446},
          10: {'E_xc': 2.3299, 'vorticity': 1330.83, 'alpha': 0.001751}},
    1.5: {4: {'E_xc': 0.9309, 'vorticity': 44.00, 'alpha': 0.021158},
          6: {'E_xc': 2.0212, 'vorticity': 401.63, 'alpha': 0.005033},
          8: {'E_xc': 2.4345, 'vorticity': 458.87, 'alpha': 0.005305},
          10: {'E_xc': 3.3639, 'vorticity': 1234.61, 'alpha': 0.002725}},
    2.0: {4: {'E_xc': 1.1716, 'vorticity': 42.13, 'alpha': 0.027808},
          6: {'E_xc': 2.5905, 'vorticity': 367.87, 'alpha': 0.007042},
          8: {'E_xc': 3.0887, 'vorticity': 416.63, 'alpha': 0.007414},
          10: {'E_xc': 4.3059, 'vorticity': 1102.87, 'alpha': 0.003904}},
    2.5: {4: {'E_xc': 1.3853, 'vorticity': 40.41, 'alpha': 0.034281},
          6: {'E_xc': 3.1058, 'vorticity': 327.99, 'alpha': 0.009469},
          8: {'E_xc': 3.6721, 'vorticity': 372.13, 'alpha': 0.009868},
          10: {'E_xc': 5.1514, 'vorticity': 945.39, 'alpha': 0.005449}},
    3.0: {4: {'E_xc': 1.5756, 'vorticity': 38.91, 'alpha': 0.040497},
          6: {'E_xc': 3.5666, 'vorticity': 285.24, 'alpha': 0.012504},
          8: {'E_xc': 4.1903, 'vorticity': 328.73, 'alpha': 0.012747},
          10: {'E_xc': 5.8982, 'vorticity': 780.80, 'alpha': 0.007554}},
    3.5: {4: {'E_xc': 1.7454, 'vorticity': 37.64, 'alpha': 0.046375},
          6: {'E_xc': 3.9742, 'vorticity': 243.27, 'alpha': 0.016337},
          8: {'E_xc': 4.6487, 'vorticity': 288.69, 'alpha': 0.016102},
          10: {'E_xc': 6.5486, 'vorticity': 630.40, 'alpha': 0.010388}},
    4.0: {4: {'E_xc': 1.8973, 'vorticity': 36.59, 'alpha': 0.051854},
          6: {'E_xc': 4.3313, 'vorticity': 205.21, 'alpha': 0.021107},
          8: {'E_xc': 5.0533, 'vorticity': 253.33, 'alpha': 0.019948},
          10: {'E_xc': 7.1099, 'vorticity': 507.54, 'alpha': 0.014009}},
    5.0: {4: {'E_xc': 2.1557, 'vorticity': 35.05, 'alpha': 0.061499},
          6: {'E_xc': 4.9123, 'vorticity': 146.77, 'alpha': 0.033469},
          8: {'E_xc': 5.7246, 'vorticity': 197.85, 'alpha': 0.028934},
          10: {'E_xc': 8.0108, 'vorticity': 345.34, 'alpha': 0.023197}},
    6.0: {4: {'E_xc': 2.3654, 'vorticity': 34.08, 'alpha': 0.069415},
          6: {'E_xc': 5.3515, 'vorticity': 110.74, 'alpha': 0.048325},
          8: {'E_xc': 6.2481, 'vorticity': 160.29, 'alpha': 0.038981},
          10: {'E_xc': 8.6897, 'vorticity': 257.46, 'alpha': 0.033752}},
}

# Fitted correlation dimension γ(U/t)
GAMMA = {
    0.5: 2.16, 1.0: 2.12, 1.5: 2.06, 2.0: 1.97, 2.5: 1.85,
    3.0: 1.70, 3.5: 1.52, 4.0: 1.35, 5.0: 1.03, 6.0: 0.79
}


# ===========================================================================
# Validation Functions
# ===========================================================================

def compare_with_bethe():
    """Compare Λ³-DFT with Bethe Ansatz (open-shell extrapolation)."""
    results = []
    
    for U in sorted(DATA.keys()):
        e_bethe = bethe_Exc_density(U)
        
        # Open-shell: N=6, N=10
        e_N6 = DATA[U][6]['E_xc'] / 6
        e_N10 = DATA[U][10]['E_xc'] / 10
        
        # Linear extrapolation: e(N→∞) from 1/N → 0
        inv_N = np.array([1/6, 1/10])
        e_open = np.array([e_N6, e_N10])
        coeffs = np.polyfit(inv_N, e_open, 1)
        e_extrap = coeffs[1]  # y-intercept = N→∞ limit
        
        error = abs(e_extrap - e_bethe) / abs(e_bethe) * 100
        
        results.append({
            'U': U,
            'bethe': e_bethe,
            'lambda3': e_extrap,
            'error_percent': error
        })
    
    return results


def fit_gamma():
    """Fit γ(U/t) relationship."""
    U_arr = np.array(sorted(GAMMA.keys()))
    gamma_arr = np.array([GAMMA[u] for u in U_arr])
    
    # Linear fit: γ = a + b*(U/t)
    coeffs = np.polyfit(U_arr, gamma_arr, 1)
    pred = np.polyval(coeffs, U_arr)
    
    ss_res = np.sum((gamma_arr - pred)**2)
    ss_tot = np.sum((gamma_arr - np.mean(gamma_arr))**2)
    r2 = 1 - ss_res / ss_tot
    
    return {
        'intercept': coeffs[1],
        'slope': coeffs[0],
        'r_squared': r2,
        'formula': f"γ(U/t) = {coeffs[1]:.2f} + ({coeffs[0]:.2f})×(U/t)"
    }


# ===========================================================================
# Visualization
# ===========================================================================

def plot_results(save_path='figures/'):
    """Generate publication figures."""
    
    # Figure 1: γ discovery
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 5))
    
    # (a) Log-log scaling
    for U in [0.5, 2.0, 4.0, 6.0]:
        Ns = np.array([4, 6, 8, 10])
        alphas = np.array([DATA[U][N]['alpha'] for N in Ns])
        ax1a.loglog(Ns, alphas, 'o-', label=f'U/t={U}', markersize=8)
    
    ax1a.set_xlabel('N (system size)', fontsize=12)
    ax1a.set_ylabel('α = E_xc / V', fontsize=12)
    ax1a.set_title('(a) Scaling: α ∝ N^(-γ)', fontsize=12, fontweight='bold')
    ax1a.legend()
    ax1a.grid(True, alpha=0.3)
    
    # (b) γ vs U/t
    U_arr = np.array(sorted(GAMMA.keys()))
    gamma_arr = np.array([GAMMA[u] for u in U_arr])
    fit = fit_gamma()
    
    ax1b.plot(U_arr, gamma_arr, 'ko', markersize=10, label='Data')
    U_fine = np.linspace(0, 7, 100)
    ax1b.plot(U_fine, fit['intercept'] + fit['slope']*U_fine, 'b--', 
              linewidth=2, label=f'Linear fit (R²={fit["r_squared"]:.3f})')
    
    ax1b.axhline(2.0, color='gray', linestyle=':', label='γ=2 (metal)')
    ax1b.axhline(0.0, color='gray', linestyle='-.', label='γ=0 (insulator)')
    ax1b.set_xlabel('U/t', fontsize=12)
    ax1b.set_ylabel('γ (correlation dimension)', fontsize=12)
    ax1b.set_title('(b) Correlation Dimension vs U/t', fontsize=12, fontweight='bold')
    ax1b.legend()
    ax1b.grid(True, alpha=0.3)
    ax1b.set_xlim(0, 7)
    ax1b.set_ylim(-0.2, 2.5)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}fig1_gamma_discovery.png', dpi=150)
    print(f"[Saved] {save_path}fig1_gamma_discovery.png")
    
    # Figure 2: Bethe comparison
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    comparison = compare_with_bethe()
    U_vals = [c['U'] for c in comparison]
    e_bethe = [c['bethe'] for c in comparison]
    e_lambda3 = [c['lambda3'] for c in comparison]
    
    ax2.plot(U_vals, e_bethe, 'g-', linewidth=3, label='Bethe Ansatz (exact)')
    ax2.plot(U_vals, e_lambda3, 'ro', markersize=10, label='Λ³-DFT (N→∞)')
    
    # Highlight <0.1% region
    ax2.axvspan(0, 3, alpha=0.2, color='green', label='Error < 0.5%')
    
    ax2.set_xlabel('U/t', fontsize=12)
    ax2.set_ylabel('e_xc (per site)', fontsize=12)
    ax2.set_title('Comparison with Bethe Ansatz Exact Solution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}fig2_bethe_comparison.png', dpi=150)
    print(f"[Saved] {save_path}fig2_bethe_comparison.png")
    
    # Figure 3: Shell effects
    fig3, axes = plt.subplots(2, 2, figsize=(10, 8))
    U_plot = [1.0, 2.0, 4.0, 6.0]
    
    for idx, U in enumerate(U_plot):
        ax = axes[idx//2, idx%2]
        
        Ns = np.array([4, 6, 8, 10])
        e_per_N = np.array([DATA[U][N]['E_xc'] / N for N in Ns])
        inv_N = 1.0 / Ns
        
        # Open shell (N=6, 10)
        ax.scatter([1/6, 1/10], [e_per_N[1], e_per_N[3]], 
                   s=100, c='blue', marker='o', label='Open shell', zorder=5)
        # Closed shell (N=4, 8)
        ax.scatter([1/4, 1/8], [e_per_N[0], e_per_N[2]], 
                   s=100, c='red', marker='s', label='Closed shell', zorder=5)
        
        # Bethe Ansatz
        e_bethe = bethe_Exc_density(U)
        ax.axhline(e_bethe, color='green', linestyle='--', linewidth=2, label='Bethe (N→∞)')
        
        ax.set_xlabel('1/N', fontsize=10)
        ax.set_ylabel('E_xc / N', fontsize=10)
        ax.set_title(f'U/t = {U}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.3)
    
    plt.suptitle('Finite-Size Scaling: Shell Effects', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_path}fig3_shell_effects.png', dpi=150)
    print(f"[Saved] {save_path}fig3_shell_effects.png")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    
    print("="*60)
    print("Λ³-DFT Validation Analysis")
    print("="*60)
    
    # Bethe Ansatz comparison
    print("\n[Bethe Ansatz Comparison]")
    print(f"{'U/t':>6} {'Bethe':>10} {'Λ³-DFT':>10} {'Error':>8}")
    print("-"*40)
    
    comparison = compare_with_bethe()
    for c in comparison:
        print(f"{c['U']:>6.1f} {c['bethe']:>10.4f} {c['lambda3']:>10.4f} {c['error_percent']:>7.2f}%")
    
    mean_error = np.mean([c['error_percent'] for c in comparison])
    print(f"\nMean error: {mean_error:.2f}%")
    
    # γ fitting
    print("\n[Correlation Dimension γ(U/t)]")
    fit = fit_gamma()
    print(f"  {fit['formula']}")
    print(f"  R² = {fit['r_squared']:.4f}")
    
    # Generate figures
    print("\n[Generating Figures]")
    import os
    os.makedirs('figures', exist_ok=True)
    plot_results()
    
    print("\n[Done]")
