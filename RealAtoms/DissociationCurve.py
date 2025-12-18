"""
Λ³-DFT: H₂ Dissociation Curve
=============================

FINAL MISSION: Track γ(R) as H₂ is stretched!

R ≈ 0.74 Å (equilibrium): Weak correlation → γ ≈ 2
R → ∞ (dissociation): Strong correlation → γ → 0

This is the Hubbard U/t → ∞ limit in real chemistry!
"""

import numpy as np
from scipy import stats
import time

try:
    from pyscf import gto, scf, fci
    PYSCF_OK = True
except ImportError:
    PYSCF_OK = False
    print("PySCF not available!")

def compute_vorticity(dm2, norb):
    """Compute vorticity from the 2-RDM"""
    n_so = 2 * norb
    
    # Expand to spin-orbital basis
    Gamma = np.zeros((n_so, n_so, n_so, n_so))
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    Gamma[p, q, r, s] = dm2[p, q, r, s]
    
    M = Gamma.reshape(n_so**2, n_so**2)
    
    # Singular value decomposition
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    total_var = np.sum(S**2)
    if total_var < 1e-14:
        return 0.0, 0, S
    
    cumvar = np.cumsum(S**2) / total_var
    k = np.searchsorted(cumvar, 0.95) + 1
    k = max(k, 2)
    k = min(k, len(S))
    
    # Projection to dominant subspace
    S_proj = U[:, :k]
    M_lambda = S_proj.T @ M @ S_proj
    
    # Gradient and current
    grad_M = np.zeros_like(M_lambda)
    grad_M[:-1, :] = M_lambda[1:, :] - M_lambda[:-1, :]
    J_lambda = M_lambda @ grad_M
    
    # Vorticity measure
    vorticity = np.sum((J_lambda - J_lambda.T)**2)
    
    return np.sqrt(vorticity), k, S

def run_h2_dissociation():
    print("=" * 70)
    print("Λ³-DFT: H₂ DISSOCIATION – THE FINAL FRONTIER")
    print("=" * 70)
    print("\nTracking γ(R) from equilibrium to dissociation!")
    print("This is the Hubbard U/t → ∞ limit in real chemistry!\n")
    
    if not PYSCF_OK:
        print("PySCF not available!")
        return None
    
    # Bond distances in angstroms
    # From equilibrium (0.74 Å) to highly stretched (5.0 Å)
    R_values = [0.5, 0.74, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    basis = 'cc-pvdz'  # Reasonable basis for correlation effects
    
    results = []
    
    for R in R_values:
        print(f"\n{'='*60}")
        print(f"R = {R:.2f} Å")
        print("=" * 60)
        
        t0 = time.time()
        
        # Build H₂ molecule
        mol = gto.M(
            atom=f"H 0 0 0; H 0 0 {R}",
            basis=basis,
            symmetry=False,
            verbose=0,
            unit='angstrom'
        )
        
        n_orb = mol.nao
        print(f"  Number of orbitals: {n_orb}")
        
        # RHF (expected to fail at dissociation — intentionally!)
        mf = scf.RHF(mol)
        mf.kernel()
        E_HF = mf.e_tot
        print(f"  E(RHF) = {E_HF:.6f} Ha")
        
        # FCI (exact for two electrons)
        cisolver = fci.FCI(mf)
        E_FCI, ci_vec = cisolver.kernel()
        
        E_corr = E_FCI - E_HF
        print(f"  E(FCI) = {E_FCI:.6f} Ha")
        print(f"  Correlation energy = {E_corr:.6f} Ha = {E_corr * 27.2114:.4f} eV")
        
        # Two-particle reduced density matrix
        from pyscf import fci as fci_mod
        nelec = (1, 1)
        dm2 = fci_mod.direct_spin1.make_rdm12(ci_vec, n_orb, nelec)[1]
        
        # Vorticity analysis
        V, k, singular_values = compute_vorticity(dm2, n_orb)
        
        alpha = abs(E_corr) / V if V > 1e-10 else float('inf')
        
        # Effective U/t proxy (correlation strength indicator)
        # Small at equilibrium, large at dissociation
        U_t_eff = abs(E_corr) / (abs(E_HF) / 2) if E_HF != 0 else 0
        
        results.append({
            'R': R,
            'E_HF': E_HF,
            'E_FCI': E_FCI,
            'E_corr': E_corr,
            'V': V,
            'alpha': alpha,
            'k': k,
            'U_t_eff': U_t_eff,
            'S': singular_values
        })
        
        print(f"  V = {V:.4f}, α = {alpha:.4f}, k = {k}")
        print(f"  Effective U/t proxy: {U_t_eff:.3f}")
        print(f"  Elapsed time: {time.time()-t0:.1f} s")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: H₂ DISSOCIATION CURVE")
    print("=" * 70)
    
    print(f"\n  {'R (Å)':<8} {'E_corr (eV)':<12} {'V':<10} {'α':<10} {'k':<5} {'U/t_eff':<8}")
    print("  " + "-" * 55)
    
    for r in results:
        print(f"  {r['R']:<8.2f} {r['E_corr']*27.2114:<12.4f} {r['V']:<10.4f} "
              f"{r['alpha']:<10.4f} {r['k']:<5} {r['U_t_eff']:<8.3f}")
    
    # γ analysis: weak vs strong correlation
    print("\n" + "=" * 70)
    print("γ ANALYSIS: WEAK → STRONG CORRELATION")
    print("=" * 70)
    
    equilibrium = [r for r in results if r['R'] == 0.74][0]
    stretched = [r for r in results if r['R'] >= 3.0]
    
    print(f"\n  [Equilibrium R = 0.74 Å] (Weak correlation)")
    print(f"    E_corr = {equilibrium['E_corr']*27.2114:.4f} eV")
    print(f"    V = {equilibrium['V']:.4f}")
    print(f"    α = {equilibrium['alpha']:.4f}")
    
    if stretched:
        print(f"\n  [Dissociation R ≥ 3.0 Å] (Strong correlation)")
        for s in stretched:
            print(f"    R={s['R']:.1f}: E_corr={s['E_corr']*27.2114:.4f} eV, "
                  f"V={s['V']:.4f}, α={s['alpha']:.4f}")
    
    # Infer γ behavior from α scaling
    print("\n" + "-" * 50)
    print("γ INFERENCE FROM α SCALING")
    print("-" * 50)
    
    print("\n  [Observation]")
    print(f"    Equilibrium (R=0.74): α = {equilibrium['alpha']:.4f}")
    if results[-1]['alpha'] != float('inf'):
        print(f"    Dissociation (R=5.0): α = {results[-1]['alpha']:.4f}")
        ratio = results[-1]['alpha'] / equilibrium['alpha']
        print(f"    Ratio: {ratio:.2f}×")
        
        if ratio > 1:
            print("\n    → Increasing α = decreasing correlation efficiency = strong-correlation limit!")
            print("    → γ is approaching 0!")
        else:
            print("\n    → Decreasing α = anomalous behavior, requires investigation")
    
    print("\n" + "=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)
        
    return results

if __name__ == "__main__":
    results = run_h2_dissociation()
