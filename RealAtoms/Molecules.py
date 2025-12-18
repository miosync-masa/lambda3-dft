"""
Λ³-DFT: Real Atoms/Molecules with PySCF
=======================================

Compute γ from ACTUAL quantum chemistry calculations
"""

import numpy as np
from scipy import stats
import time

def compute_vorticity_from_2rdm(rdm2, n_orb):
    """Compute vorticity from 2-RDM (same as Hubbard version)"""
    # rdm2 shape: (n_orb, n_orb, n_orb, n_orb)
    # Reshape to matrix
    M = rdm2.reshape(n_orb**2, n_orb**2)
    
    # SVD with dynamic k (95% variance)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    total_var = np.sum(S**2)
    if total_var < 1e-14:
        return 0.0, 0, 0
    
    cumvar = np.cumsum(S**2) / total_var
    k = np.searchsorted(cumvar, 0.95) + 1
    k = max(k, 2)
    k = min(k, len(S))
    
    # Projection
    S_proj = U[:, :k]
    M_lambda = S_proj.T @ M @ S_proj
    
    # Gradient
    grad_M = np.zeros_like(M_lambda)
    grad_M[:-1, :] = M_lambda[1:, :] - M_lambda[:-1, :]
    
    # Current and vorticity
    J_lambda = M_lambda @ grad_M
    vorticity = np.sum((J_lambda - J_lambda.T)**2)
    
    return np.sqrt(vorticity), vorticity, k

def run_atom_fci(atom_symbol, basis='cc-pvdz'):
    """Run FCI calculation for an atom and extract 2-RDM"""
    from pyscf import gto, scf, fci
    
    print(f"\n{'='*60}")
    print(f"Atom: {atom_symbol}, Basis: {basis}")
    print('='*60)
    
    # Build molecule (single atom)
    mol = gto.Mole()
    mol.atom = f'{atom_symbol} 0 0 0'
    mol.basis = basis
    mol.spin = 0  # Closed shell
    mol.build()
    
    n_elec = mol.nelectron
    n_orb = mol.nao
    print(f"  Electrons: {n_elec}, Orbitals: {n_orb}")
    
    # HF calculation
    mf = scf.RHF(mol)
    mf.verbose = 0
    E_hf = mf.kernel()
    print(f"  E(HF) = {E_hf:.6f} Ha")
    
    # FCI calculation
    t0 = time.time()
    cisolver = fci.FCI(mf)
    cisolver.verbose = 0
    E_fci, fcivec = cisolver.kernel()
    t_fci = time.time() - t0
    print(f"  E(FCI) = {E_fci:.6f} Ha (time: {t_fci:.1f}s)")
    
    # Correlation energy
    E_corr = E_fci - E_hf
    print(f"  E_corr = {E_corr:.6f} Ha = {E_corr*27.211:.3f} eV")
    
    # Extract 2-RDM
    print("  Computing 2-RDM...")
    rdm1, rdm2 = cisolver.make_rdm12(fcivec, n_orb, (n_elec//2, n_elec//2))
    
    # Compute vorticity
    print("  Computing vorticity...")
    V, V2, k = compute_vorticity_from_2rdm(rdm2, n_orb)
    
    alpha = abs(E_corr) / V if V > 1e-10 else 0
    
    print(f"  V = {V:.6f} (k={k})")
    print(f"  α = E_corr/V = {alpha:.4f}")
    
    return {
        'atom': atom_symbol,
        'basis': basis,
        'n_elec': n_elec,
        'n_orb': n_orb,
        'E_hf': E_hf,
        'E_fci': E_fci,
        'E_corr': E_corr,
        'V': V,
        'alpha': alpha,
        'k': k
    }

def run_molecule_fci(name, geometry, basis='cc-pvdz'):
    """Run FCI calculation for a molecule"""
    from pyscf import gto, scf, fci
    
    print(f"\n{'='*60}")
    print(f"Molecule: {name}, Basis: {basis}")
    print('='*60)
    
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.spin = 0
    mol.build()
    
    n_elec = mol.nelectron
    n_orb = mol.nao
    print(f"  Electrons: {n_elec}, Orbitals: {n_orb}")
    
    if n_orb > 20:
        print(f"  WARNING: Large system, FCI may be slow/impossible")
    
    # HF
    mf = scf.RHF(mol)
    mf.verbose = 0
    E_hf = mf.kernel()
    print(f"  E(HF) = {E_hf:.6f} Ha")
    
    # FCI
    t0 = time.time()
    try:
        cisolver = fci.FCI(mf)
        cisolver.verbose = 0
        cisolver.max_cycle = 100
        E_fci, fcivec = cisolver.kernel()
        t_fci = time.time() - t0
        print(f"  E(FCI) = {E_fci:.6f} Ha (time: {t_fci:.1f}s)")
        
        E_corr = E_fci - E_hf
        print(f"  E_corr = {E_corr:.6f} Ha")
        
        # 2-RDM
        print("  Computing 2-RDM...")
        rdm1, rdm2 = cisolver.make_rdm12(fcivec, n_orb, (n_elec//2, n_elec//2))
        
        print("  Computing vorticity...")
        V, V2, k = compute_vorticity_from_2rdm(rdm2, n_orb)
        
        alpha = abs(E_corr) / V if V > 1e-10 else 0
        
        print(f"  V = {V:.6f} (k={k})")
        print(f"  α = {alpha:.4f}")
        
        return {
            'name': name,
            'n_elec': n_elec,
            'n_orb': n_orb,
            'E_corr': E_corr,
            'V': V,
            'alpha': alpha
        }
    except Exception as e:
        print(f"  FCI failed: {e}")
        return None

def main():
    print("=" * 70)
    print("Λ³-DFT: Real Quantum Chemistry Calculations")
    print("=" * 70)
    
    results = []
    
    # 1. Atoms (closed shell)
    print("\n" + "="*70)
    print("PART 1: ATOMS")
    print("="*70)
    
    atoms = ['He', 'Be', 'Ne']  # Closed shell atoms
    
    for atom in atoms:
        try:
            r = run_atom_fci(atom, basis='cc-pvdz')
            results.append(r)
        except Exception as e:
            print(f"  Failed: {e}")
    
    # 2. Small molecules
    print("\n" + "="*70)
    print("PART 2: SMALL MOLECULES")
    print("="*70)
    
    molecules = [
        ('H2', 'H 0 0 0; H 0 0 0.74'),
        ('LiH', 'Li 0 0 0; H 0 0 1.6'),
        ('HF', 'H 0 0 0; F 0 0 0.92'),
    ]
    
    for name, geom in molecules:
        try:
            r = run_molecule_fci(name, geom, basis='cc-pvdz')
            if r:
                results.append(r)
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'System':<10} {'n_e':<5} {'n_orb':<6} {'E_corr(eV)':<12} {'V':<10} {'α':<10}")
    print("-"*60)
    
    for r in results:
        name = r.get('atom', r.get('name', '?'))
        n_e = r['n_elec']
        n_orb = r['n_orb']
        E_corr_eV = r['E_corr'] * 27.211
        V = r['V']
        alpha = r['alpha']
        print(f"{name:<10} {n_e:<5} {n_orb:<6} {E_corr_eV:<12.4f} {V:<10.4f} {alpha:<10.4f}")
    
    # Try to extract γ from scaling
    print("\n" + "="*70)
    print("γ ANALYSIS")
    print("="*70)
    
    # Group by similar systems and check scaling
    atom_results = [r for r in results if 'atom' in r]
    if len(atom_results) >= 2:
        n_arr = np.array([r['n_elec'] for r in atom_results])
        a_arr = np.array([r['alpha'] for r in atom_results])
        
        valid = a_arr > 0
        if np.sum(valid) >= 2:
            log_n = np.log(n_arr[valid])
            log_a = np.log(a_arr[valid])
            slope, intercept, r_val, _, _ = stats.linregress(log_n, log_a)
            gamma = -slope
            
            print(f"\nAtoms: α vs n_elec scaling")
            print(f"  γ = {gamma:.2f} (R² = {r_val**2:.3f})")
            print(f"  → a = 1/(1+γ) = {1/(1+gamma):.3f}")
    
    return results

if __name__ == "__main__":
    results = main()
