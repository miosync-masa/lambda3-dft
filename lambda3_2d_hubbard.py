"""
Λ³-DFT: 2D Square Lattice Hubbard - JAX + Dynamic k
====================================================

2D version for the "missing dimension" in the paper!
Based on 3D code, simplified to 2D square lattice.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sp_linalg
from scipy import stats
from itertools import combinations
import time

# JAX imports
import jax
import jax.numpy as jnp
from jax import jit

print(f"JAX devices: {jax.devices()}")

def get_fock_states(N_sites, N_up, N_down):
    """Generate all Fock states for given particle numbers"""
    up_configs = list(combinations(range(N_sites), N_up))
    down_configs = list(combinations(range(N_sites), N_down))
    states = [(up, down) for up in up_configs for down in down_configs]
    return states, up_configs, down_configs

def state_to_index(state, up_configs, down_configs):
    """Convert state to index in Hilbert space"""
    return up_configs.index(state[0]) * len(down_configs) + down_configs.index(state[1])

def compute_fermi_sign(occ, i, j):
    """Compute fermionic sign for hopping from i to j"""
    occ_list = sorted(occ)
    if i < j:
        count = sum(1 for k in occ_list if i < k < j)
    else:
        count = sum(1 for k in occ_list if j < k < i)
    return (-1) ** count

def build_hubbard_2d(Lx, Ly, U, t=1.0):
    """
    Build 2D square lattice Hubbard Hamiltonian
    
    Parameters:
    -----------
    Lx, Ly : int
        Lattice dimensions
    U : float
        On-site interaction strength
    t : float
        Hopping parameter (default 1.0)
    
    Returns:
    --------
    dict with H (sparse), dim, N_sites, states, configs
    """
    N_sites = Lx * Ly
    N_up = N_sites // 2
    N_down = N_sites // 2
    
    print(f"  2D Square Hubbard: {Lx}×{Ly} = {N_sites} sites")
    
    states, up_configs, down_configs = get_fock_states(N_sites, N_up, N_down)
    dim = len(states)
    print(f"  Hilbert space dimension: {dim}")
    
    # Site indexing: i = x + Lx * y
    def idx(x, y):
        return x + Lx * y
    
    # Generate bonds (nearest-neighbor with PBC)
    bonds = []
    for y in range(Ly):
        for x in range(Lx):
            i = idx(x, y)
            # 2D: only (1,0) and (0,1) directions
            for (dx, dy) in [(1, 0), (0, 1)]:
                nx, ny = (x + dx) % Lx, (y + dy) % Ly
                j = idx(nx, ny)
                if i != j:  # Avoid self-loops for 1×N or N×1
                    bonds.append((min(i, j), max(i, j)))
    bonds = list(set(bonds))
    print(f"  Number of bonds: {len(bonds)}")
    
    # Build sparse Hamiltonian
    H_data, H_row, H_col = [], [], []
    
    for state_idx, state in enumerate(states):
        up_occ, down_occ = state
        up_set, down_set = set(up_occ), set(down_occ)
        
        # Diagonal: U * (number of doubly occupied sites)
        n_double = len(up_set & down_set)
        H_data.append(U * n_double)
        H_row.append(state_idx)
        H_col.append(state_idx)
        
        # Off-diagonal: hopping
        for (i, j) in bonds:
            for occ, other, is_up in [(up_occ, down_occ, True), (down_occ, up_occ, False)]:
                occ_set = set(occ)
                for src, dst in [(i, j), (j, i)]:
                    if src in occ_set and dst not in occ_set:
                        new_occ = tuple(sorted((occ_set - {src}) | {dst}))
                        new_state = (new_occ, other) if is_up else (other, new_occ)
                        try:
                            new_idx = state_to_index(new_state, up_configs, down_configs)
                            sign = compute_fermi_sign(occ, src, dst)
                            H_data.append(-t * sign)
                            H_row.append(new_idx)
                            H_col.append(state_idx)
                        except:
                            pass
    
    H = sparse.csr_matrix((H_data, (H_row, H_col)), shape=(dim, dim))
    H = (H + H.T) / 2  # Ensure Hermiticity
    
    return {
        'H': H, 
        'dim': dim, 
        'N_sites': N_sites,
        'Lx': Lx,
        'Ly': Ly,
        'states': states,
        'up_configs': up_configs, 
        'down_configs': down_configs
    }

def compute_2rdm(psi, states, N_sites):
    """Compute 2-particle reduced density matrix"""
    n_so = 2 * N_sites  # spin-orbitals
    Gamma = np.zeros((n_so, n_so, n_so, n_so), dtype=np.float64)
    
    psi_sq = np.abs(psi)**2
    
    for idx, state in enumerate(states):
        p = psi_sq[idx]
        if p < 1e-14:
            continue
        
        up_occ, down_occ = state
        
        # Up-up correlations
        for i in up_occ:
            for j in up_occ:
                if i != j:
                    Gamma[i, j, i, j] += p
        
        # Down-down correlations
        for i in down_occ:
            for j in down_occ:
                if i != j:
                    Gamma[i + N_sites, j + N_sites, i + N_sites, j + N_sites] += p
        
        # Up-down correlations
        for i in up_occ:
            for j in down_occ:
                Gamma[i, j + N_sites, i, j + N_sites] += p
                Gamma[j + N_sites, i, j + N_sites, i] += p
    
    return Gamma

def compute_vorticity_dynamic_k(Gamma, N_sites, variance_threshold=0.95):
    """
    Compute vorticity with dynamic k selection (95% variance)
    """
    n_so = 2 * N_sites
    M = Gamma.reshape(n_so**2, n_so**2)
    
    # SVD
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Dynamic k: capture 95% variance
    total_var = np.sum(S**2)
    if total_var < 1e-14:
        return 0.0, 0.0, 0
    
    cumvar = np.cumsum(S**2) / total_var
    k = np.searchsorted(cumvar, variance_threshold) + 1
    k = max(k, 2)
    k = min(k, len(S))
    
    # Projection to Λ-space
    S_proj = U[:, :k]
    M_lambda = S_proj.T @ M @ S_proj
    
    # Gradient (finite difference)
    grad_M = np.zeros_like(M_lambda)
    grad_M[:-1, :] = M_lambda[1:, :] - M_lambda[:-1, :]
    
    # Λ-current
    J_lambda = M_lambda @ grad_M
    
    # Vorticity = antisymmetric part squared
    vorticity = np.sum((J_lambda - J_lambda.T)**2)
    
    return np.sqrt(vorticity), vorticity, k

def run_2d_hubbard():
    """Main calculation for 2D square lattice"""
    print("=" * 70)
    print("Λ³-DFT: 2D Square Lattice Hubbard Model")
    print("=" * 70)
    
    # System sizes: need at least 2 for γ extraction
    # 2×4=8, 3×4=12, 4×4=16 (if feasible)
    sizes = [
        (2, 4),   # 8 sites, dim ~ 4,900
        (3, 4),   # 12 sites, dim ~ 853,776
        # (4, 4), # 16 sites, dim ~ 165,636,900 (too large!)
    ]
    
    U_t_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    results = {}
    E0_cache = {}
    
    for U_t in U_t_values:
        print(f"\n{'='*60}")
        print(f"U/t = {U_t}")
        print("=" * 60)
        results[U_t] = []
        
        for (Lx, Ly) in sizes:
            N = Lx * Ly
            print(f"\n--- {Lx}×{Ly} = {N} sites ---")
            
            t0 = time.time()
            
            # Non-interacting reference
            key = (Lx, Ly)
            if key not in E0_cache:
                print("  Building H(U=0)...")
                sys0 = build_hubbard_2d(Lx, Ly, U=0.0)
                print("  Diagonalizing U=0...")
                if sys0['dim'] < 50000:
                    E0 = np.linalg.eigvalsh(sys0['H'].toarray())[0]
                else:
                    E0, _ = sp_linalg.eigsh(sys0['H'], k=1, which='SA')
                    E0 = E0[0]
                E0_cache[key] = E0
                print(f"  E(U=0) = {E0:.6f}")
            E0 = E0_cache[key]
            
            # Interacting system
            print(f"  Building H(U/t={U_t})...")
            sys = build_hubbard_2d(Lx, Ly, U=U_t)
            
            print("  Diagonalizing...")
            if sys['dim'] < 50000:
                evals, evecs = np.linalg.eigh(sys['H'].toarray())
            else:
                evals, evecs = sp_linalg.eigsh(sys['H'], k=1, which='SA')
            E, psi = evals[0], evecs[:, 0]
            E_xc = E - E0
            
            print("  Computing 2-RDM...")
            t1 = time.time()
            Gamma = compute_2rdm(psi, sys['states'], sys['N_sites'])
            t_2rdm = time.time() - t1
            
            # Verify 2-RDM trace
            N_e = N  # half-filling
            trace = np.einsum('pqpq->', Gamma)
            expected_trace = N_e * (N_e - 1)
            print(f"  2-RDM trace: {trace:.4f} (expected {expected_trace})")
            
            print("  Computing vorticity (dynamic k)...")
            t2 = time.time()
            V, V2, k_used = compute_vorticity_dynamic_k(Gamma, sys['N_sites'])
            t_vort = time.time() - t2
            
            alpha = abs(E_xc) / V if V > 1e-10 else 0
            dt = time.time() - t0
            
            results[U_t].append({
                'Lx': Lx, 'Ly': Ly, 'N': N,
                'E': E, 'E0': E0, 'E_xc': E_xc,
                'V': V, 'alpha': alpha, 'k': k_used
            })
            
            print(f"  E = {E:.6f}, E_xc = {E_xc:.6f}")
            print(f"  V = {V:.4f}, α = {alpha:.4f}, k = {k_used}")
            print(f"  Time: {dt:.1f}s (2-RDM: {t_2rdm:.1f}s, Vort: {t_vort:.1f}s)")
    
    # =========================================================================
    # γ extraction
    # =========================================================================
    print("\n" + "=" * 70)
    print("γ EXTRACTION (2D Square Lattice)")
    print("=" * 70)
    
    gamma_results = []
    for U_t, data in results.items():
        if len(data) >= 2 and all(d['alpha'] > 0 and d['V'] > 0 for d in data):
            N_arr = np.array([d['N'] for d in data])
            a_arr = np.array([d['alpha'] for d in data])
            
            log_N = np.log(N_arr)
            log_a = np.log(a_arr)
            slope, intercept, r, _, _ = stats.linregress(log_N, log_a)
            gamma = -slope
            
            gamma_results.append({
                'U_t': U_t, 'gamma': gamma, 'R2': r**2,
                'data': data
            })
            
            print(f"\nU/t = {U_t}:")
            for d in data:
                print(f"  {d['Lx']}×{d['Ly']}={d['N']}: α={d['alpha']:.4f}, V={d['V']:.4f}, k={d['k']}")
            print(f"  → γ = {gamma:.3f} (R² = {r**2:.3f})")
    
    # =========================================================================
    # Summary table
    # =========================================================================
    if gamma_results:
        print("\n" + "=" * 70)
        print("SUMMARY: γ(U/t) for 2D Square Lattice")
        print("=" * 70)
        print(f"{'U/t':>6} {'γ':>8} {'R²':>8} {'a=1/(1+γ)':>12}")
        print("-" * 40)
        for res in gamma_results:
            a_pred = 1 / (1 + res['gamma']) if res['gamma'] > -1 else float('inf')
            print(f"{res['U_t']:>6.1f} {res['gamma']:>8.3f} {res['R2']:>8.3f} {a_pred:>12.3f}")
        
        # Linear fit γ(U/t)
        U_arr = np.array([r['U_t'] for r in gamma_results])
        g_arr = np.array([r['gamma'] for r in gamma_results])
        slope, intercept, r, _, _ = stats.linregress(U_arr, g_arr)
        
        print(f"\nLinear fit: γ(U/t) = {intercept:.2f} + {slope:.2f}×(U/t)")
        print(f"R² = {r**2:.3f}")
        
        mean_gamma = np.mean(g_arr)
        print(f"\nMean γ_2D = {mean_gamma:.2f}")
        print(f"Predicted optimal a = 1/(1+γ) = {1/(1+mean_gamma):.3f}")
        
        # Comparison with other dimensions
        print("\n" + "=" * 70)
        print("DIMENSIONAL COMPARISON")
        print("=" * 70)
        print(f"{'Dimension':>12} {'γ (weak U)':>12} {'γ (strong U)':>12}")
        print("-" * 40)
        print(f"{'1D chain':>12} {'~2.1':>12} {'~0.8':>12}")
        print(f"{'2D square':>12} {f'~{gamma_results[0][\"gamma\"]:.1f}':>12} {f'~{gamma_results[-1][\"gamma\"]:.1f}':>12}")
        print(f"{'3D cubic':>12} {'~0':>12} {'~0':>12}")
    
    return results, gamma_results

if __name__ == "__main__":
    results, gamma_results = run_2d_hubbard()
