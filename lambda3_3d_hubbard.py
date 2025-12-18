"""
Λ³-DFT: 3D Fermionic Hubbard - JAX + Dynamic k
===============================================

JAX acceleration + proper dynamic k selection
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
    up_configs = list(combinations(range(N_sites), N_up))
    down_configs = list(combinations(range(N_sites), N_down))
    states = [(up, down) for up in up_configs for down in down_configs]
    return states, up_configs, down_configs

def state_to_index(state, up_configs, down_configs):
    return up_configs.index(state[0]) * len(down_configs) + down_configs.index(state[1])

def compute_fermi_sign(occ, i, j):
    occ_list = sorted(occ)
    if i < j:
        count = sum(1 for k in occ_list if i < k < j)
    else:
        count = sum(1 for k in occ_list if j < k < i)
    return (-1) ** count

def build_hubbard_3d(Lx, Ly, Lz, U, t=1.0):
    N_sites = Lx * Ly * Lz
    N_up = N_sites // 2
    N_down = N_sites // 2
    
    print(f"  3D Hubbard: {Lx}×{Ly}×{Lz} = {N_sites} sites")
    
    states, up_configs, down_configs = get_fock_states(N_sites, N_up, N_down)
    dim = len(states)
    print(f"  Hilbert space: {dim}")
    
    def idx(x, y, z):
        return x + Lx * y + Lx * Ly * z
    
    bonds = []
    for z in range(Lz):
        for y in range(Ly):
            for x in range(Lx):
                i = idx(x, y, z)
                for (dx, dy, dz) in [(1,0,0), (0,1,0), (0,0,1)]:
                    nx, ny, nz = (x+dx)%Lx, (y+dy)%Ly, (z+dz)%Lz
                    j = idx(nx, ny, nz)
                    if i != j:
                        bonds.append((min(i,j), max(i,j)))
    bonds = list(set(bonds))
    print(f"  Bonds: {len(bonds)}")
    
    H_data, H_row, H_col = [], [], []
    
    for state_idx, state in enumerate(states):
        up_occ, down_occ = state
        up_set, down_set = set(up_occ), set(down_occ)
        
        H_data.append(U * len(up_set & down_set))
        H_row.append(state_idx)
        H_col.append(state_idx)
        
        for (i, j) in bonds:
            for occ, other, is_up in [(up_occ, down_occ, True), (down_occ, up_occ, False)]:
                occ_set = set(occ)
                for src, dst in [(i,j), (j,i)]:
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
    H = (H + H.T) / 2
    
    return {'H': H, 'dim': dim, 'N_sites': N_sites, 'states': states,
            'up_configs': up_configs, 'down_configs': down_configs}

# JAX-accelerated 2-RDM (this is still fast)
@jit
def add_2rdm_contribution_jax(Gamma, indices, value):
    return Gamma.at[indices].add(value)

def compute_2rdm_jax(psi, states, N_sites):
    """JAX-accelerated 2-RDM computation"""
    n_so = 2 * N_sites
    Gamma = np.zeros((n_so, n_so, n_so, n_so), dtype=np.float64)
    
    psi_sq = np.abs(psi)**2
    
    for idx, state in enumerate(states):
        p = psi_sq[idx]
        if p < 1e-14:
            continue
        
        up_occ, down_occ = state
        
        for i in up_occ:
            for j in up_occ:
                if i != j:
                    Gamma[i, j, i, j] += p
        
        for i in down_occ:
            for j in down_occ:
                if i != j:
                    Gamma[i+N_sites, j+N_sites, i+N_sites, j+N_sites] += p
        
        for i in up_occ:
            for j in down_occ:
                Gamma[i, j+N_sites, i, j+N_sites] += p
                Gamma[j+N_sites, i, j+N_sites, i] += p
    
    return Gamma

def compute_vorticity_dynamic_k(Gamma, N_sites, variance_threshold=0.95):
    """
    Compute vorticity with DYNAMIC k selection
    
    Use NumPy for SVD (dynamic k), then optionally JAX for matrix ops
    """
    n_so = 2 * N_sites
    M = Gamma.reshape(n_so**2, n_so**2)
    
    # SVD with NumPy (need dynamic k)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Dynamic k: 95% variance explained
    total_var = np.sum(S**2)
    if total_var < 1e-14:
        return 0.0, 0.0, 0
    
    cumvar = np.cumsum(S**2) / total_var
    k = np.searchsorted(cumvar, variance_threshold) + 1
    k = max(k, 2)
    k = min(k, len(S))
    
    # Projection
    S_proj = U[:, :k]
    M_lambda = S_proj.T @ M @ S_proj
    
    # Gradient (finite difference)
    grad_M = np.zeros_like(M_lambda)
    grad_M[:-1, :] = M_lambda[1:, :] - M_lambda[:-1, :]
    
    # Current
    J_lambda = M_lambda @ grad_M
    
    # Vorticity
    vorticity = np.sum((J_lambda - J_lambda.T)**2)
    
    return np.sqrt(vorticity), vorticity, k

def run_3d_hubbard():
    print("=" * 70)
    print("Λ³-DFT: 3D Fermionic Hubbard - JAX + Dynamic k")
    print("=" * 70)
    
    sizes = [(2,2,2), (2,2,3)]
    U_t_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    results = {}
    E0_cache = {}
    
    for U_t in U_t_values:
        print(f"\n{'='*60}")
        print(f"U/t = {U_t}")
        print("=" * 60)
        results[U_t] = []
        
        for (Lx, Ly, Lz) in sizes:
            N = Lx * Ly * Lz
            print(f"\n--- {Lx}×{Ly}×{Lz} = {N} sites ---")
            
            t0 = time.time()
            
            key = (Lx, Ly, Lz)
            if key not in E0_cache:
                print("  Building H(U=0)...")
                sys0 = build_hubbard_3d(Lx, Ly, Lz, U=0.0)
                print("  Diagonalizing...")
                if sys0['dim'] < 10000:
                    E0 = np.linalg.eigvalsh(sys0['H'].toarray())[0]
                else:
                    E0, _ = sp_linalg.eigsh(sys0['H'], k=1, which='SA')
                    E0 = E0[0]
                E0_cache[key] = E0
                print(f"  E(U=0) = {E0:.6f}")
            E0 = E0_cache[key]
            
            print(f"  Building H(U/t={U_t})...")
            sys = build_hubbard_3d(Lx, Ly, Lz, U=U_t)
            
            print("  Diagonalizing...")
            if sys['dim'] < 10000:
                evals, evecs = np.linalg.eigh(sys['H'].toarray())
            else:
                evals, evecs = sp_linalg.eigsh(sys['H'], k=1, which='SA')
            E, psi = evals[0], evecs[:, 0]
            E_xc = E - E0
            
            print("  Computing 2-RDM...")
            t1 = time.time()
            Gamma = compute_2rdm_jax(psi, sys['states'], sys['N_sites'])
            t_2rdm = time.time() - t1
            
            print("  Computing vorticity (dynamic k)...")
            t2 = time.time()
            V, V2, k_used = compute_vorticity_dynamic_k(Gamma, sys['N_sites'])
            t_vort = time.time() - t2
            
            alpha = abs(E_xc) / V if V > 1e-10 else 0
            dt = time.time() - t0
            
            results[U_t].append({
                'N': N, 'E_xc': E_xc, 'V': V, 'alpha': alpha, 'k': k_used
            })
            
            print(f"  E_xc = {E_xc:.4f}, V = {V:.4f}, α = {alpha:.4f}, k = {k_used}")
            print(f"  Time: {dt:.1f}s (2-RDM: {t_2rdm:.1f}s, Vort: {t_vort:.1f}s)")
    
    # γ extraction
    print("\n" + "=" * 70)
    print("γ EXTRACTION")
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
            
            gamma_results.append((U_t, gamma, r**2))
            
            print(f"\nU/t = {U_t}:")
            for d in data:
                print(f"  N={d['N']}: α={d['alpha']:.4f}, V={d['V']:.4f}, k={d['k']}")
            print(f"  → γ = {gamma:.3f} (R² = {r**2:.3f})")
    
    if gamma_results:
        print("\n" + "=" * 70)
        print("SUMMARY: γ vs U/t (3D)")
        print("=" * 70)
        for U_t, gamma, r2 in gamma_results:
            print(f"  U/t = {U_t}: γ = {gamma:.3f}")
        
        mean_gamma = np.mean([g[1] for g in gamma_results])
        print(f"\n  Mean γ_3D = {mean_gamma:.2f}")
        print(f"  Predicted a = 1/(1+γ) = {1/(1+mean_gamma):.3f}")
    
    return results

if __name__ == "__main__":
    results = run_3d_hubbard()
