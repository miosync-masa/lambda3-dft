"""
===========================================================================
Λ³-DFT: Exchange-Correlation from Two-Particle Vorticity
===========================================================================

Computes exchange-correlation energy E_xc for the 1D Hubbard model
using the vorticity of the two-particle density matrix in Λ-space.

Theory:
    E_xc = α(U/t, N) · ∫(∇×J_Λ)² dA
    
    where:
        α(U/t, N) = A · √(U/t) · N^(-γ(U/t))
        γ(U/t) = 2.43 - 0.27·(U/t)   [correlation dimension]
        J_Λ = M_Λ · ∇M_Λ             [Λ-space current]
        M_Λ = S^T · M · S            [SVD-projected 2-RDM]

Reference:
    Iizumi, M. "Density Functional Theory Without Magic Numbers:
    Exchange-Correlation from Vorticity" Phys. Rev. B (2025)

Author: Masamichi Iizumi, Tamaki Iizumi
License: MIT
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import svd
import time
from numba import njit, prange


# ===========================================================================
# Numba-accelerated 2-RDM Calculation
# ===========================================================================

@njit(parallel=True)
def compute_2rdm_numba(psi, basis, n_spin_orb):
    """
    Compute 2-RDM using Numba parallelization.
    
    Definition:
        Γ_pqrs = ⟨Ψ|c†_p c†_q c_s c_r|Ψ⟩
    
    Parameters
    ----------
    psi : ndarray
        Ground state wavefunction
    basis : ndarray
        Fock space basis states (integer representation)
    n_spin_orb : int
        Number of spin-orbitals (2 × n_sites)
    
    Returns
    -------
    Gamma : ndarray
        2-RDM tensor of shape (n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb)
    """
    dim = len(basis)
    Gamma = np.zeros((n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb))
    
    # Build lookup table for basis index
    max_state = 0
    for i in range(dim):
        if basis[i] > max_state:
            max_state = basis[i]
    
    lookup = np.full(max_state + 1, -1, dtype=np.int64)
    for i in range(dim):
        lookup[basis[i]] = i
    
    # Compute Γ_pqrs = ⟨Ψ|c†_p c†_q c_s c_r|Ψ⟩
    for p in prange(n_spin_orb):
        for q in range(n_spin_orb):
            for r in range(n_spin_orb):
                for s in range(n_spin_orb):
                    result = 0.0
                    
                    for k in range(dim):
                        state = basis[k]
                        
                        # Apply c_r
                        if not ((state >> r) & 1):
                            continue
                        state1 = state ^ (1 << r)
                        sign1 = 0
                        for b in range(r):
                            if (state >> b) & 1:
                                sign1 += 1
                        
                        # Apply c_s
                        if not ((state1 >> s) & 1):
                            continue
                        state2 = state1 ^ (1 << s)
                        sign2 = sign1
                        for b in range(s):
                            if (state1 >> b) & 1:
                                sign2 += 1
                        
                        # Apply c†_q
                        if (state2 >> q) & 1:
                            continue
                        state3 = state2 ^ (1 << q)
                        sign3 = sign2
                        for b in range(q):
                            if (state2 >> b) & 1:
                                sign3 += 1
                        
                        # Apply c†_p
                        if (state3 >> p) & 1:
                            continue
                        state4 = state3 ^ (1 << p)
                        sign4 = sign3
                        for b in range(p):
                            if (state3 >> b) & 1:
                                sign4 += 1
                        
                        # Lookup final state
                        if state4 <= max_state:
                            idx4 = lookup[state4]
                            if idx4 >= 0:
                                sign = 1 if (sign4 % 2 == 0) else -1
                                result += sign * psi[idx4] * psi[k]
                    
                    Gamma[p, q, r, s] = result
    
    return Gamma


# ===========================================================================
# Hubbard Model Class
# ===========================================================================

class HubbardModel:
    """
    1D Hubbard model at half-filling with periodic boundary conditions.
    
    Hamiltonian:
        H = -t Σ_⟨ij⟩,σ (c†_iσ c_jσ + h.c.) + U Σ_i n_i↑ n_i↓
    
    Parameters
    ----------
    n_sites : int
        Number of lattice sites
    n_electrons : int
        Number of electrons (= n_sites for half-filling)
    t : float
        Hopping parameter (default: 1.0)
    U : float
        On-site Coulomb repulsion
    """
    
    def __init__(self, n_sites, n_electrons, t=1.0, U=4.0):
        self.n_sites = n_sites
        self.n_electrons = n_electrons
        self.t = t
        self.U = U
        self.ground_state = None
        self.E_total = None
        self.basis = None
        
    def _generate_basis(self):
        """Generate Fock space basis with fixed particle number."""
        n = self.n_sites
        n_elec = self.n_electrons
        basis = []
        for state in range(2**(2*n)):
            if bin(state).count('1') == n_elec:
                basis.append(state)
        return basis
    
    def _fermion_sign(self, state, bit_from, bit_to):
        """Compute fermionic sign from anticommutation."""
        if bit_from > bit_to:
            bit_from, bit_to = bit_to, bit_from
        count = 0
        for b in range(bit_from + 1, bit_to):
            if (state >> b) & 1:
                count += 1
        return (-1) ** count
    
    def _build_hamiltonian(self, basis):
        """Build sparse Hamiltonian matrix."""
        n = self.n_sites
        dim = len(basis)
        basis_idx = {state: i for i, state in enumerate(basis)}
        
        H = lil_matrix((dim, dim), dtype=np.float64)
        
        for i, state in enumerate(basis):
            # On-site interaction: U Σ_i n_i↑ n_i↓
            for site in range(n):
                n_up = (state >> (2*site)) & 1
                n_down = (state >> (2*site + 1)) & 1
                H[i, i] += self.U * n_up * n_down
            
            # Hopping: -t Σ_⟨ij⟩,σ (c†_iσ c_jσ + h.c.)
            for site in range(n):
                next_site = (site + 1) % n  # Periodic BC
                for spin in [0, 1]:
                    bit_i = 2*site + spin
                    bit_j = 2*next_site + spin
                    
                    # Hopping i → j
                    if (state >> bit_i) & 1 and not ((state >> bit_j) & 1):
                        new_state = state ^ (1 << bit_i) ^ (1 << bit_j)
                        if new_state in basis_idx:
                            sign = self._fermion_sign(state, bit_i, bit_j)
                            H[i, basis_idx[new_state]] -= self.t * sign
                    
                    # Hopping j → i
                    if (state >> bit_j) & 1 and not ((state >> bit_i) & 1):
                        new_state = state ^ (1 << bit_j) ^ (1 << bit_i)
                        if new_state in basis_idx:
                            sign = self._fermion_sign(state, bit_j, bit_i)
                            H[i, basis_idx[new_state]] -= self.t * sign
        
        return H.tocsr()
    
    def solve(self):
        """Find ground state using Lanczos method."""
        self.basis = self._generate_basis()
        dim = len(self.basis)
        
        H = self._build_hamiltonian(self.basis)
        
        if dim > 100:
            eigenvalues, eigenvectors = eigsh(H, k=1, which='SA')
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())
            eigenvalues = eigenvalues[:1]
            eigenvectors = eigenvectors[:, :1]
        
        self.E_total = eigenvalues[0]
        self.ground_state = eigenvectors[:, 0]
        self.H_sparse = H
        
        return self.E_total
    
    def compute_Exc(self):
        """
        Compute exchange-correlation energy.
        
        Definition:
            E_xc = E_total(U) - E_total(U=0)
        """
        if self.ground_state is None:
            self.solve()
        
        # Compute non-interacting energy
        U_orig = self.U
        self.U = 0.0
        H_nonint = self._build_hamiltonian(self.basis)
        
        if len(self.basis) > 100:
            E_nonint = eigsh(H_nonint, k=1, which='SA')[0][0]
        else:
            E_nonint = np.linalg.eigvalsh(H_nonint.toarray())[0]
        
        self.U = U_orig
        return self.E_total - E_nonint
    
    def compute_2rdm(self):
        """Compute 2-RDM using Numba acceleration."""
        if self.ground_state is None:
            self.solve()
        
        n_spin_orb = 2 * self.n_sites
        basis_arr = np.array(self.basis, dtype=np.int64)
        psi_arr = self.ground_state.astype(np.float64)
        
        Gamma = compute_2rdm_numba(psi_arr, basis_arr, n_spin_orb)
        return Gamma


# ===========================================================================
# Vorticity Calculation
# ===========================================================================

def compute_vorticity(Gamma):
    """
    Compute integrated vorticity in Λ-space.
    
    Steps:
        1. Reshape 2-RDM: Γ_pqrs → M_(pq),(rs)
        2. SVD projection: M_Λ = S^T M S
        3. Compute current: J_Λ = M_Λ · ∇M_Λ
        4. Compute vorticity: V = Σ_ij (J_ij - J_ji)²
    
    Parameters
    ----------
    Gamma : ndarray
        2-RDM tensor
    
    Returns
    -------
    dict
        'vorticity': integrated vorticity V
        'effective_rank': number of significant singular values
    """
    n = Gamma.shape[0]
    n_pairs = n * n
    
    # Reshape: Γ_pqrs → M_(pq),(rs)
    M = Gamma.reshape(n_pairs, n_pairs)
    
    # SVD projection
    U, s, Vt = svd(M)
    S = U
    effective_rank = np.sum(s > 1e-10)
    
    # Project to Λ-space: M_Λ = S^T M S
    M_lambda = S.T @ M @ S
    
    # Gradient: ∇M_Λ (finite difference)
    grad_M = np.zeros_like(M_lambda)
    grad_M[:-1, :] = np.diff(M_lambda, axis=0)
    grad_M[-1, :] = grad_M[-2, :]
    
    # Current: J_Λ = M_Λ · ∇M_Λ
    J_lambda = M_lambda @ grad_M.T
    
    # Vorticity: (∇×J)_ij = J_ij - J_ji
    curl_J = J_lambda - J_lambda.T
    vorticity = np.sum(curl_J ** 2)
    
    return {
        'vorticity': vorticity,
        'effective_rank': effective_rank
    }


# ===========================================================================
# Main Calculation
# ===========================================================================

if __name__ == "__main__":
    
    print("="*60)
    print("Λ³-DFT: Exchange-Correlation from Vorticity")
    print("="*60)
    
    # Warmup Numba JIT
    print("\n[Numba JIT compilation...]")
    _warm = HubbardModel(n_sites=2, n_electrons=2, U=2.0)
    _warm.solve()
    _warm.compute_2rdm()
    print("[Done]")
    
    # Parameters
    U_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    N_values = [4, 6, 8, 10]
    
    results = []
    
    for U in U_values:
        print(f"\n--- U/t = {U} ---")
        
        for N in N_values:
            model = HubbardModel(n_sites=N, n_electrons=N, t=1.0, U=U)
            E_total = model.solve()
            E_xc = model.compute_Exc()
            
            Gamma = model.compute_2rdm()
            vort = compute_vorticity(Gamma)
            
            alpha = E_xc / vort['vorticity'] if vort['vorticity'] > 1e-10 else 0
            
            results.append({
                'U': U, 'N': N, 'E_xc': E_xc,
                'vorticity': vort['vorticity'], 'alpha': alpha
            })
            
            print(f"  N={N}: E_xc={E_xc:.4f}, V={vort['vorticity']:.2f}, α={alpha:.6f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'U/t':>5} {'N':>4} {'E_xc':>10} {'Vorticity':>12} {'alpha':>12}")
    print("-"*50)
    for r in results:
        print(f"{r['U']:>5.1f} {r['N']:>4d} {r['E_xc']:>10.4f} {r['vorticity']:>12.2f} {r['alpha']:>12.6f}")
