"""
===========================================================================
Spin Correlation and Local Vorticity Analysis for Hubbard Ladders
===========================================================================

Computes:
1. Spin correlation function S(r) = <S_i · S_j>
2. Plaquette vorticity density (local vorticity on each 2×2 plaquette)

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import svd
from numba import njit, prange

# ===========================================================================
# Import base classes from main code (or redefine here)
# ===========================================================================

@njit(parallel=True)
def compute_2rdm_numba(psi, basis, n_spin_orb):
    """Compute 2-RDM using Numba parallelization."""
    dim = len(basis)
    Gamma = np.zeros((n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb))

    max_state = 0
    for i in range(dim):
        if basis[i] > max_state:
            max_state = basis[i]

    lookup = np.full(max_state + 1, -1, dtype=np.int64)
    for i in range(dim):
        lookup[basis[i]] = i

    for p in prange(n_spin_orb):
        for q in range(n_spin_orb):
            for r in range(n_spin_orb):
                for s in range(n_spin_orb):
                    result = 0.0
                    for k in range(dim):
                        state = basis[k]
                        if not ((state >> r) & 1):
                            continue
                        state1 = state ^ (1 << r)
                        sign1 = 0
                        for b in range(r):
                            if (state >> b) & 1:
                                sign1 += 1
                        if not ((state1 >> s) & 1):
                            continue
                        state2 = state1 ^ (1 << s)
                        sign2 = sign1
                        for b in range(s):
                            if (state1 >> b) & 1:
                                sign2 += 1
                        if (state2 >> q) & 1:
                            continue
                        state3 = state2 ^ (1 << q)
                        sign3 = sign2
                        for b in range(q):
                            if (state2 >> b) & 1:
                                sign3 += 1
                        if (state3 >> p) & 1:
                            continue
                        state4 = state3 ^ (1 << p)
                        sign4 = sign3
                        for b in range(p):
                            if (state3 >> b) & 1:
                                sign4 += 1
                        if state4 <= max_state:
                            idx4 = lookup[state4]
                            if idx4 >= 0:
                                sign = 1 if (sign4 % 2 == 0) else -1
                                result += sign * psi[idx4] * psi[k]
                    Gamma[p, q, r, s] = result
    return Gamma


class LadderHubbardModel:
    """2-leg Hubbard ladder with spin correlation analysis."""

    def __init__(self, Lx, Ly=2, n_electrons=None, t=1.0, U=4.0):
        self.Lx = Lx
        self.Ly = Ly
        self.n_sites = Lx * Ly
        self.n_electrons = n_electrons if n_electrons else self.n_sites
        self.t = t
        self.U = U
        self.ground_state = None
        self.E_total = None
        self.basis = None

    def _site_index(self, x, y):
        return x + self.Lx * y

    def _site_coords(self, i):
        """Inverse: 1D index -> (x, y)"""
        return i % self.Lx, i // self.Lx

    def _generate_basis(self):
        n = self.n_sites
        n_elec = self.n_electrons
        basis = []
        for state in range(2**(2*n)):
            if bin(state).count('1') == n_elec:
                basis.append(state)
        return basis

    def _fermion_sign(self, state, bit_from, bit_to):
        if bit_from > bit_to:
            bit_from, bit_to = bit_to, bit_from
        count = 0
        for b in range(bit_from + 1, bit_to):
            if (state >> b) & 1:
                count += 1
        return (-1) ** count

    def _hopping_bonds(self):
        bonds = []
        for y in range(self.Ly):
            for x in range(self.Lx):
                i = self._site_index(x, y)
                if x + 1 < self.Lx:
                    j = self._site_index(x + 1, y)
                    bonds.append((i, j))
                if y + 1 < self.Ly:
                    j = self._site_index(x, y + 1)
                    bonds.append((i, j))
        return bonds

    def _build_hamiltonian(self, basis):
        n = self.n_sites
        dim = len(basis)
        basis_idx = {state: i for i, state in enumerate(basis)}
        H = lil_matrix((dim, dim), dtype=np.float64)
        bonds = self._hopping_bonds()

        for i, state in enumerate(basis):
            for site in range(n):
                n_up = (state >> (2*site)) & 1
                n_down = (state >> (2*site + 1)) & 1
                H[i, i] += self.U * n_up * n_down

            for site_i, site_j in bonds:
                for spin in (0, 1):
                    bit_i = 2*site_i + spin
                    bit_j = 2*site_j + spin
                    if (state >> bit_i) & 1 and not ((state >> bit_j) & 1):
                        new_state = state ^ (1 << bit_i) ^ (1 << bit_j)
                        if new_state in basis_idx:
                            sign = self._fermion_sign(state, bit_i, bit_j)
                            H[i, basis_idx[new_state]] -= self.t * sign
                    if (state >> bit_j) & 1 and not ((state >> bit_i) & 1):
                        new_state = state ^ (1 << bit_j) ^ (1 << bit_i)
                        if new_state in basis_idx:
                            sign = self._fermion_sign(state, bit_j, bit_i)
                            H[i, basis_idx[new_state]] -= self.t * sign
        return H.tocsr()

    def solve(self):
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
        return self.E_total

    def compute_2rdm(self):
        if self.ground_state is None:
            self.solve()
        n_spin_orb = 2 * self.n_sites
        basis_arr = np.array(self.basis, dtype=np.int64)
        psi_arr = self.ground_state.astype(np.float64)
        Gamma = compute_2rdm_numba(psi_arr, basis_arr, n_spin_orb)
        return Gamma

    # ===========================================================================
    # Spin Correlation Functions
    # ===========================================================================
    
    def compute_spin_correlation(self, site_i, site_j):
        """
        Compute <S_i · S_j> = <S_i^z S_j^z> + (1/2)(<S_i^+ S_j^-> + <S_i^- S_j^+>)
        
        Using spin-orbital indexing:
          - site i, up:   2*i
          - site i, down: 2*i + 1
        
        S_i^z = (1/2)(n_i↑ - n_i↓)
        S_i^+ S_j^- = c†_i↑ c_i↓ c†_j↓ c_j↑
        """
        if self.ground_state is None:
            self.solve()
        
        psi = self.ground_state
        basis = self.basis
        dim = len(basis)
        basis_idx = {state: k for k, state in enumerate(basis)}
        
        # Spin orbital indices
        i_up, i_down = 2*site_i, 2*site_i + 1
        j_up, j_down = 2*site_j, 2*site_j + 1
        
        # <S_i^z S_j^z>
        Sz_Sz = 0.0
        for k, state in enumerate(basis):
            ni_up = (state >> i_up) & 1
            ni_down = (state >> i_down) & 1
            nj_up = (state >> j_up) & 1
            nj_down = (state >> j_down) & 1
            
            Siz = 0.5 * (ni_up - ni_down)
            Sjz = 0.5 * (nj_up - nj_down)
            
            Sz_Sz += psi[k]**2 * Siz * Sjz
        
        # <S_i^+ S_j^-> = <c†_i↑ c_i↓ c†_j↓ c_j↑>
        Sp_Sm = 0.0
        for k, state in enumerate(basis):
            # Need: j↑ occupied, j↓ empty, i↓ occupied, i↑ empty
            if not ((state >> j_up) & 1):
                continue
            if (state >> j_down) & 1:
                continue
            if not ((state >> i_down) & 1):
                continue
            if (state >> i_up) & 1:
                continue
            
            # Apply operators: c_j↑, c†_j↓, c_i↓, c†_i↑
            state1 = state ^ (1 << j_up)  # annihilate j↑
            sign1 = sum(1 for b in range(j_up) if (state >> b) & 1) % 2
            
            state2 = state1 ^ (1 << j_down)  # create j↓
            sign2 = (sign1 + sum(1 for b in range(j_down) if (state1 >> b) & 1)) % 2
            
            state3 = state2 ^ (1 << i_down)  # annihilate i↓
            sign3 = (sign2 + sum(1 for b in range(i_down) if (state2 >> b) & 1)) % 2
            
            state4 = state3 ^ (1 << i_up)  # create i↑
            sign4 = (sign3 + sum(1 for b in range(i_up) if (state3 >> b) & 1)) % 2
            
            if state4 in basis_idx:
                sign = 1 if sign4 == 0 else -1
                Sp_Sm += sign * psi[basis_idx[state4]] * psi[k]
        
        # <S_i^- S_j^+> = <c†_i↓ c_i↑ c†_j↑ c_j↓>
        Sm_Sp = 0.0
        for k, state in enumerate(basis):
            # Need: j↓ occupied, j↑ empty, i↑ occupied, i↓ empty
            if not ((state >> j_down) & 1):
                continue
            if (state >> j_up) & 1:
                continue
            if not ((state >> i_up) & 1):
                continue
            if (state >> i_down) & 1:
                continue
            
            state1 = state ^ (1 << j_down)
            sign1 = sum(1 for b in range(j_down) if (state >> b) & 1) % 2
            
            state2 = state1 ^ (1 << j_up)
            sign2 = (sign1 + sum(1 for b in range(j_up) if (state1 >> b) & 1)) % 2
            
            state3 = state2 ^ (1 << i_up)
            sign3 = (sign2 + sum(1 for b in range(i_up) if (state2 >> b) & 1)) % 2
            
            state4 = state3 ^ (1 << i_down)
            sign4 = (sign3 + sum(1 for b in range(i_down) if (state3 >> b) & 1)) % 2
            
            if state4 in basis_idx:
                sign = 1 if sign4 == 0 else -1
                Sm_Sp += sign * psi[basis_idx[state4]] * psi[k]
        
        # Total: <S_i · S_j> = <S_i^z S_j^z> + (1/2)(<S_i^+ S_j^-> + <S_i^- S_j^+>)
        return Sz_Sz + 0.5 * (Sp_Sm + Sm_Sp)

    def compute_all_spin_correlations(self):
        """Compute S(i,j) for all site pairs."""
        n = self.n_sites
        S_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    S_matrix[i, j] = 0.75  # <S_i · S_i> = S(S+1) = 3/4
                else:
                    S_matrix[i, j] = self.compute_spin_correlation(i, j)
        return S_matrix

    def compute_rung_leg_correlations(self):
        """
        Compute spin correlations along rung and leg directions.
        Returns:
            S_rung: average <S_i · S_j> for rung pairs
            S_leg: list of <S_i · S_j> for leg distances r = 1, 2, ...
        """
        # Rung correlations (y=0 to y=1 for same x)
        S_rung_list = []
        for x in range(self.Lx):
            i = self._site_index(x, 0)
            j = self._site_index(x, 1)
            S_rung_list.append(self.compute_spin_correlation(i, j))
        S_rung = np.mean(S_rung_list)
        
        # Leg correlations (same y, distance r along x)
        max_r = self.Lx - 1
        S_leg = []
        for r in range(1, max_r + 1):
            S_leg_r = []
            for y in range(self.Ly):
                for x in range(self.Lx - r):
                    i = self._site_index(x, y)
                    j = self._site_index(x + r, y)
                    S_leg_r.append(self.compute_spin_correlation(i, j))
            S_leg.append(np.mean(S_leg_r))
        
        return S_rung, S_leg

    # ===========================================================================
    # Local Vorticity (Plaquette)
    # ===========================================================================

    def compute_local_vorticity(self, Gamma):
        """
        Compute vorticity density for each 2×2 plaquette.
        
        Plaquette (x, y) consists of sites:
          (x, y), (x+1, y), (x, y+1), (x+1, y+1)
        """
        n_spin_orb = 2 * self.n_sites
        plaquettes = []
        
        for y in range(self.Ly - 1):
            for x in range(self.Lx - 1):
                # Sites in plaquette
                sites = [
                    self._site_index(x, y),
                    self._site_index(x+1, y),
                    self._site_index(x, y+1),
                    self._site_index(x+1, y+1)
                ]
                
                # Extract local 2-RDM for these sites
                spin_orbs = []
                for s in sites:
                    spin_orbs.extend([2*s, 2*s + 1])
                
                n_local = len(spin_orbs)
                local_Gamma = np.zeros((n_local, n_local, n_local, n_local))
                
                for pi, p in enumerate(spin_orbs):
                    for qi, q in enumerate(spin_orbs):
                        for ri, r in enumerate(spin_orbs):
                            for si, s in enumerate(spin_orbs):
                                local_Gamma[pi, qi, ri, si] = Gamma[p, q, r, s]
                
                # Compute vorticity for this plaquette
                n_pairs = n_local * n_local
                M = local_Gamma.reshape(n_pairs, n_pairs)
                
                U, s, Vt = svd(M)
                S = U
                M_lambda = S.T @ M @ S
                
                grad_M = np.zeros_like(M_lambda)
                grad_M[:-1, :] = np.diff(M_lambda, axis=0)
                grad_M[-1, :] = grad_M[-2, :] if len(grad_M) > 1 else 0
                
                J_lambda = M_lambda @ grad_M.T
                curl_J = J_lambda - J_lambda.T
                vorticity = np.sum(curl_J ** 2)
                
                plaquettes.append({
                    'x': x, 'y': y,
                    'sites': sites,
                    'vorticity': vorticity
                })
        
        return plaquettes


def compute_vorticity(Gamma):
    """Compute integrated vorticity (same as main code)."""
    n = Gamma.shape[0]
    n_pairs = n * n
    M = Gamma.reshape(n_pairs, n_pairs)
    U, s, Vt = svd(M)
    S = U
    M_lambda = S.T @ M @ S
    grad_M = np.zeros_like(M_lambda)
    grad_M[:-1, :] = np.diff(M_lambda, axis=0)
    grad_M[-1, :] = grad_M[-2, :]
    J_lambda = M_lambda @ grad_M.T
    curl_J = J_lambda - J_lambda.T
    vorticity = np.sum(curl_J ** 2)
    return {'vorticity': vorticity, 'effective_rank': np.sum(s > 1e-10)}


# ===========================================================================
# Main Analysis
# ===========================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("Spin Correlation & Local Vorticity Analysis")
    print("="*70)
    
    # Parameters
    Lx = 4
    Ly = 2
    U_values = [0.5, 1.0, 2.0, 3.0, 4.0]
    doping_levels = [0, 1, 2]
    
    results = []
    
    for U in U_values:
        for d in doping_levels:
            n_sites = Lx * Ly
            n_elec = n_sites - d
            if n_elec <= 0:
                continue
            
            print(f"\n--- U/t = {U}, doping d = {d} (N_e = {n_elec}) ---")
            
            model = LadderHubbardModel(Lx=Lx, Ly=Ly, n_electrons=n_elec, t=1.0, U=U)
            E = model.solve()
            
            # Spin correlations
            S_rung, S_leg = model.compute_rung_leg_correlations()
            
            print(f"  S_rung (r=1): {S_rung:.4f}")
            print(f"  S_leg:  ", end="")
            for r, S in enumerate(S_leg, 1):
                print(f"r={r}: {S:.4f}  ", end="")
            print()
            
            # Full 2-RDM and vorticity
            Gamma = model.compute_2rdm()
            vort = compute_vorticity(Gamma)
            
            # Local vorticity
            plaquettes = model.compute_local_vorticity(Gamma)
            plaq_vorts = [p['vorticity'] for p in plaquettes]
            
            print(f"  Total vorticity: {vort['vorticity']:.2e}")
            print(f"  Plaquette vorticity: mean={np.mean(plaq_vorts):.2e}, std={np.std(plaq_vorts):.2e}")
            
            results.append({
                'U': U,
                'd': d,
                'S_rung': S_rung,
                'S_leg': S_leg,
                'total_V': vort['vorticity'],
                'plaq_vorts': plaq_vorts,
                'plaquettes': plaquettes
            })
    
    # ===========================================================================
    # Plotting
    # ===========================================================================
    
    print("\n" + "="*70)
    print("Generating Figures...")
    print("="*70)
    
    # Figure 3a: Spin Correlation Function
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # (a) S(r) for different U at half-filling (d=0)
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(U_values)))
    
    for i, U in enumerate(U_values):
        # Find d=0 result
        for r in results:
            if r['U'] == U and r['d'] == 0:
                S_leg = r['S_leg']
                S_rung = r['S_rung']
                r_vals = list(range(1, len(S_leg) + 1))
                
                # Plot leg correlations
                ax1.plot(r_vals, S_leg, 'o-', color=colors[i], 
                        label=f'U/t={U} (leg)', markersize=8)
                
                # Mark rung correlation at r=0.5 (symbolic)
                ax1.plot(0.5, S_rung, 's', color=colors[i], 
                        markersize=12, markeredgecolor='black')
                break
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=-0.75, color='red', linestyle=':', alpha=0.5, label='Singlet limit')
    ax1.set_xlabel('Distance r (along leg)', fontsize=12)
    ax1.set_ylabel(r'$\langle S_i \cdot S_j \rangle$', fontsize=12)
    ax1.set_title('(a) Spin Correlation: Half-filling (d=0)', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.set_xlim(-0.1, Lx)
    ax1.set_ylim(-0.8, 0.3)
    ax1.text(0.3, -0.6, 'Rung\n(singlet)', fontsize=10, ha='center')
    
    # (b) Plaquette vorticity heatmap for different conditions
    ax2 = axes[1]
    
    # Compare d=0 vs d=2 at U=2.0
    U_compare = 2.0
    d_compare = [0, 2]
    
    heatmap_data = []
    for d in d_compare:
        for r in results:
            if r['U'] == U_compare and r['d'] == d:
                # Create 2D array for plaquettes
                plaq_grid = np.zeros((Ly-1, Lx-1))
                for p in r['plaquettes']:
                    plaq_grid[p['y'], p['x']] = p['vorticity']
                heatmap_data.append((d, plaq_grid))
                break
    
    if len(heatmap_data) == 2:
        # Side by side comparison
        combined = np.vstack([heatmap_data[0][1], np.zeros((1, Lx-1))*np.nan, heatmap_data[1][1]])
        im = ax2.imshow(combined, cmap='hot', aspect='auto')
        ax2.set_title(f'(b) Plaquette Vorticity Density (U/t={U_compare})', fontsize=14)
        ax2.set_xlabel('Plaquette x', fontsize=12)
        ax2.set_ylabel('d=0 (top) | d=2 (bottom)', fontsize=12)
        plt.colorbar(im, ax=ax2, label='Vorticity')
    
    plt.tight_layout()
    plt.savefig('/content/fig3_spin_vorticity.png', dpi=150, bbox_inches='tight')
    plt.savefig('/content/fig3_spin_vorticity.pdf', bbox_inches='tight')
    print("[Saved] fig3_spin_vorticity.png/pdf")
    

    # ===========================================================================
    # Summary Table
    # ===========================================================================
    
    print("\n" + "="*70)
    print("SUMMARY: Spin Correlations")
    print("="*70)
    print(f"{'U/t':>5} {'d':>3} {'S_rung':>10} {'S_leg(r=1)':>12} {'S_leg(r=2)':>12}")
    print("-"*50)
    for r in results:
        S_leg1 = r['S_leg'][0] if len(r['S_leg']) > 0 else 0
        S_leg2 = r['S_leg'][1] if len(r['S_leg']) > 1 else 0
        print(f"{r['U']:>5.1f} {r['d']:>3d} {r['S_rung']:>10.4f} {S_leg1:>12.4f} {S_leg2:>12.4f}")
    
    print("\n[Analysis Complete]")
