"""
===========================================================================
Plaquette Vorticity Heatmap: 9-Panel Figure
===========================================================================

U/t = 2.0, 2.5, 3.0 × d = 0, 1, 2

Author: Masamichi Iizumi, Tamaki Iizumi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import svd
from numba import njit, prange

# ===========================================================================
# Numba-accelerated 2-RDM (same as before)
# ===========================================================================

@njit(parallel=True)
def compute_2rdm_numba(psi, basis, n_spin_orb):
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

    def compute_local_vorticity(self, Gamma):
        """Compute vorticity density for each 2×2 plaquette."""
        plaquettes = []
        for y in range(self.Ly - 1):
            for x in range(self.Lx - 1):
                sites = [
                    self._site_index(x, y),
                    self._site_index(x+1, y),
                    self._site_index(x, y+1),
                    self._site_index(x+1, y+1)
                ]
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
                n_pairs = n_local * n_local
                M = local_Gamma.reshape(n_pairs, n_pairs)
                U_svd, s_svd, Vt = svd(M)
                S = U_svd
                M_lambda = S.T @ M @ S
                grad_M = np.zeros_like(M_lambda)
                grad_M[:-1, :] = np.diff(M_lambda, axis=0)
                grad_M[-1, :] = grad_M[-2, :] if len(grad_M) > 1 else 0
                J_lambda = M_lambda @ grad_M.T
                curl_J = J_lambda - J_lambda.T
                vorticity = np.sum(curl_J ** 2)
                plaquettes.append({
                    'x': x, 'y': y,
                    'vorticity': vorticity
                })
        return plaquettes


# ===========================================================================
# Main: 9-Panel Heatmap
# ===========================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("9-Panel Plaquette Vorticity Heatmap")
    print("="*70)
    
    # Parameters
    Lx = 5  # 2×5 ladder → 4 plaquettes
    Ly = 2
    U_values = [2.0, 2.5, 3.0]
    doping_levels = [0, 1, 2]
    
    # Storage for all data
    all_data = {}
    
    print("\n[Computing...]")
    
    for U in U_values:
        for d in doping_levels:
            n_sites = Lx * Ly
            n_elec = n_sites - d
            
            print(f"  U/t={U}, d={d}...", end=" ", flush=True)
            
            model = LadderHubbardModel(Lx=Lx, Ly=Ly, n_electrons=n_elec, t=1.0, U=U)
            E = model.solve()
            Gamma = model.compute_2rdm()
            plaquettes = model.compute_local_vorticity(Gamma)
            
            # Create grid (1 row × Lx-1 columns for 2-leg ladder)
            plaq_grid = np.zeros((1, Lx-1))
            for p in plaquettes:
                plaq_grid[0, p['x']] = p['vorticity']
            
            all_data[(U, d)] = {
                'grid': plaq_grid,
                'mean': np.mean([p['vorticity'] for p in plaquettes]),
                'std': np.std([p['vorticity'] for p in plaquettes])
            }
            
            print(f"mean={all_data[(U,d)]['mean']:.2f}")
    
    # ===========================================================================
    # Create 9-Panel Figure
    # ===========================================================================
    
    print("\n[Generating Figure...]")
    
    # Find global min/max for consistent colormap
    all_vorts = []
    for key, val in all_data.items():
        all_vorts.extend(val['grid'].flatten())
    vmin, vmax = min(all_vorts), max(all_vorts)
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    
    # Title
    fig.suptitle(r'Plaquette Vorticity Density: $\Lambda^3$-DFT', fontsize=16, fontweight='bold')
    
    # Labels for rows and columns
    row_labels = ['d=0\n(Mott)', 'd=1\n(SC?)', 'd=2\n(Metal)']
    col_labels = ['U/t = 2.0', 'U/t = 2.5', 'U/t = 3.0']
    
    # Normalize colormap
    norm = Normalize(vmin=vmin*0.9, vmax=vmax*1.1)
    
    for i, d in enumerate(doping_levels):
        for j, U in enumerate(U_values):
            ax = axes[i, j]
            data = all_data[(U, d)]['grid']
            
            # Plot heatmap
            im = ax.imshow(data, cmap='hot', norm=norm, aspect='auto')
            
            # Add value annotations
            for x in range(data.shape[1]):
                val = data[0, x]
                color = 'white' if val > (vmax + vmin) / 2 else 'black'
                ax.text(x, 0, f'{val:.1f}', ha='center', va='center', 
                       fontsize=10, color=color, fontweight='bold')
            
            # Labels
            if i == 0:
                ax.set_title(col_labels[j], fontsize=14, fontweight='bold')
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=12, fontweight='bold')
            
            # Remove ticks
            ax.set_xticks(range(Lx-1))
            ax.set_xticklabels([f'P{k+1}' for k in range(Lx-1)], fontsize=9)
            ax.set_yticks([])
            
            # Add mean value
            mean_val = all_data[(U, d)]['mean']
            ax.text(0.98, 0.02, f'μ={mean_val:.1f}', transform=ax.transAxes,
                   fontsize=9, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Vorticity', fontsize=12)
    
    # Add annotations for physics
    fig.text(0.02, 0.85, '← Localized\n   (high V)', fontsize=10, 
             ha='left', va='top', color='darkred')
    fig.text(0.02, 0.25, '← Delocalized\n   (low V)', fontsize=10,
             ha='left', va='bottom', color='darkblue')
    
    plt.tight_layout(rect=[0.05, 0, 0.9, 0.95])
    
    # Save
    plt.savefig('/content/fig3b_heatmap_9panel.png', dpi=150, bbox_inches='tight')
    plt.savefig('/content/fig3b_heatmap_9panel.pdf', bbox_inches='tight')
    print("[Saved] fig3b_heatmap_9panel.png/pdf")
    
    # ===========================================================================
    # Print Summary Table
    # ===========================================================================
    
    print("\n" + "="*70)
    print("SUMMARY: Plaquette Vorticity (mean ± std)")
    print("="*70)
    print(f"{'':>8} {'U/t=2.0':>15} {'U/t=2.5':>15} {'U/t=3.0':>15}")
    print("-"*55)
    for d in doping_levels:
        row = f"d={d:>3}  "
        for U in U_values:
            m = all_data[(U, d)]['mean']
            s = all_data[(U, d)]['std']
            row += f"{m:>6.1f} ± {s:<5.1f}  "
        print(row)
    
    print("\n[Done]")
