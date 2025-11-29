"""
Two-Axis Phase Diagram: γ and α vs Doping
"""

import numpy as np
import matplotlib.pyplot as plt

# Data
U_values = [0.5, 1.0, 1.5, 2.0, 2.5]
d_values = [0, 1, 2]

gamma_data = {
    0.5: [0.630, 1.290, 1.593],
    1.0: [0.648, 1.256, 1.573],
    1.5: [0.681, 1.200, 1.550],
    2.0: [0.734, 1.125, 1.517],
    2.5: [0.803, 1.036, 1.467],
}

alpha_Lx5 = {
    0.5: [0.001208, 0.000872, 0.000912],
    1.0: [0.002446, 0.001760, 0.001821],
    1.5: [0.003834, 0.002752, 0.002797],
    2.0: [0.005488, 0.003935, 0.003898],
    2.5: [0.007519, 0.005396, 0.005178],
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (a) γ vs doping
ax1 = axes[0]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(U_values)))
for i, U in enumerate(U_values):
    ax1.plot(d_values, gamma_data[U], 'o-', color=colors[i],
             markersize=10, linewidth=2, label=f'U/t={U}')

ax1.axvline(0, color='blue', linestyle='--', alpha=0.5, label='Mott (γ min)')
ax1.set_xlabel('Doping d (holes)', fontsize=14)
ax1.set_ylabel('γ (correlation dimension)', fontsize=14)
ax1.set_title('(a) γ vs Doping: Minimum at Mott (d=0)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks([0, 1, 2])
ax1.set_xticklabels(['0\n(Mott)', '1\n(SC?)', '2\n(Metal)'])

# (b) α vs doping
ax2 = axes[1]
for i, U in enumerate(U_values):
    ax2.plot(d_values, alpha_Lx5[U], 's-', color=colors[i],
             markersize=10, linewidth=2, label=f'U/t={U}')

ax2.axvline(1, color='red', linestyle='--', alpha=0.5, label='SC (α min)')
ax2.set_xlabel('Doping d (holes)', fontsize=14)
ax2.set_ylabel('α = E_xc / Vorticity', fontsize=14)
ax2.set_title('(a) α vs Doping: Minimum at SC (d=1)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(['0\n(Mott)', '1\n(SC?)', '2\n(Metal)'])

plt.suptitle('Λ³-DFT: Two-Axis Phase Diagram', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('fig5_two_axis_phase_diagram.png', dpi=150, bbox_inches='tight')
print("[Saved] fig5_two_axis_phase_diagram.png")
