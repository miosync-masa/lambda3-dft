# Λ³-DFT: Two-Leg Hubbard Ladder Analysis

**Part II of the Λ³-DFT series**

## Overview

This directory contains the code, data, and figures for the two-leg Hubbard ladder analysis, extending the Λ³-DFT framework to quasi-2D systems.

## Key Findings

1. **Rung Absorption**: γ_ladder ≈ γ_chain - 1 ≈ 1 (rung singlets absorb one correlation dimension)
2. **Two-Axis Phase Diagram**:
   - γ-axis: minimum at d=0 (Mott insulator)
   - α-axis: minimum at d=1 (optimal doping / SC dome)
3. **Edge Effect**: d=1 shows 2× vorticity at edges vs bulk → explains R² degradation

## Data Files

| File | Description |
|------|-------------|
| `ladder_gamma.csv` | Correlation dimension comparison: chain vs ladder |
| `plaquette_vorticity.csv` | 9-panel heatmap data (U/t × doping) |
| `spin_correlation.csv` | Rung spin correlations and vorticity |

## Figures

| Figure | Description |
|--------|-------------|
| `fig1.pdf` | (a) γ vs U/t for chain and ladder (b) R² vs U/t |
| `fig2.pdf` | Two-axis phase diagram: (a) γ vs d (b) α vs d |
| `fig3a_spin_rung.pdf` | Rung spin correlation vs U/t |
| `fig3b_heatmap_9panel.pdf` | Plaquette vorticity density heatmap |

## Citation

If you use this code or data, please cite:

```bibtex
@article{Iizumi2025ladder,
  author = {Iizumi, Masamichi},
  title = {Geometric Origin of Optimal Doping: Two-Axis Phase Diagram from $\Lambda^3$-DFT},
  journal = {Physical Review B},
  year = {2025},
  note = {submitted}
}
```

## License

MIT License - See parent repository

## Related

- Part I: [Density Functional Theory Without Magic Numbers](../README.md)
