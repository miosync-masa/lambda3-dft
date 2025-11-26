# Λ³-DFT: Density Functional Theory Without Magic Numbers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Exchange-correlation energy derived from two-particle vorticity in Λ-space.

## Paper

**"Density Functional Theory Without Magic Numbers: Exchange-Correlation from Vorticity"**

Submitted to *Physical Review B* (2025)

## Key Results

- **0.5% mean error** vs Bethe Ansatz exact solution
- Discovery of **correlation dimension γ(U/t) = 2.43 - 0.27(U/t)**
- Geometric interpretation of Mott metal-insulator transition

## Universal Formula
```
E_xc = A × √(U/t) × N^(-γ(U/t)) × ∫(∇×J_Λ)² dA
```

where:
- `γ(U/t)`: correlation dimension (2→0 as U/t increases)
- `J_Λ`: current in Λ-space from 2-RDM
- `∇×J_Λ`: vorticity (non-commutativity measure)

## Installation
```bash
git clone https://github.com/miosync-masa/lambda3-dft.git
cd lambda3-dft
pip install -r requirements.txt
```

## Usage

### Run main calculation
```bash
python lambda3_dft_hubbard.py
```

### Run validation analysis
```bash
python validation_analysis.py
```

## Files

| File | Description |
|------|-------------|
| `lambda3_dft_hubbard.py` | Main calculation (Hubbard model, 2-RDM, vorticity) |
| `validation_analysis.py` | Bethe Ansatz comparison, γ fitting, figures |
| `data/results_summary.csv` | Raw data (40 points) |
| `figures/` | Publication figures |

## Citation
```bibtex
@article{Iizumi2025,
  author = {Iizumi, Masamichi},
  title = {Density Functional Theory Without Magic Numbers: 
           Exchange-Correlation from Vorticity},
  journal = {Physical Review B},
  year = {2025},
  note = {Submitted}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Masamichi Iizumi (Miosync Inc.)
