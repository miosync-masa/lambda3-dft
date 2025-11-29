# ===========================================================
# FULL SCAN: Ladder × U × Electron Number
# ===========================================================

Lx_values = [2, 3, 4, 5]  # optional: add 5
U_values  = [0.5, 1.0, 1.5, 2.0, 2.5]
doping_levels = [0, 1, 2]  # electrons removed (hole doping)

results = []

for Lx in Lx_values:
    Ly = 2
    n_sites = Lx * Ly

    for U in U_values:

        for d in doping_levels:
            n_elec = n_sites - d
            if n_elec <= 0:
                continue

            print(f"\n=== Lx={Lx}, U/t={U}, N_e={n_elec} (d={d}) ===")

            model = LadderHubbardModel(
                Lx=Lx, Ly=Ly,
                n_electrons=n_elec,
                t=1.0, U=U
            )
            E_total = model.solve()
            E_xc = model.compute_Exc()

            Gamma = model.compute_2rdm()
            vort = compute_vorticity(Gamma)

            alpha = E_xc / vort['vorticity'] if vort['vorticity'] > 1e-12 else 0.0

            results.append({
                'Lx': Lx,
                'U': U,
                'nelec': n_elec,
                'holes': d,
                'Exc': E_xc,
                'V': vort['vorticity'],
                'alpha': alpha,
                'rank': vort['effective_rank'],
            })

            print(f"  Exc={E_xc:.4f}, V={vort['vorticity']:.2e}, "
                  f"alpha={alpha:.6f}, rank={vort['effective_rank']}")
