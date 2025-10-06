
#!/usr/bin/env python3
# saltelli_driver.py
"""
Sample u ~ U[0,1]^10 with SALib's Saltelli sequence, generate analytic soundings,
compute diagnostics, and (optionally) write WRF input_sounding files.

Usage:
  python saltelli_driver.py --n_base 256 --out env_catalog.csv --write_soundings ./soundings
Requires SALib if you want Saltelli sampling (otherwise you can pass a CSV of u's).
"""
import argparse, os, csv, numpy as np
from envgen_analytic import generate_sounding, diagnostics, write_input_sounding

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_base', type=int, default=256, help='Base samples for Saltelli')
    ap.add_argument('--out', default='env_catalog.csv')
    ap.add_argument('--write_soundings', default=None, help='Directory to write input_sounding files (optional)')
    ap.add_argument('--u_csv', default=None, help='Optional CSV with rows of u in [0,1]^10 instead of sampling')
    args = ap.parse_args()

    # Build u-samples
    if args.u_csv:
        U = np.loadtxt(args.u_csv, delimiter=',')
        if U.ndim == 1:
            U = U[None, :]
    else:
        try:
            from SALib.sample import saltelli
            from SALib.util import scale_samples
            problem = {'num_vars': 10, 'names': [f'u{i}' for i in range(10)], 'bounds': [[0,1]]*10}
            U = saltelli.sample(problem, args.n_base, calc_second_order=False)
        except Exception as e:
            raise SystemExit("SALib not available. Provide --u_csv CSV of u in [0,1]^10 or install SALib.") from e

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    if args.write_soundings:
        os.makedirs(args.write_soundings, exist_ok=True)

    rows = []
    for i, u in enumerate(U):
        sound = generate_sounding(u.tolist())
        diag = diagnostics(sound)
        rows.append({'idx': i, **diag})
        if args.write_soundings:
            write_input_sounding(os.path.join(args.write_soundings, f'input_sounding_{i:05d}'), sound)

    # Write diagnostics catalog
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    if args.write_soundings:
        print(f"Soundings in {args.write_soundings}")

if __name__ == '__main__':
    main()
