"""Choose the gingival margin height h_f: simulated visible fractions per (jaw, tooth family) against the visible
fractions of real IOS crowns registered to their CBCT teeth. Only summary statistics of the real pairs are used.

    # one statistics-only run per candidate value
    python -m cbct2ios simulate --root TF3 --audit AUDIT --splits train --stats-only \
        --params '{"h_f": 2.5}' --output-dir calib/h_f_2.5
    python -m cbct2ios.calibrate --real real_fractions.csv --runs calib/h_f_* --output calib/report.json

`real_fractions.csv` has one row per real tooth with the columns jaw (upper/lower), family
(incisor/canine/premolar/molar) and visible_fraction (scanned crown area / whole tooth area).
The run with the smallest tooth-weighted absolute difference of the group medians is reported as `best`.
With the paired data used for the released recipe, h_f = 3.0 mm was the best of 1.5-3.0 mm.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import read_csv, write_json
from .simulate import CAUSES


def groups(rows, value):
    out = defaultdict(list)
    for r in rows:
        out[(r['jaw'], r['family'])].append(value(r))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--real', type=Path, required=True, help='CSV: jaw, family, visible_fraction per real tooth')
    parser.add_argument('--runs', type=Path, nargs='+', required=True, help='output folders of simulate --stats-only')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    real = groups(read_csv(args.real), lambda r: float(r['visible_fraction']))
    report = dict(real={f'{j}_{f}': dict(teeth=len(v), median=float(np.median(v))) for (j, f), v in sorted(real.items())},
                  runs={})
    for folder in args.runs:
        rows = [r for p in sorted((folder / 'case_manifests').glob('*.csv')) if p.stat().st_size for r in read_csv(p)]
        eligible = [r for r in rows if not r['exclusion']]
        simulated = groups(eligible, lambda r: float(r['visible_fraction']))
        table, weighted, total = {}, 0., 0
        for key in sorted(real):
            if key not in simulated:
                continue
            difference = float(np.median(simulated[key]) - np.median(real[key]))
            table[f'{key[0]}_{key[1]}'] = dict(teeth=len(simulated[key]), median=float(np.median(simulated[key])),
                                               difference=difference)
            weighted += abs(difference) * len(simulated[key])
            total += len(simulated[key])
        report['runs'][folder.name] = dict(
            teeth=len(eligible), cases=len({r['case_id'] for r in rows}), weighted_abs_difference=weighted / max(total, 1),
            groups=table, cause_fractions={c: float(np.median([float(r[f'{c}_area_mm2']) / float(r['total_area_mm2'])
                                                               for r in eligible])) for c in CAUSES} if eligible else {})
    best = min(report['runs'], key=lambda k: report['runs'][k]['weighted_abs_difference'])
    report['best'] = best
    write_json(args.output, report)
    for name, run in report['runs'].items():
        print(name, f"teeth {run['teeth']} weighted |diff| {run['weighted_abs_difference']:.4f}",
              ' '.join(f"{k}:{v['median']:.3f}({v['difference']:+.3f})" for k, v in run['groups'].items()))
    print('best', best)


if __name__ == '__main__':
    main()
