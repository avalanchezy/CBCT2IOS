"""ToothFairy3 -> simulated intraoral scans, command line.

    python -m cbct2ios audit     --root TF3 --audit AUDIT           per-case label audit (JSON per case)
    python -m cbct2ios aggregate --audit AUDIT                      duplicate groups, patient-level splits
    python -m cbct2ios simulate  --root TF3 --audit AUDIT --output-dir OUT [--cases F_009 ...]
    python -m cbct2ios manifest  --output-dir OUT                   all_teeth.csv, manifest.csv, summary.json

`simulate` runs the simulator (cbct2ios.simulate) on every eligible jaw and writes, per case and jaw, the simulated
scan (visible crowns, visible parts of crowns/bridges, gingiva), every tooth surface with the occlusion cause of each
face, and one statistics row per tooth. Without --audit the cases are audited on the fly (no split column).
Only the label volumes (labelsTr) are read; the CBCT intensities are not needed.

Default recipe (the one used for the released training data):
  --smooth-teeth  every tooth surface smoothed (vertices held within 0.15 mm of the label surface) and
                  Loop-subdivided once, before the visible crown and the hidden part are cut from it
  --densify-ios   every scan piece smoothed and resampled to the vertex density of real IOS (30 vertices/mm^2)
Use --no-smooth-teeth / --no-densify-ios for the raw marching-cubes surfaces of the 0.3 mm labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import trimesh

from . import audit as audit_module
from .geometry import densify_open_mesh, family, loop_subdivide, smooth_label_surface
from .io import read_csv, write_csv, write_json, write_mesh
from .simulate import BRIDGE, CAUSES, CROWN, PARAMS, simulate_jaw


MIN_JAW_TEETH = audit_module.MIN_JAW_TEETH
MIN_EXPOSED = 4.5          # crown top above the median bone crest (mm); below: unerupted / impacted
MIN_VISIBLE_AREA = 15.     # mm^2 of simulated IOS crown
MAX_AXIS_ANGLE = 45.       # degrees between tooth and jaw axis
MIN_PIECE_FACES = 10       # visible patches with fewer faces are not part of the scan
CAUSE_HEX = ('#2A78D6',   # visible: the simulated IOS crown
             '#B98B4E',   # in bone
             '#E87BA4',   # under the gingiva
             '#EDA100',   # proximal (contact with a neighbour, or hidden behind it)
             '#6250D6')   # other hidden supragingival faces
CAUSE_COLORS = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] + [255] for h in CAUSE_HEX], np.uint8)
PROSTHETIC_NAMES = {BRIDGE: 'bridge', CROWN: 'crown'}


def submesh(mesh, faces):
    part = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)[faces], process=False)
    part.remove_unreferenced_vertices()
    return part


def densify_teeth(sim):
    """Densified ground truth: every tooth and prosthetic surface of the simulated jaw is smoothed with every vertex
    held within 0.15 mm of the label surface and Loop-subdivided once. Visibility and occlusion causes are per face;
    the four children of a face inherit them, so the visible crown and the hidden part are cut from the same densified
    surface. Runs after the simulation, which ray-casts on the label surface."""
    def dense(mesh):
        smooth = smooth_label_surface(mesh)
        return trimesh.Trimesh(*loop_subdivide(smooth.vertices, smooth.faces), process=False)
    for t in sim['teeth'].values():
        t['mesh'] = dense(t['mesh'])
        for key in ('visible', 'causes'):
            if key in t:
                t[key] = np.tile(t[key], 4)
        t['smoothed'] = True
    for p in sim['prosthetics']:
        p['mesh'] = dense(p['mesh'])
        if 'visible' in p:
            p['visible'] = np.tile(p['visible'], 4)
    return sim


def scan_pieces(sim, densify=True):
    """The simulated scan: visible crown per tooth, visible parts of crowns/bridges, gingiva."""
    crowns = {f: submesh(t['mesh'], t['visible']) for f, t in sim['teeth'].items() if t['visible'].sum() >= MIN_PIECE_FACES}
    prosthetics = [(p['label'], submesh(p['mesh'], p['visible'])) for p in sim['prosthetics']
                   if p['visible'].sum() >= MIN_PIECE_FACES]
    gingiva = sim['gingiva']
    if densify:
        crowns = {f: densify_open_mesh(m) for f, m in crowns.items()}
        prosthetics = [(label, densify_open_mesh(m)) for label, m in prosthetics]
        gingiva = densify_open_mesh(gingiva)
    return dict(crowns=crowns, prosthetics=prosthetics, gingiva=gingiva)


def target_exclusion(audit_tooth, stats):
    if audit_tooth['audit_exclusion']:
        return audit_tooth['audit_exclusion']
    if stats is None:
        return 'no_bone_crest'
    reasons = [name for name, bad in (('unerupted', stats['exposed_height_mm'] < MIN_EXPOSED),
                                      ('small_crown', stats['visible_area_mm2'] < MIN_VISIBLE_AREA),
                                      ('tilted', stats['axis_angle_deg'] > MAX_AXIS_ANGLE)) if bad]
    return ';'.join(reasons)


def export_jaw(sim, pieces, folder, recipe):
    """Meshes in world millimetres (the affine of the label volume)."""
    write_mesh(folder / 'gingiva.ply', pieces['gingiva'])
    for fdi, crown in pieces['crowns'].items():
        write_mesh(folder / 'crowns' / f'{fdi}.ply', crown)
    for k, (label, mesh) in enumerate(pieces['prosthetics']):
        write_mesh(folder / 'prosthetics' / f'{PROSTHETIC_NAMES[label]}_{k}.ply', mesh)
    scan = trimesh.util.concatenate([pieces['gingiva'], *pieces['crowns'].values(), *(m for _, m in pieces['prosthetics'])])
    write_mesh(folder / 'scan.ply', scan)
    faces = {}
    for fdi, t in sim['teeth'].items():
        write_mesh(folder / 'teeth' / f'{fdi}.ply', t['mesh'], CAUSE_COLORS[t['causes']])
        faces[f'{fdi}_visible'] = t['visible']
        faces[f'{fdi}_cause'] = t['causes'].astype(np.uint8)
    np.savez_compressed(folder / 'faces.npz', **faces)
    write_json(folder / 'jaw.json', dict(
        jaw_axis=sim['jaw_axis'], arch_center=sim['arch_center'], metal=sim['metal'], params=sim['params'], recipe=recipe,
        causes=list(CAUSES), teeth=[int(f) for f in sim['order']], failed={str(k): v for k, v in sim['failed'].items()},
        scan=dict(vertices=len(scan.vertices), faces=len(scan.faces), area_mm2=float(scan.area))))


def simulate_case(root, case, case_row, audit, output, *, params=None, stats_only=False, densify=True, smooth_teeth=True):
    seg, affine = audit_module.load_labels(root, case)
    rows, statuses, jaws = [], [], []
    recipe = dict(teeth_mesh='densified' if smooth_teeth else 'label_surface', ios_mesh='densified' if densify else 'label_surface')
    for jaw in ('upper', 'lower'):
        if str(case_row[f'{jaw}_eligible']) != 'True':
            statuses.append(dict(case_id=case, jaw=jaw, status='jaw_ineligible'))
            continue
        began = time.perf_counter()
        try:
            sim = simulate_jaw(seg, affine, jaw, params)
        except Exception as error:  # one failing jaw must not lose the other jaw of the case
            statuses.append(dict(case_id=case, jaw=jaw, status='simulation_error', detail=repr(error)))
            print(f'{case} {jaw}: simulation error {error!r}', flush=True)
            continue
        if sim['status'] != 'ok':
            statuses.append(dict(case_id=case, jaw=jaw, status=sim['status']))
            continue
        simulated = time.perf_counter()
        pieces = None
        if not stats_only:
            try:
                if smooth_teeth:
                    densify_teeth(sim)
                pieces = scan_pieces(sim, densify)
                export_jaw(sim, pieces, output / 'cases' / case / jaw, recipe)
            except Exception as error:
                statuses.append(dict(case_id=case, jaw=jaw, status='export_error', detail=repr(error)))
                print(f'{case} {jaw}: export error {error!r}', flush=True)
                continue
        jaws.append(dict(case_id=case, jaw=jaw, status='ok', simulate_seconds=simulated - began,
                         export_seconds=time.perf_counter() - simulated, metal=sim['metal'],
                         teeth=len(sim['teeth']), prosthetics=len(sim['prosthetics'])))
        for fdi, t in sim['teeth'].items():
            audit_tooth = audit['teeth'].get(str(fdi), dict(audit_exclusion='not_audited'))
            stats = t.get('stats')
            row = dict(sample_id=f'tf3_{case}_{fdi}', case_id=case, subset=case_row['subset'], fdi=fdi, family=family(fdi),
                       jaw=jaw, split=case_row.get('split', ''), source_group_id=case_row.get('source_group_id', case),
                       grade='M' if sim['metal'] else 'A', exclusion=target_exclusion(audit_tooth, stats), **(stats or {}))
            if pieces is not None:
                crown = pieces['crowns'].get(fdi)
                row.update(tooth_mesh=f'cases/{case}/{jaw}/teeth/{fdi}.ply',
                           crown_mesh=f'cases/{case}/{jaw}/crowns/{fdi}.ply' if crown is not None else '',
                           crown_vertices=len(crown.vertices) if crown is not None else 0,
                           crown_vertices_per_mm2=len(crown.vertices) / crown.area if crown is not None and crown.area else 0.)
            rows.append(row)
            if stats:
                print(f"{row['sample_id']}: {row['exclusion'] or 'target'} visible {stats['visible_fraction']:.2f}", flush=True)
    write_csv(output / 'case_manifests' / f'{case}.csv', rows)
    write_json(output / 'case_status' / f'{case}.json', dict(statuses=statuses, jaws=jaws))
    return rows


def manifest(output):
    rows = [r for path in sorted((output / 'case_manifests').glob('*.csv')) if path.stat().st_size for r in read_csv(path)]
    # A jaw whose alveolar bone is mostly outside the field of view (upper jaws of the P subset) has gingiva and
    # visible crowns on a few teeth only: no realistic scan context, so none of its teeth are targets.
    scene = Counter((r['case_id'], r['jaw']) for r in rows if r.get('visible_fraction'))
    for r in rows:
        if not r['exclusion'] and scene[(r['case_id'], r['jaw'])] < MIN_JAW_TEETH:
            r['exclusion'] = 'sparse_jaw_scene'
    targets = [r for r in rows if not r['exclusion']]
    write_csv(output / 'all_teeth.csv', rows)
    write_csv(output / 'manifest.csv', targets)
    summary = dict(teeth_simulated=len(rows), targets=len(targets), cases=len({r['case_id'] for r in targets}),
                   exclusions=dict(Counter(e for r in rows for e in r['exclusion'].split(';') if e)),
                   by_split={s: dict(Counter(r['grade'] for r in targets if r['split'] == s)) for s in sorted({r['split'] for r in targets})},
                   by_family=dict(Counter(r['family'] for r in targets)), by_subset=dict(Counter(r['subset'] for r in targets)),
                   visible_fraction_median=float(np.median([float(r['visible_fraction']) for r in targets])) if targets else None,
                   cause_fraction_median={c: float(np.median([float(r[f'{c}_area_mm2']) / float(r['total_area_mm2']) for r in targets]))
                                          for c in CAUSES} if targets else None)
    write_json(output / 'summary.json', summary)
    return summary


def label_cases(root):
    return sorted(audit_module.case_name(p) for p in (Path(root) / 'labelsTr').glob('ToothFairy3*.nii.gz'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('audit', 'aggregate', 'simulate', 'manifest'))
    parser.add_argument('--root', type=Path, help='ToothFairy3 folder (with labelsTr)')
    parser.add_argument('--audit', type=Path, help='audit folder (audit/<case>.json, cases.csv)')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--cases', nargs='+', help='case ids such as F_009 P_147 S_0000')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--params', default='{}', help=f'JSON overrides of the simulation parameters {sorted(PARAMS)}')
    parser.add_argument('--stats-only', action='store_true', help='simulate and record statistics, write no meshes')
    parser.add_argument('--densify-ios', action=argparse.BooleanOptionalAction, default=True,
                        help='scan pieces smoothed and resampled to real IOS density (default on)')
    parser.add_argument('--smooth-teeth', action=argparse.BooleanOptionalAction, default=True,
                        help='tooth surfaces smoothed (<= 0.15 mm) and Loop-subdivided (default on)')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.action == 'audit':
        cases = [c for c in label_cases(args.root) if not args.cases or c in args.cases][args.shard_index::args.shards]
        for case in cases:
            path = args.audit / 'audit' / f'{case}.json'
            if args.resume and path.exists():
                continue
            write_json(path, audit_module.audit_case(args.root, case))
            print(case, flush=True)
        return
    if args.action == 'aggregate':
        print(json.dumps(audit_module.aggregate(args.audit), indent=2), flush=True)
        return
    output = args.output_dir.resolve()
    if args.action == 'manifest':
        print(json.dumps(manifest(output), indent=2), flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    params = json.loads(args.params)
    unknown = set(params) - set(PARAMS)
    if unknown:
        parser.error(f'unknown simulation parameters: {sorted(unknown)}')
    if args.audit:
        case_rows = {r['case_id']: r for r in read_csv(args.audit / 'cases.csv') if r['split'] in args.splits}
    else:
        case_rows = {c: None for c in label_cases(args.root)}
    cases = sorted(c for c in case_rows if not args.cases or c in args.cases)
    for case in cases[args.shard_index::args.shards]:
        if args.resume and (output / 'case_manifests' / f'{case}.csv').exists():
            continue
        began = time.perf_counter()
        try:
            if args.audit:
                audit = json.loads((args.audit / 'audit' / f'{case}.json').read_text())
                row = case_rows[case]
            else:
                audit = audit_module.audit_case(args.root, case)
                row = dict(subset=audit['subset'], **{f'{j}_eligible': audit['jaws'][j]['eligible'] for j in ('upper', 'lower')})
            simulate_case(args.root, case, row, audit, output, params=params, stats_only=args.stats_only,
                          densify=args.densify_ios, smooth_teeth=args.smooth_teeth)
        except (OSError, ValueError, RuntimeError, MemoryError) as error:
            write_json(output / 'case_status' / f'{case}.json', dict(statuses=[dict(case_id=case, status='case_error',
                                                                                     detail=str(error))]))
            print(f'{case}: {error}', flush=True)
        print(f'{case}: {time.perf_counter() - began:.0f} s', flush=True)
    if args.shards == 1:
        print(json.dumps(manifest(output), indent=2), flush=True)


if __name__ == '__main__':
    main()
