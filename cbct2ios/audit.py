"""Label audit and patient-level splits of ToothFairy3.

audit      per case, labels only: tooth masks, field-of-view truncation, prosthetic contact, tooth centroids
aggregate  near-duplicate groups (tooth-centroid configurations), F/P/S-stratified 70/10/20 split
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from .geometry import family, transform_points
from .io import write_csv, write_json
from .simulate import BONE, BRIDGE, CROWN, IMPLANT, TEETH, bounding_boxes, present, tooth_mask


SEED = 20261103
MIN_JAW_TEETH = 6
MIN_VOXELS = 2000          # smaller tooth labels are fragments or root remnants
MIN_MAIN_FRACTION = .9
MIN_BORDER = 2             # voxels between the tooth and the volume border (FOV truncation)
DUPLICATE_RMS = .2         # mm, tooth-centroid configuration match; on ToothFairy3 the 4 identical pairs sit near 0,
                           # different patients start at ~0.4 mm (8 shared teeth), and nothing lies in between


def case_name(label_path):
    """labelsTr/ToothFairy3S_0000.nii.gz -> S_0000 (label files carry no channel suffix)."""
    return Path(label_path).name[len('ToothFairy3'):-len('.nii.gz')]


def subset_of(case):
    return case[0]


def load_labels(root, case):
    image = nib.load(Path(root) / 'labelsTr' / f'ToothFairy3{case}.nii.gz')
    return np.asarray(image.dataobj).astype(np.int16), image.affine


def audit_case(root, case):
    seg, affine = load_labels(root, case)
    boxes = bounding_boxes(seg)
    metal = np.isin(seg, (BRIDGE, CROWN, IMPLANT))
    out = dict(case_id=case, subset=subset_of(case), shape=list(seg.shape),
               zooms=[float(z) for z in np.linalg.norm(affine[:3, :3], axis=0)], teeth={}, jaws={})
    for jaw, fdis in TEETH.items():
        present_teeth = []
        for fdi in fdis:
            if not (present(boxes, fdi) or present(boxes, fdi + 100)):
                continue
            info = tooth_mask(seg, fdi, boxes)
            mask, start = info.pop('mask'), info.pop('start')
            region = tuple(slice(int(a), int(a + s)) for a, s in zip(start, mask.shape))
            near_metal = bool((ndimage.binary_dilation(mask, iterations=2) & metal[region]).any())
            centroid = transform_points((np.argwhere(mask).mean(0) + start)[None], affine)[0]
            reasons = [name for name, bad in (('fragment', info['voxels'] < MIN_VOXELS),
                                              ('split_label', info['main_fraction'] < MIN_MAIN_FRACTION),
                                              ('fov_truncated', info['border_voxels'] < MIN_BORDER),
                                              ('prosthetic_contact', near_metal)) if bad]
            out['teeth'][str(fdi)] = dict(info, jaw=jaw, centroid=centroid.tolist(), near_metal=near_metal,
                                          audit_exclusion=';'.join(reasons))
            present_teeth.append(fdi)
        bone = int((seg == BONE[jaw]).sum())
        out['jaws'][jaw] = dict(teeth=len(present_teeth), bone_voxels=bone,
                                eligible=bool(bone > 0 and len(present_teeth) >= MIN_JAW_TEETH))
    out['metal_voxels'] = {str(k): int((seg == k).sum()) for k in (BRIDGE, CROWN, IMPLANT)}
    return out


def kabsch_rms(a, b):
    """Rigid RMS between corresponding point sets, reflections allowed (headers may flip an axis)."""
    a, b = a - a.mean(0), b - b.mean(0)
    best = np.inf
    for flip in (1, -1):
        bb = b * np.array([1, 1, flip])
        u, _, vt = np.linalg.svd(a.T @ bb)
        d = np.sign(np.linalg.det(u @ vt))
        rotation = u @ np.diag([1, 1, d]) @ vt
        best = min(best, float(np.sqrt(((a @ rotation - bb) ** 2).sum(1).mean())))
    return best


def duplicate_pairs(audits, min_common=8):
    centroids = {a['case_id']: {int(f): np.array(t['centroid']) for f, t in a['teeth'].items()} for a in audits}
    cases = sorted(centroids)
    pairs = []
    for i, x in enumerate(cases):
        for y in cases[i + 1:]:
            common = sorted(set(centroids[x]) & set(centroids[y]))
            if len(common) < min_common:
                continue
            rms = kabsch_rms(np.array([centroids[x][f] for f in common]), np.array([centroids[y][f] for f in common]))
            if rms < DUPLICATE_RMS * 3:
                pairs.append(dict(case_a=x, case_b=y, common_teeth=len(common), rms_mm=rms, duplicate=rms < DUPLICATE_RMS))
    return pairs


def groups_from_pairs(cases, pairs):
    parent = {c: c for c in cases}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for p in pairs:
        if p['duplicate']:
            parent[find(p['case_a'])] = find(p['case_b'])
    return {c: find(c) for c in cases}


def assign_splits(cases, groups, fractions=(('train', .7), ('val', .1), ('test', .2)), seed=SEED):
    """Group-level split, stratified by subset (F/P/S)."""
    rng = np.random.default_rng(seed)
    split = {}
    by_subset = defaultdict(list)
    for group in sorted(set(groups.values())):
        by_subset[subset_of(group)].append(group)
    for subset, members in sorted(by_subset.items()):
        members = list(rng.permutation(members))
        n = len(members)
        n_test = int(round(n * fractions[2][1]))
        n_val = int(round(n * fractions[1][1]))
        for k, group in enumerate(members):
            split[group] = 'test' if k < n_test else 'val' if k < n_test + n_val else 'train'
    return {c: split[groups[c]] for c in cases}


def aggregate(audit_dir):
    audit_dir = Path(audit_dir)
    audits = [json.loads(p.read_text()) for p in sorted((audit_dir / 'audit').glob('*.json'))]
    cases = [a['case_id'] for a in audits]
    pairs = duplicate_pairs(audits)
    groups = groups_from_pairs(cases, pairs)
    splits = assign_splits(cases, groups)
    case_rows, tooth_rows = [], []
    for a in audits:
        c = a['case_id']
        case_rows.append(dict(case_id=c, subset=a['subset'], split=splits[c], source_group_id=groups[c],
                              upper_teeth=a['jaws']['upper']['teeth'], lower_teeth=a['jaws']['lower']['teeth'],
                              upper_eligible=a['jaws']['upper']['eligible'], lower_eligible=a['jaws']['lower']['eligible'],
                              **{f'metal_{k}': v for k, v in a['metal_voxels'].items()}))
        for fdi, t in a['teeth'].items():
            exclusion = t['audit_exclusion'] or ('' if a['jaws'][t['jaw']]['eligible'] else 'jaw_ineligible')
            tooth_rows.append(dict(case_id=c, subset=a['subset'], split=splits[c], fdi=int(fdi), jaw=t['jaw'],
                                   family=family(int(fdi)), voxels=t['voxels'], pulp_voxels=t['pulp_voxels'],
                                   border_voxels=t['border_voxels'], main_fraction=t['main_fraction'],
                                   audit_exclusion=exclusion))
    write_csv(audit_dir / 'cases.csv', case_rows)
    write_csv(audit_dir / 'teeth_audit.csv', tooth_rows)
    write_csv(audit_dir / 'duplicate_pairs.csv', pairs)
    kept = [t for t in tooth_rows if not t['audit_exclusion']]
    summary = dict(
        cases=len(cases), by_subset=dict(Counter(a['subset'] for a in audits)),
        groups=len(set(groups.values())), duplicate_pairs=sum(p['duplicate'] for p in pairs),
        cases_by_split={s: dict(Counter(r['subset'] for r in case_rows if r['split'] == s)) for s in ('train', 'val', 'test')},
        teeth_labelled=len(tooth_rows), teeth_after_audit=len(kept),
        exclusions=dict(Counter(e for t in tooth_rows for e in t['audit_exclusion'].split(';') if e)),
        teeth_after_audit_by_split=dict(Counter(t['split'] for t in kept)),
        teeth_after_audit_by_family=dict(Counter(t['family'] for t in kept)),
        eligible_jaws=dict(Counter(f"{j}:{a['jaws'][j]['eligible']}" for a in audits for j in a['jaws'])))
    write_json(audit_dir / 'summary.json', summary)
    return summary
