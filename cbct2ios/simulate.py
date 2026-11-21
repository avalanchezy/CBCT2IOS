"""Simulated intraoral scan (IOS) of one jaw from ToothFairy3 labels.

Tooth surfaces come from the tooth and pulp labels. Around each tooth the alveolar bone crest is measured per
angular sector; the gingival margin lies a fixed height above it, higher between teeth (the papilla, limited by
the contact point). A soft-tissue envelope (collars plus mucosa over the alveolar bone) gives the gingiva
surface of the simulated scan. Supragingival faces are visible when a narrow cone towards one of the scanner
directions is free of teeth and gingiva of the same jaw (the opposing jaw is not in the scene) and the face is
not closer than a minimum gap to a neighbour. Every face gets an occlusion cause; the visible faces are the
simulated IOS crown and everything else is the unobserved surface (what a crown-to-root completion method has
to produce).
"""
from __future__ import annotations

import math

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import trimesh

from .geometry import WORKERS, arch_neighbors, surface_samples, transform_points


TEETH = {'upper': tuple(q * 10 + t for q in (1, 2) for t in range(1, 9)),
         'lower': tuple(q * 10 + t for q in (3, 4) for t in range(1, 9))}
BONE = {'upper': 2, 'lower': 1}
BRIDGE, CROWN, IMPLANT = 8, 9, 10
VISIBLE, IN_BONE, GINGIVA, PROXIMAL, OTHER = range(5)
CAUSES = ('visible', 'bone', 'gingiva', 'proximal', 'other')
SECTORS = 24
SECTOR_ANGLES = -np.pi + (np.arange(SECTORS) + .5) * 2 * np.pi / SECTORS
PARAMS = dict(
    # Kois "normal crest": facial margin 3 mm, papilla 3-4.5 mm above the crest. h_f = 3.0 is also the closest
    # setting to the visible fractions of real paired IOS crowns within 1.5-3.0 mm (cbct2ios/calibrate.py).
    h_f=3.0,           # buccal/lingual gingival margin above the bone crest (mm)
    h_p=4.5,           # maximum papilla height above the interdental crest (mm)
    g_min=.4,          # supragingival faces closer than this to a neighbour are proximal (mm)
    alpha=5.,          # visibility cone half angle (deg)
    incidence=70.,     # maximum angle between face normal and viewing direction (deg)
    reach=20.,         # occluders farther than this along a ray are ignored (mm)
    contact=.45,       # tooth surface within this distance of the bone label is in bone contact (mm)
    mucosa=1.5,        # soft tissue over the alveolar bone (mm)
    collar=1.5,        # maximum free-gingiva thickness on buccal/lingual sides (mm)
    embrasure=3.,      # maximum soft-tissue reach between teeth (mm)
    band=8.,           # scanned gingiva within this distance of the teeth (mm)
    smooth=.3,         # Gaussian smoothing of the soft-tissue envelope (mm)
    island=1.,         # visible patches smaller than this are dropped (mm^2)
    papilla_gap=1.5,   # no papilla when neighbours are farther apart than this (mm)
    min_exposed=1.5,   # the margin stays at least this far below the crown top (mm)
)
SURFACE_SIGMA = .5     # Gaussian smoothing of tooth masks before marching cubes (voxels)
VOXEL_MM = .3


# ----------------------------------------------------------------------------------------------- volumes

def bounding_boxes(seg):
    return ndimage.find_objects(seg.astype(np.int32), max_label=148)


def present(boxes, label):
    return label <= len(boxes) and boxes[label - 1] is not None


def label_mask(seg, labels, boxes, pad=2):
    """Union of labels as (mask, start) in a padded crop around them; (None, None) when absent."""
    found = [boxes[label - 1] for label in labels if present(boxes, label)]
    if not found:
        return None, None
    low = np.maximum(np.min([[s.start for s in box] for box in found], 0) - pad, 0)
    high = np.minimum(np.max([[s.stop for s in box] for box in found], 0) + pad, seg.shape)
    crop = seg[tuple(slice(int(a), int(b)) for a, b in zip(low, high))]
    return np.isin(crop, labels), low


def tooth_mask(seg, fdi, boxes):
    """Filled tooth-plus-pulp mask (largest 26-connected part) and label statistics."""
    mask, start = label_mask(seg, (fdi, fdi + 100), boxes)
    if mask is None:
        return None
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel())[1:]
    keep = labels == sizes.argmax() + 1
    filled = ndimage.binary_fill_holes(keep)
    voxels = np.argwhere(keep)
    low, high = voxels.min(0) + start, voxels.max(0) + start
    crop = seg[tuple(slice(int(a), int(a + s)) for a, s in zip(start, mask.shape))]
    return dict(mask=filled, start=start, voxels=int(sizes.sum()), components=int(count),
                main_fraction=float(sizes.max() / sizes.sum()), pulp_voxels=int((crop == fdi + 100).sum()),
                tooth_voxels=int((crop == fdi).sum()),
                border_voxels=int(min(low.min(), (np.array(seg.shape) - 1 - high).min())),
                filled_voxels=int(filled.sum() - keep.sum()))


def mask_surface(mask, start, affine, sigma=SURFACE_SIGMA):
    """Largest marching-cubes surface of a Gaussian-smoothed mask in world coordinates (outward normals),
    and the smoothed field with its voxel origin (for inside tests)."""
    field = ndimage.gaussian_filter(np.pad(mask, 3).astype(np.float32), sigma)
    origin = np.asarray(start) - 3
    vertices, faces, _, _ = marching_cubes(field, .5)
    mesh = trimesh.Trimesh(transform_points(vertices + origin, affine), faces, process=False)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda p: p.area)
    if mesh.volume < 0:
        mesh.faces = mesh.faces[:, ::-1]
    return mesh, field, origin


def world_to_voxel(points, affine, start=0):
    return transform_points(points, np.linalg.inv(affine)) - start


# ----------------------------------------------------------------------------------------------- profiles

def circular_interp(profile, angles):
    """Linear interpolation of a per-sector profile at angles (radians)."""
    position = (np.asarray(angles) + np.pi) / (2 * np.pi) * SECTORS - .5
    low = np.floor(position).astype(int)
    weight = position - low
    return profile[low % SECTORS] * (1 - weight) + profile[(low + 1) % SECTORS] * weight


def fill_circular(profile):
    ok = np.isfinite(profile)
    if not ok.any():
        return None
    index = np.arange(SECTORS)
    known = index[ok]
    return np.interp(index, np.concatenate([known - SECTORS, known, known + SECTORS]), np.tile(profile[ok], 3))


def sector_of(theta):
    return ((np.asarray(theta) + np.pi) / (2 * np.pi) * SECTORS).astype(int) % SECTORS


class ToothGeometry:
    """Long axis (towards the crown), simulator directions (buccal, mesial) and cylindrical coordinates."""

    def __init__(self, fdi, mesh, bone_distance):
        self.fdi, self.mesh = fdi, mesh
        vertices = np.asarray(mesh.vertices)
        self.center = vertices.mean(0)
        axis = np.linalg.eigh(np.cov((vertices - self.center).T))[1][:, -1]
        h = (vertices - self.center) @ axis
        self.contact = bone_distance <= PARAMS['contact']
        self.contact_fraction = float(self.contact.mean())
        top, bottom = h > np.percentile(h, 75), h < np.percentile(h, 25)
        # The crown is the end with less alveolar bone contact.
        self.axis = -axis if self.contact[top].mean() > self.contact[bottom].mean() else axis
        self.buccal = self.mesial = None

    def set_directions(self, buccal, mesial):
        a = self.axis
        b = buccal - (buccal @ a) * a
        self.buccal = b / np.linalg.norm(b)
        m = mesial - (mesial @ a) * a - (mesial @ self.buccal) * self.buccal
        self.mesial = m / np.linalg.norm(m) if np.linalg.norm(m) > 1e-6 else np.cross(a, self.buccal)

    def cylindrical(self, points):
        r = np.asarray(points) - self.center
        return r @ self.axis, np.arctan2(r @ self.mesial, r @ self.buccal)

    def top(self):
        return float(((np.asarray(self.mesh.vertices) - self.center) @ self.axis).max())


def crest_profile(geometry, *, impute_drop=3.):
    """98th percentile height of bone-contact vertices per sector; a missing thin plate is imputed."""
    h, theta = geometry.cylindrical(geometry.mesh.vertices)
    sector = sector_of(theta)
    profile = np.full(SECTORS, np.nan)
    for s in range(SECTORS):
        selected = geometry.contact & (sector == s)
        if selected.sum() >= 5:
            profile[s] = np.percentile(h[selected], 98)
    filled = fill_circular(profile)
    info = dict(crest_sectors=int(np.isfinite(profile).sum()), buccal_imputed=False, lingual_imputed=False)
    if filled is None:
        return None, info
    filled = np.array([np.median(filled[[s - 1, s, (s + 1) % SECTORS]]) for s in range(SECTORS)])
    buccal, lingual = np.abs(SECTOR_ANGLES) < np.radians(35), np.abs(SECTOR_ANGLES) > np.radians(145)
    proximal = np.abs(np.abs(SECTOR_ANGLES) - np.pi / 2) < np.radians(35)
    reference = float(np.median(filled[proximal]))
    for name, side, other in (('buccal', buccal, lingual), ('lingual', lingual, buccal)):
        if filled[side].mean() < min(filled[other].mean(), reference) - impute_drop:
            # Thin plates are often missing from CBCT labels; follow the interproximal crest instead.
            filled[side] = reference - 1.
            info[f'{name}_imputed'] = True
    return filled, info


def papilla_heights(geometry, crest, neighbours, params):
    """Papilla height above the interdental crest per side: up to the contact point, at most h_p."""
    heights, contacts = {}, {}
    vertices = np.asarray(geometry.mesh.vertices)
    for side, tree in neighbours.items():
        if tree is None:
            heights[side], contacts[side] = params['h_f'], None
            continue
        distance = tree.query(vertices, workers=WORKERS)[0]
        k = int(distance.argmin())
        h, theta = geometry.cylindrical(vertices[k:k + 1])
        above = float(h[0] - circular_interp(crest, theta)[0])
        contacts[side] = dict(gap_mm=float(distance[k]), contact_above_crest_mm=above)
        heights[side] = float(np.clip(above, params['h_f'], params['h_p'])) if distance[k] <= params['papilla_gap'] \
            else params['h_f']
    return heights, contacts


def margin_profile(crest, papilla, h_f):
    """Gingival margin = crest + h_f, raised towards the papilla heights on the mesial and distal sides."""
    mesial = np.clip(np.cos(SECTOR_ANGLES - np.pi / 2), 0, None) ** 4
    distal = np.clip(np.cos(SECTOR_ANGLES + np.pi / 2), 0, None) ** 4
    return crest + h_f + (papilla['mesial'] - h_f) * mesial + (papilla['distal'] - h_f) * distal


# ----------------------------------------------------------------------------------------------- scene

def soft_tissue(geometries, profiles, owners, distance, bone_distance, crop_start, affine, params):
    """Gingiva envelope on the jaw crop: free-gingiva collars, papillae, alveolar bone and its mucosa."""
    limit = max(params['embrasure'], params['mucosa']) + params['band']
    index = np.argwhere((distance > 0) & (distance <= limit))
    world = transform_points(index + crop_start, affine)
    owner = owners[tuple(index.T)]
    d = distance[tuple(index.T)]
    collar = np.zeros(len(index), bool)
    below_margin = np.ones(len(index), bool)
    for k, (geometry, profile) in enumerate(zip(geometries, profiles)):
        selected = np.flatnonzero(owner == k)
        if profile is None or not len(selected):
            continue
        h, theta = geometry.cylindrical(world[selected])
        margin = circular_interp(profile['margin'], theta)
        crest = circular_interp(profile['crest'], theta)
        reach = params['collar'] + (params['embrasure'] - params['collar']) * np.abs(np.sin(theta)) ** 4
        thickness = np.minimum(reach, .4 + .5 * (margin - h))
        collar[selected] = (h <= margin) & (h >= crest - 3.) & (d[selected] <= thickness)
        below_margin[selected] = h <= margin
    mucosa = (bone_distance[tuple(index.T)] <= params['mucosa']) & below_margin
    tissue = distance == 0
    tissue[tuple(index[collar | mucosa].T)] = True
    tissue |= (bone_distance == 0) & (distance <= params['band'] + 2.)
    return tissue


def envelope_surface(volume, crop_start, affine, sigma_voxels):
    field = ndimage.gaussian_filter(np.pad(volume, 2).astype(np.float32), sigma_voxels)
    vertices, faces, _, _ = marching_cubes(field, .5)
    vertices = vertices - 2
    mesh = trimesh.Trimesh(transform_points(vertices + crop_start, affine), faces, process=False)
    if mesh.volume < 0:
        mesh.faces = mesh.faces[:, ::-1]
    return mesh, vertices


def cone(direction, alpha, count=8):
    helper = np.array([1., 0, 0]) if abs(direction[0]) < .9 else np.array([0, 1., 0])
    a = np.cross(direction, helper)
    a /= np.linalg.norm(a)
    b = np.cross(direction, a)
    phi = 2 * np.pi * np.arange(count) / count
    t = math.radians(alpha)
    return math.cos(t) * direction + math.sin(t) * (np.cos(phi)[:, None] * a + np.sin(phi)[:, None] * b)


def scanner_directions(occlusal, buccal):
    """Occlusal, tilted, buccal/lingual and slightly apical views around one tooth (jaw occlusal axis)."""
    buccal = buccal - (buccal @ occlusal) * occlusal
    buccal = buccal / np.linalg.norm(buccal)
    mesial = np.cross(occlusal, buccal)
    out = [occlusal]
    for elevation, azimuths in ((60, range(0, 360, 45)), (30, (0, 30, 60, -30, -60, 180, 150, 210, 120, 240)),
                                (0, (0, 30, -30, 180, 150, 210)), (-15, (0, 180))):
        e = math.radians(elevation)
        for azimuth in azimuths:
            a = math.radians(azimuth)
            out.append(math.cos(e) * (math.cos(a) * buccal + math.sin(a) * mesial) + math.sin(e) * occlusal)
    out = np.array(out)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


class Scene:
    def __init__(self, meshes):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.ids = [self.scene.add_triangles(o3d.core.Tensor(np.asarray(m.vertices, np.float32)),
                                             o3d.core.Tensor(np.asarray(m.faces, np.uint32))) for m in meshes]

    def rays(self, origins, direction):
        return o3d.core.Tensor(np.hstack([origins, np.broadcast_to(direction, origins.shape)]).astype(np.float32))

    def occluded(self, origins, direction, reach):
        return self.scene.test_occlusions(self.rays(origins, direction), tnear=0., tfar=reach).numpy().astype(bool)

    def first_hit(self, origins, direction, reach):
        out = self.scene.cast_rays(self.rays(origins, direction))
        geometry = out['geometry_ids'].numpy().astype(np.int64)
        geometry[(out['t_hit'].numpy() > reach) | (geometry == o3d.t.geometry.RaycastingScene.INVALID_ID)] = -1
        return geometry


def visible_faces(scene, mesh, candidates, directions, params):
    """Faces whose whole viewing cone towards at least one scanner direction is free."""
    normals = np.asarray(mesh.face_normals)
    origins = np.asarray(mesh.triangles_center) + .03 * normals
    visible = np.zeros(len(normals), bool)
    limit = math.cos(math.radians(params['incidence']))
    for direction in directions:
        todo = np.flatnonzero(candidates & ~visible & (normals @ direction >= limit))
        for ray in np.vstack([direction[None], cone(direction, params['alpha'])]):
            if not len(todo):
                break
            todo = todo[~scene.occluded(origins[todo], ray, params['reach'])]
        visible[todo] = True
    return visible


def blocked_by_teeth(scene, mesh, faces, directions, params, others):
    """For faces seen from nowhere: True when some admissible direction is first blocked by another tooth."""
    normals = np.asarray(mesh.face_normals)[faces]
    origins = np.asarray(mesh.triangles_center)[faces] + .03 * normals
    out = np.zeros(len(faces), bool)
    limit = math.cos(math.radians(params['incidence']))
    for direction in directions:
        valid = np.flatnonzero(normals @ direction >= limit)
        if len(valid):
            out[valid[np.isin(scene.first_hit(origins[valid], direction, params['reach']), others)]] = True
    return out


def drop_islands(mesh, keep, minimum):
    """Keep face-connected visible patches of at least `minimum` mm^2."""
    faces = np.flatnonzero(keep)
    out = np.zeros_like(keep)
    if not len(faces):
        return out
    adjacency = mesh.face_adjacency
    both = keep[adjacency[:, 0]] & keep[adjacency[:, 1]]
    area = mesh.area_faces
    for component in trimesh.graph.connected_components(adjacency[both], nodes=faces, min_len=1):
        if area[component].sum() >= minimum:
            out[component] = True
    return out


def dense_tree(mesh):
    return cKDTree(surface_samples(mesh, .1, seed=0)[0])


def simulate_jaw(seg, affine, jaw, params=None):
    """Simulated IOS of one jaw.

    Returns per tooth: mesh, smoothed mask field (inside tests), face causes, visible mask, crest/margin
    profiles, simulator axes, statistics; for the jaw: gingiva mesh, prosthetic meshes with visible masks and
    the jaw occlusal axis. Every present tooth of the jaw is part of the scene and of the context.
    """
    params = dict(PARAMS, **(params or {}))
    boxes = bounding_boxes(seg)
    fdis = [f for f in TEETH[jaw] if present(boxes, f) or present(boxes, f + 100)]
    if len(fdis) < 2:
        return dict(status='too_few_teeth', teeth={})
    found = [boxes[f - 1] for f in fdis if present(boxes, f)] + [boxes[f + 99] for f in fdis if present(boxes, f + 100)]
    pad = int(math.ceil((params['band'] + 4.) / VOXEL_MM))
    crop_start = np.maximum(np.min([[s.start for s in b] for b in found], 0) - pad, 0)
    crop_stop = np.minimum(np.max([[s.stop for s in b] for b in found], 0) + pad, seg.shape)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(crop_start, crop_stop))
    seg_crop = seg[crop]
    bone = seg_crop == BONE[jaw]
    if not bone.any():
        return dict(status='no_bone_label', teeth={})
    bone_distance = ndimage.distance_transform_edt(~bone, sampling=VOXEL_MM)
    teeth, failed = {}, {}
    for fdi in fdis:
        info = tooth_mask(seg, fdi, boxes)
        if info is None or info['voxels'] < 200:
            failed[fdi] = 'too_small'
            continue
        mesh, field, origin = mask_surface(info['mask'], info['start'], affine)
        d = ndimage.map_coordinates(bone_distance, world_to_voxel(mesh.vertices, affine, crop_start).T,
                                    order=1, mode='nearest')
        teeth[fdi] = dict(info=info, mesh=mesh, field=field, field_origin=origin, geometry=ToothGeometry(fdi, mesh, d))
    anchored = [t['geometry'].axis for t in teeth.values() if t['geometry'].contact_fraction > .05]
    if len(teeth) < 2 or not anchored:
        return dict(status='no_bone_contact', teeth={}, failed=failed)
    jaw_axis = np.sum(anchored, 0)
    jaw_axis /= np.linalg.norm(jaw_axis)
    for t in teeth.values():
        g = t['geometry']
        if g.axis @ jaw_axis < 0:
            g.axis = -g.axis
        t['axis_angle_deg'] = float(np.degrees(np.arccos(np.clip(g.axis @ jaw_axis, -1, 1))))
    crowns = {f: t['geometry'].center + 6. * t['geometry'].axis for f, t in teeth.items()}
    arch_center = np.mean(list(crowns.values()), 0)
    crown_top = float(np.median([(crowns[f] - arch_center) @ jaw_axis + t['geometry'].top() - 6.
                                 for f, t in teeth.items()]))
    for fdi, t in teeth.items():
        mesial, distal = arch_neighbors(fdi, crowns)
        buccal = crowns[fdi] - arch_center
        toward = crowns[mesial] - crowns[fdi] if mesial else crowns[fdi] - crowns[distal]
        t['geometry'].set_directions(buccal - (buccal @ jaw_axis) * jaw_axis, toward)
        t['neighbours'] = dict(mesial=mesial, distal=distal)
    trees = {f: dense_tree(t['mesh']) for f, t in teeth.items()}
    # Crowns and bridges of this jaw join the scene (their centre lies below the occlusal level of this jaw);
    # implants stay inside bone and only flag the jaw as metal-bearing.
    prosthetics = []
    metal = False
    for label in (BRIDGE, CROWN, IMPLANT):
        mask, start = label_mask(seg, (label,), boxes)
        if mask is None:
            continue
        parts, count = ndimage.label(mask, structure=np.ones((3, 3, 3)))
        for k in range(1, count + 1):
            part = parts == k
            if part.sum() < 50:
                continue
            centre = transform_points((np.argwhere(part).mean(0) + start)[None], affine)[0]
            if min(float(trees[f].query(centre)[0]) for f in teeth) > 15.:
                continue
            if label == IMPLANT:
                voxel = np.round(world_to_voxel(centre[None], affine, crop_start)[0]).astype(int)
                if np.all((voxel >= 0) & (voxel < bone.shape)) and bone_distance[tuple(voxel)] <= 3.:
                    metal = True
                continue
            if (centre - arch_center) @ jaw_axis > crown_top + 1.:
                continue
            mesh, _, _ = mask_surface(part, start, affine)
            prosthetics.append(dict(label=label, mesh=mesh, mask=part, start=start))
            metal = True
    for fdi, t in teeth.items():
        g = t['geometry']
        crest, crest_info = crest_profile(g)
        t['crest_info'] = crest_info
        if crest is None:
            t['profile'] = None
            continue
        neighbours = {side: trees[other] if other else None for side, other in t['neighbours'].items()}
        papilla, contacts = papilla_heights(g, crest, neighbours, params)
        top = g.top()
        margin = np.minimum(margin_profile(crest, papilla, params['h_f']), top - params['min_exposed'])
        t['profile'] = dict(crest=crest, margin=margin, papilla=papilla, contacts=contacts, top=top)
    # Soft-tissue envelope on the jaw crop.
    order = list(teeth)
    owners_volume = np.zeros(seg_crop.shape, np.int32)
    for k, fdi in enumerate(order):
        info = teeth[fdi]['info']
        local = np.asarray(info['start']) - crop_start
        lo = np.maximum(local, 0)
        hi = np.minimum(local + info['mask'].shape, seg_crop.shape)
        sub = info['mask'][tuple(slice(int(a - b), int(c - b)) for a, b, c in zip(lo, local, hi))]
        view = owners_volume[tuple(slice(int(a), int(c)) for a, c in zip(lo, hi))]
        view[sub] = k + 1
    hard = owners_volume > 0
    for p in prosthetics:
        local = np.asarray(p['start']) - crop_start
        lo = np.maximum(local, 0)
        hi = np.minimum(local + p['mask'].shape, seg_crop.shape)
        sub = p['mask'][tuple(slice(int(a - b), int(c - b)) for a, b, c in zip(lo, local, hi))]
        hard[tuple(slice(int(a), int(c)) for a, c in zip(lo, hi))] |= sub
    distance, nearest = ndimage.distance_transform_edt(~hard, sampling=VOXEL_MM, return_indices=True)
    owners = owners_volume[tuple(nearest)] - 1
    del nearest
    tissue = soft_tissue([teeth[f]['geometry'] for f in order], [teeth[f]['profile'] for f in order], owners,
                         distance, bone_distance, crop_start, affine, params)
    envelope, local_vertices = envelope_surface(tissue, crop_start, affine, params['smooth'] / VOXEL_MM)
    centers = local_vertices[envelope.faces].mean(1)
    face_distance = ndimage.map_coordinates(distance, centers.T, order=1, mode='nearest')
    # Where the crop meets the volume border the envelope is cut flat by the field of view: not a scan surface.
    low_edge = np.where(crop_start == 0, 1.5, -np.inf)
    high_edge = np.where(crop_stop == np.array(seg.shape), np.array(seg_crop.shape) - 2.5, np.inf)
    in_view = np.all((centers > low_edge) & (centers < high_edge), 1)
    gingiva = envelope.copy()
    gingiva.update_faces((face_distance > .3) & (face_distance <= params['band']) & in_view)
    gingiva.remove_unreferenced_vertices()
    # Visibility in the single-jaw scene.
    scene = Scene([teeth[f]['mesh'] for f in order] + [p['mesh'] for p in prosthetics] + [gingiva])
    hard_ids = scene.ids[:len(order) + len(prosthetics)]
    for k, fdi in enumerate(order):
        t = teeth[fdi]
        mesh, g = t['mesh'], t['geometry']
        causes = np.full(len(mesh.faces), OTHER, np.uint8)
        t['visible'] = np.zeros(len(mesh.faces), bool)
        t['causes'] = causes
        if t['profile'] is None:
            continue
        h, theta = g.cylindrical(mesh.triangles_center)
        crest = circular_interp(t['profile']['crest'], theta)
        margin = circular_interp(t['profile']['margin'], theta)
        causes[h < crest] = IN_BONE
        causes[(h >= crest) & (h < margin)] = GINGIVA
        supragingival = h >= margin
        gap = np.full(len(mesh.faces), np.inf)
        for other in t['neighbours'].values():
            if other is not None:
                gap = np.minimum(gap, trees[other].query(mesh.triangles_center, workers=WORKERS)[0])
        for p in prosthetics:
            gap = np.minimum(gap, cKDTree(p['mesh'].vertices).query(mesh.triangles_center, workers=WORKERS)[0])
        tight = supragingival & (gap < params['g_min'])
        causes[tight] = PROXIMAL
        directions = scanner_directions(jaw_axis, g.buccal)
        candidates = supragingival & ~tight
        visible = drop_islands(mesh, visible_faces(scene, mesh, candidates, directions, params), params['island'])
        hidden = np.flatnonzero(candidates & ~visible)
        if len(hidden):
            neighbour = blocked_by_teeth(scene, mesh, hidden, directions, params, [i for i in hard_ids if i != scene.ids[k]])
            causes[hidden] = np.where(neighbour, PROXIMAL, OTHER)
        causes[visible] = VISIBLE
        t['visible'] = visible
        t['stats'] = tooth_stats(t, mesh, causes, params)
    for p in prosthetics:
        mesh = p['mesh']
        h = (np.asarray(mesh.triangles_center) - arch_center) @ jaw_axis
        centre = np.asarray(mesh.vertices).mean(0)
        directions = scanner_directions(jaw_axis, centre - arch_center)
        p['visible'] = visible_faces(scene, mesh, h > np.percentile(h, 30), directions, params)
    return dict(status='ok', teeth=teeth, failed=failed, prosthetics=prosthetics, gingiva=gingiva,
                jaw_axis=jaw_axis, arch_center=arch_center, order=order, metal=metal, params=params)


def tooth_stats(t, mesh, causes, params):
    area = mesh.area_faces
    total = float(area.sum())
    out = {f'{name}_area_mm2': float(area[causes == k].sum()) for k, name in enumerate(CAUSES)}
    profile = t['profile']
    out.update(total_area_mm2=total, visible_fraction=out['visible_area_mm2'] / max(total, 1e-9),
               exposed_height_mm=float(profile['top'] - np.median(profile['crest'])),
               crest_range_mm=float(np.ptp(profile['crest'])), papilla_mesial_mm=profile['papilla']['mesial'],
               papilla_distal_mm=profile['papilla']['distal'], axis_angle_deg=t['axis_angle_deg'],
               contact_fraction=t['geometry'].contact_fraction, **{k: v for k, v in t['crest_info'].items()})
    for side in ('mesial', 'distal'):
        contact = profile['contacts'].get(side)
        out[f'{side}_gap_mm'] = contact['gap_mm'] if contact else float('nan')
    return out

