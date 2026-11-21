"""README figures from the output of `python -m cbct2ios simulate` on one case (default: F_009, lower jaw, tooth 46).

    python -m cbct2ios simulate --root TF3 --cases F_009 --output-dir out/default
    python -m cbct2ios simulate --root TF3 --cases F_009 --output-dir out/raw --no-smooth-teeth --no-densify-ios
    python scripts/make_figures.py --root TF3 --default out/default --raw out/raw --figures docs

Needs matplotlib in addition to requirements.txt. Meshes are drawn with a small orthographic painter's renderer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
import trimesh  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbct2ios.geometry import transform_points  # noqa: E402
from cbct2ios.pipeline import CAUSE_HEX  # noqa: E402

BG, INK, INK2 = '#F7F6F2', '#1B2430', '#4A5563'
GUM, OTHER_CROWN, TOOTH = '#E7A3B5', '#B9C7DA', '#E6DFD0'
CAUSE_LABELS = ('visible: the simulated scan of the crown', 'in bone', 'under the gingiva',
                'proximal: contact with or hidden by a neighbour', 'other hidden')


def style(size=12.5):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Segoe UI', 'Arial', 'DejaVu Sans'],
                         'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG, 'text.color': INK,
                         'font.size': size, 'axes.titlesize': size + 1, 'axes.titleweight': 'semibold',
                         'legend.frameon': False})


def rgb(color):
    color = color.lstrip('#')
    return np.array([int(color[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def frame(toward, up):
    """Rows: screen right, screen up, toward the viewer."""
    toward = unit(toward)
    right = unit(np.cross(up, toward))
    return np.stack([right, np.cross(toward, right), toward])


def render(ax, items, rotation, *, pad=.6, light=(.35, .5, .8), edge=None, edge_width=.15):
    """Painter's algorithm. items: dicts(mesh, color=hex | [F, 3] array, faces=optional face mask)."""
    polys, colors, depth = [], [], []
    light = unit(light)
    for item in items:
        mesh = item['mesh']
        faces = np.asarray(mesh.faces)
        color = item['color']
        if 'faces' in item:
            faces = faces[item['faces']]
            color = color if isinstance(color, str) else np.asarray(color)[item['faces']]
        if not len(faces):
            continue
        tri = (np.asarray(mesh.vertices) @ rotation.T)[faces]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        shade = .42 + .58 * np.abs(normal @ light)
        base = np.broadcast_to(rgb(color), (len(faces), 3)) if isinstance(color, str) else np.asarray(color)
        polys.append(tri[:, :, :2])
        colors.append(np.clip(base * shade[:, None], 0, 1))
        depth.append(tri[:, :, 2].mean(1))
    polys, colors, depth = np.concatenate(polys), np.concatenate(colors), np.concatenate(depth)
    order = np.argsort(depth)
    ax.add_collection(PolyCollection(polys[order], facecolors=colors[order],
                                     edgecolors=colors[order] if edge is None else edge, linewidths=edge_width))
    low, high = polys.reshape(-1, 2).min(0), polys.reshape(-1, 2).max(0)
    ax.set_xlim(low[0] - pad, high[0] + pad)
    ax.set_ylim(low[1] - pad, high[1] + pad)
    ax.set_aspect('equal')
    ax.axis('off')


def load(path):
    return trimesh.load(path, process=False)


class Jaw:
    """One simulated jaw as written by `simulate`."""

    def __init__(self, folder):
        self.folder = Path(folder)
        info = json.loads((self.folder / 'jaw.json').read_text())
        self.up, self.center = np.array(info['jaw_axis']), np.array(info['arch_center'])
        self.teeth = {int(p.stem): load(p) for p in sorted((self.folder / 'teeth').glob('*.ply'))}
        self.crowns = {int(p.stem): load(p) for p in sorted((self.folder / 'crowns').glob('*.ply'))}
        self.gingiva = load(self.folder / 'gingiva.ply')
        faces = np.load(self.folder / 'faces.npz')
        self.visible = {f: faces[f'{f}_visible'] for f in self.teeth}
        self.cause = {f: faces[f'{f}_cause'] for f in self.teeth}

    def flat(self, v):
        return unit(v - (v @ self.up) * self.up)

    def centroid(self, fdi):
        return np.asarray(self.teeth[fdi].vertices).mean(0)

    def directions(self, fdi):
        """Mesial (along the arch, from the distal to the mesial neighbour) and buccal (outward) of a tooth."""
        quadrant, position = divmod(fdi, 10)
        neighbours = [quadrant * 10 + p for p in (position - 1, position + 1) if 1 <= p <= 8 and quadrant * 10 + p in self.teeth]
        if position == 1:
            other = {1: 21, 2: 11, 3: 41, 4: 31}[quadrant]
            neighbours = [n for n in [other] if n in self.teeth] + neighbours
        mesial_side, distal_side = neighbours[0], neighbours[-1]
        mesial = self.flat(self.centroid(mesial_side) - self.centroid(distal_side))
        buccal = unit(np.cross(self.up, mesial))
        if buccal @ (self.centroid(fdi) - self.center) < 0:
            buccal = -buccal
        return mesial, buccal

    def oblique(self, fdi, front=.62, side=.6, above=.5):
        incisors = [f for f in self.teeth if f % 10 in (1, 2)]
        molars = [f for f in self.teeth if f % 10 in (6, 7)]
        ahead = self.flat(np.mean([self.centroid(f) for f in incisors], 0) - np.mean([self.centroid(f) for f in molars], 0))
        lateral = self.flat(self.centroid(fdi) - self.center)
        lateral = unit(lateral - (lateral @ ahead) * ahead)
        return frame(front * ahead + side * lateral + above * self.up, self.up)

    def cause_colors(self, fdi):
        return np.array([rgb(c) for c in CAUSE_HEX])[self.cause[fdi]]

    def area_fraction(self, fdi):
        area = self.teeth[fdi].area_faces
        return {k: float(area[self.cause[fdi] == k].sum() / area.sum()) for k in range(len(CAUSE_HEX))}


def plane_image(volume, affine, origin, u, w, us, ws):
    uu, ww = np.meshgrid(us, ws)
    world = origin + uu[..., None] * u + ww[..., None] * w
    voxel = transform_points(world.reshape(-1, 3), np.linalg.inv(affine)).T
    return ndimage.map_coordinates(volume, voxel, order=1, mode='constant', cval=float(volume.min())).reshape(uu.shape)


def section(mesh, origin, normal, u, w):
    """Intersection of a mesh with a plane, as 2-D segments in (u, w) and the face index of each segment."""
    lines, faces = trimesh.intersections.mesh_plane(mesh, normal, origin, return_faces=True)
    rel = lines - origin
    return np.stack([rel @ u, rel @ w], -1), faces


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=.1)
    plt.close(fig)
    print('wrote', path, f'{Path(path).stat().st_size / 1e6:.2f} MB')


def legend(fig, extra=(), causes=(0, 1, 2, 3), y=0.):
    handles = [Patch(color=CAUSE_HEX[k], label=CAUSE_LABELS[k]) for k in causes] + [Patch(color=c, label=l) for c, l in extra]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=10.5, bbox_to_anchor=(.5, y),
               columnspacing=1.4, handlelength=1.6)


def fig_overview(jaw, ct, affine, fdi, path):
    """The CBCT section with the simulated gingiva and the visible crown, the simulated scan, the tooth by cause."""
    style()
    fig = plt.figure(figsize=(17.5, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=(.95, 1.75, .8), wspace=.06, left=.01, right=.99, top=.9, bottom=.12)
    ax = fig.add_subplot(gs[0])
    mesial, buccal = jaw.directions(fdi)
    origin = jaw.centroid(fdi)
    us, ws = np.arange(-11, 11.01, .1), np.arange(-14.5, 12.01, .1)
    ax.imshow(plane_image(ct, affine, origin, buccal, jaw.up, us, ws), origin='lower', extent=(us[0], us[-1], ws[0], ws[-1]),
              cmap='gray', vmin=-600, vmax=3000)
    segments, faces = section(jaw.teeth[fdi], origin, mesial, buccal, jaw.up)
    ax.add_collection(LineCollection(segments, colors=[CAUSE_HEX[c] for c in jaw.cause[fdi][faces]], linewidths=2.6,
                                     capstyle='round'))
    segments, _ = section(jaw.gingiva, origin, mesial, buccal, jaw.up)
    ax.add_collection(LineCollection(segments, colors=GUM, linewidths=2.2, linestyles=(0, (2, 1.2))))
    ax.set_xlim(us[0], us[-1])
    ax.set_ylim(ws[0], ws[-1])
    ax.text(.03, .97, '← lingual      buccal →', transform=ax.transAxes, color='white', fontsize=10.5, va='top')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'CBCT section through tooth {fdi}', loc='left')
    ax = fig.add_subplot(gs[1])
    items = [dict(mesh=jaw.gingiva, color=GUM)] + [dict(mesh=m, color=CAUSE_HEX[0] if f == fdi else OTHER_CROWN)
                                                   for f, m in jaw.crowns.items()]
    render(ax, items, jaw.oblique(fdi))
    ax.set_title('Simulated intraoral scan of the lower jaw', loc='left')
    ax = fig.add_subplot(gs[2])
    render(ax, [dict(mesh=jaw.teeth[fdi], color=jaw.cause_colors(fdi))], frame(buccal, jaw.up))
    share = jaw.area_fraction(fdi)
    ax.set_title(f'Tooth {fdi}: {share[0] * 100:.0f}% of its surface is scanned', loc='left')
    legend(fig, extra=((GUM, 'gingiva of the scan (dashed: on the section)'), (OTHER_CROWN, 'other scanned crowns')))
    save(fig, path)


def fig_causes(jaw, fdi, path):
    """Every tooth of the jaw coloured by occlusion cause, from the buccal side and from above the tongue."""
    style()
    fig = plt.figure(figsize=(17.5, 6.4))
    gs = fig.add_gridspec(1, 2, wspace=.04, left=.01, right=.99, top=.9, bottom=.12)
    items = [dict(mesh=m, color=jaw.cause_colors(f)) for f, m in jaw.teeth.items()]
    quadrant = fdi // 10
    side = [dict(mesh=m, color=jaw.cause_colors(f)) for f, m in jaw.teeth.items() if f // 10 == quadrant]
    lateral = jaw.flat(jaw.centroid(fdi) - jaw.center)
    for k, (parts, view, title) in enumerate((
            (items, jaw.oblique(fdi), 'Whole jaw from the buccal side'),
            (side, frame(-lateral + .45 * jaw.up, jaw.up), f'Teeth {quadrant}1–{quadrant}8 from the lingual side'))):
        ax = fig.add_subplot(gs[k])
        render(ax, parts, view)
        ax.set_title(title, loc='left')
    fractions = np.array([list(jaw.area_fraction(f).values()) for f in jaw.teeth])
    median = np.median(fractions, 0)
    fig.suptitle('Occlusion cause of every tooth face, lower jaw: median share per tooth '
                 + ', '.join(f'{name.split(":")[0]} {median[k] * 100:.0f}%' for k, name in enumerate(CAUSE_LABELS[:4])),
                 x=.01, ha='left', fontsize=13.5, fontweight='semibold')
    legend(fig)
    save(fig, path)


def density(mesh):
    return len(mesh.vertices) / mesh.area


def fig_densify(raw, dense, fdi, path):
    """Label surface (0.3 mm voxel steps) against the default recipe, for the tooth and for its scanned crown."""
    style()
    fig = plt.figure(figsize=(17.5, 6.8))
    gs = fig.add_gridspec(1, 4, width_ratios=(.75, .75, 1, 1), wspace=.05, left=.01, right=.99, top=.86, bottom=.12)
    mesial, buccal = dense.directions(fdi)
    tooth_view = frame(buccal, dense.up)
    crown_view = frame(.55 * buccal + .8 * dense.up, mesial)
    panels = ((raw, 'tooth', 'Label surface', 'tooth'), (dense, 'tooth', 'Default recipe', 'tooth'),
              (raw, 'crown', 'Scanned crown, label surface', 'crown'), (dense, 'crown', 'Scanned crown, default recipe', 'crown'))
    for k, (jaw, kind, title, _) in enumerate(panels):
        ax = fig.add_subplot(gs[k])
        if kind == 'tooth':
            mesh = jaw.teeth[fdi]
            render(ax, [dict(mesh=mesh, color=jaw.cause_colors(fdi))], tooth_view)
        else:
            mesh = jaw.crowns[fdi]
            render(ax, [dict(mesh=mesh, color=CAUSE_HEX[0])], crown_view, pad=.3, edge='#0B2A55', edge_width=.25)
        ax.set_title(f'{title}\n{density(mesh):.0f} vertices/mm²', loc='left', fontsize=12.5)
    fig.suptitle(f'Tooth {fdi}: the voxel steps of the 0.3 mm labels, and the surfaces after smoothing (≤ 0.15 mm), '
                 'subdivision and resampling to scan density', x=.01, ha='left', fontsize=13.5, fontweight='semibold')
    legend(fig)
    save(fig, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True, help='ToothFairy3 folder (imagesTr for the CBCT section)')
    parser.add_argument('--default', type=Path, required=True, help='simulate output, default recipe')
    parser.add_argument('--raw', type=Path, help='simulate output with --no-smooth-teeth --no-densify-ios')
    parser.add_argument('--case', default='F_009')
    parser.add_argument('--jaw', default='lower')
    parser.add_argument('--tooth', type=int, default=46)
    parser.add_argument('--figures', type=Path, default=Path('docs'))
    args = parser.parse_args(argv)
    args.figures.mkdir(parents=True, exist_ok=True)
    dense = Jaw(args.default / 'cases' / args.case / args.jaw)
    image = nib.load(args.root / 'imagesTr' / f'ToothFairy3{args.case}_0000.nii.gz')
    fig_overview(dense, np.asarray(image.dataobj, np.float32), image.affine, args.tooth, args.figures / 'overview.png')
    fig_causes(dense, args.tooth, args.figures / 'causes.png')
    if args.raw:
        fig_densify(Jaw(args.raw / 'cases' / args.case / args.jaw), dense, args.tooth, args.figures / 'densify.png')


if __name__ == '__main__':
    main()
