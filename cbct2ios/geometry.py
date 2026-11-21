"""Mesh and point helpers shared by the simulation and the export: surface sampling, arch order, constrained
smoothing, Loop subdivision and resampling of open surfaces to the vertex density of real intraoral scans."""
from __future__ import annotations

import os

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
import trimesh


WORKERS = int(os.environ.get('CBCT2IOS_WORKERS', '4'))   # threads for KD-tree queries
IOS_DENSITY = 30.   # vertices per mm^2 of real IOS crowns (median 30.1, IQR 27.9-32.3; mean edge 0.205 mm)
TAUBIN = (.5, -.53)  # lambda / mu of the Taubin low-pass filter


def transform_points(points, transform):
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def family(fdi):
    position = int(fdi) % 10
    return 'incisor' if position <= 2 else 'canine' if position == 3 else 'premolar' if position <= 5 else 'molar'


def arch_neighbors(fdi, present):
    """Nearest present mesial and distal teeth along the FDI arch order."""
    quadrant, position = divmod(int(fdi), 10)
    other = {1: 2, 2: 1, 3: 4, 4: 3}[quadrant]
    order = [quadrant * 10 + p for p in range(8, 0, -1)] + [other * 10 + p for p in range(1, 9)]
    index = order.index(int(fdi))
    mesial = next((t for t in order[index + 1:] if t in present), None)
    distal = next((t for t in reversed(order[:index]) if t in present), None)
    return mesial, distal


def surface_samples(mesh, spacing=.1, seed=0, max_count=2_000_000):
    """Deterministic area-proportional surface points with face normals and face indices."""
    area = float(mesh.area)
    count = min(max_count, max(256, int(np.ceil(2 * area / spacing ** 2))))
    rng = np.random.default_rng(seed)
    triangles = np.asarray(mesh.triangles)
    areas = np.asarray(mesh.area_faces)
    face = rng.choice(len(areas), size=count, p=areas / areas.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = triangles[face]
    points = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    return points, np.asarray(mesh.face_normals)[face], face


def open_edge_points(mesh, spacing=.1):
    edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    points = []
    for a, b in np.asarray(mesh.vertices)[edges[counts == 1]]:
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        points.append(a + np.linspace(0., 1., n + 1)[:, None] * (b - a))
    return np.concatenate(points) if points else np.zeros((0, 3))


def mesh_edges(faces):
    """Unique edges, the unique-edge index of every face edge (3i + k is edge k of face i), faces per edge."""
    face_edges = np.sort(np.asarray(faces)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    unique, inverse, counts = np.unique(face_edges, axis=0, return_inverse=True, return_counts=True)
    return unique, inverse.reshape(-1), counts


def edge_adjacency(edges, count):
    rows, cols = np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(count, count))


def smoothing_operator(count, faces):
    """Uniform neighbour average: interior vertices over all neighbours, open-edge vertices over their two
    open-edge neighbours (the edge is smoothed as a curve); vertices on irregular or non-manifold edges stay."""
    unique, _, counts = mesh_edges(faces)
    full, open_edge = edge_adjacency(unique, count), edge_adjacency(unique[counts == 1], count)
    degree, open_degree = np.asarray(full.sum(1)).ravel(), np.asarray(open_edge.sum(1)).ravel()
    fixed = np.zeros(count, bool)
    fixed[unique[counts > 2].ravel()] = True
    interior = (open_degree == 0) & (degree > 0) & ~fixed
    curve = (open_degree == 2) & ~fixed
    return (sparse.diags(np.where(interior, 1. / np.maximum(degree, 1), 0.)) @ full
            + sparse.diags(np.where(curve, .5, 0.)) @ open_edge + sparse.diags((~(interior | curve)).astype(float)))


def loop_subdivide(vertices, faces):
    """One Loop step (each triangle into four) with the boundary rules of Hoppe et al. (1994): open edges follow
    the cubic B-spline curve; vertices with other than two open edges, or on non-manifold edges, stay put.
    Child k of face i is face k * F + i, so per-face labels are carried over with np.tile(labels, 4)."""
    vertices, faces = np.asarray(vertices, np.float64), np.asarray(faces, np.int64)
    count = len(vertices)
    unique, inverse, counts = mesh_edges(faces)
    odd = vertices[unique].mean(1)
    inner = counts == 2
    opposite = np.zeros((len(unique), 3))
    np.add.at(opposite, inverse, vertices[faces[:, [2, 0, 1]].reshape(-1)])
    odd[inner] = .375 * vertices[unique[inner]].sum(1) + .125 * opposite[inner]
    full, open_edge = edge_adjacency(unique, count), edge_adjacency(unique[counts == 1], count)
    k = np.maximum(np.asarray(full.sum(1)).ravel(), 1)
    beta = (.625 - (.375 + .25 * np.cos(2 * np.pi / k)) ** 2) / k
    even = (1 - k * beta)[:, None] * vertices + beta[:, None] * (full @ vertices)
    open_degree = np.asarray(open_edge.sum(1)).ravel()
    curve = open_degree == 2
    even[curve] = .75 * vertices[curve] + .125 * (open_edge @ vertices)[curve]
    fixed = (open_degree > 0) & ~curve
    fixed[unique[counts > 2].ravel()] = True
    even[fixed] = vertices[fixed]
    middle = count + inverse.reshape(-1, 3)  # new vertices on edges (0, 1), (1, 2), (2, 0) of each face
    faces = np.concatenate([np.column_stack([faces[:, 0], middle[:, 0], middle[:, 2]]),
                            np.column_stack([faces[:, 1], middle[:, 1], middle[:, 0]]),
                            np.column_stack([faces[:, 2], middle[:, 2], middle[:, 1]]), middle])
    return np.vstack([even, odd]), faces


def constrained_taubin(vertices, faces, band=.15, iterations=200):
    """Taubin low-pass smoothing with every vertex held within `band` mm of where it started (constrained smoothing
    as in Gibson's surface nets): voxel steps flatten out while the shape stays within half a voxel. Open edges are
    smoothed along themselves; vertices on non-manifold edges stay put."""
    start = np.asarray(vertices, np.float64)
    out = start.copy()
    operator = smoothing_operator(len(out), np.asarray(faces, np.int64))
    for step in range(iterations):
        out = out + TAUBIN[step % 2] * (operator @ out - out)
        offset = out - start
        out = start + offset * np.minimum(1., band / np.maximum(np.linalg.norm(offset, axis=1), 1e-12))[:, None]
    return out


def smooth_label_surface(mesh, band=.15, iterations=200):
    """A label-surface tooth (marching cubes of a 0.3 mm mask) without its voxel steps. Faces and their order are
    unchanged, so per-face visibility and occlusion causes stay valid."""
    return trimesh.Trimesh(constrained_taubin(mesh.vertices, mesh.faces, band, iterations), np.asarray(mesh.faces),
                           process=False)


def densify_open_mesh(mesh, density=IOS_DENSITY, smoothing=200, band=.15, relax=5):
    """IOS-like copy of an open surface extracted from a voxel label (a simulated IOS crown or gingiva piece).
    1. Taubin smoothing with every vertex held within `band` mm of where it started; the open edge is smoothed
       along itself.
    2. Loop subdivision until the mesh is denser than the target: a smooth, dense version of that surface.
    3. Quadric decimation to `density` vertices per mm^2, then tangential relaxation with the vertices projected
       back onto the dense surface and its open edge."""
    import open3d as o3d
    faces = np.asarray(mesh.faces, np.int64)
    vertices = constrained_taubin(mesh.vertices, faces, band, smoothing)
    dense = trimesh.Trimesh(vertices, faces, process=False)
    for _ in range(3):
        if len(dense.vertices) >= 1.5 * density * dense.area:
            break
        dense = trimesh.Trimesh(*loop_subdivide(dense.vertices, dense.faces), process=False)
    unique, _, counts = mesh_edges(dense.faces)
    rim_length = np.linalg.norm(np.diff(dense.vertices[unique[counts == 1]], axis=1), axis=2).sum()
    spacing = np.sqrt(2 / (np.sqrt(3) * density))  # edge length of an equilateral mesh with that vertex density
    target = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(dense.vertices), o3d.utility.Vector3iVector(dense.faces))
    # A disc-like mesh has F = 2V - (open-edge vertices) - 2 faces.
    target = target.simplify_quadric_decimation(int(2 * density * dense.area - rim_length / spacing), boundary_weight=10.)
    target.remove_degenerate_triangles()
    target.remove_duplicated_triangles()
    target.remove_unreferenced_vertices()
    vertices, faces = np.asarray(target.vertices), np.asarray(target.triangles, np.int64)
    points, normals, _ = surface_samples(dense, .03, seed=5)
    rim = open_edge_points(dense, .01)
    surface_tree, rim_tree = cKDTree(points), cKDTree(rim)
    unique, _, counts = mesh_edges(faces)
    on_rim = np.zeros(len(vertices), bool)
    on_rim[unique[counts == 1].ravel()] = True
    operator = smoothing_operator(len(vertices), faces)
    for _ in range(relax):
        normal = trimesh.Trimesh(vertices, faces, process=False).vertex_normals
        move = operator @ vertices - vertices
        move[~on_rim] -= np.einsum('ij,ij->i', move[~on_rim], normal[~on_rim])[:, None] * normal[~on_rim]
        vertices = vertices + .5 * move
        nearest = surface_tree.query(vertices[~on_rim])[1]
        vertices[~on_rim] -= np.einsum('ij,ij->i', vertices[~on_rim] - points[nearest], normals[nearest])[:, None] * normals[nearest]
        vertices[on_rim] = rim[rim_tree.query(vertices[on_rim])[1]]
    return trimesh.Trimesh(vertices, faces, process=False)
