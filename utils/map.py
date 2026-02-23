import torch
import numpy as np
import math
from trimesh.remesh import subdivide


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def calc_radius(bboxes_hw_norm, map_size=64):
    if len(bboxes_hw_norm) == 0:
        return []
    minimum_radius = map_size / 32.
    scale_factor = map_size / 16.
    scales = np.linalg.norm(np.array(bboxes_hw_norm) / 2, ord=2, axis=1)
    radius = (scales * scale_factor + minimum_radius).astype(np.uint8)
    return radius


def build_z_map(depth, with_coords=True, device='cpu'):
    size = 2**depth
    z_map = torch.zeros((size, size), dtype=torch.int64)
    coords = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    ys = coords[0]
    xs = coords[1]
    for i in range(depth):
        z_map |= (xs & (1 << i)) << i | (ys & (1 << i)) << (i + 1)
    if with_coords:
        return z_map, ys.clone(), xs.clone()
    else:
        return z_map


def gen_scale_map(scales, v3ds, faces, cam_intrinsics, map_size, patch_size=28, pad=True):
    if pad:
        map_h = math.ceil(map_size[0] / 2) * 2
        map_w = math.ceil(map_size[1] / 2) * 2
    else:
        map_h = map_size[0]
        map_w = map_size[1]
    scale_map = torch.zeros((map_h, map_w, 2))

    new_v3ds = []
    for v in v3ds:
        vv, _ = subdivide(v, faces)
        new_v3ds.append(torch.from_numpy(vv))

    v3ds = torch.stack(new_v3ds)

    v2ds_homo = torch.matmul(v3ds, cam_intrinsics.transpose(-1, -2))
    v2ds = v2ds_homo[..., :2] / (v2ds_homo[..., 2, None])
    v2ds_patch = (v2ds // patch_size).int()
    v2ds_patch[..., 0] = v2ds_patch[..., 0].clamp(min=0, max=map_size[1] - 1)
    v2ds_patch[..., 1] = v2ds_patch[..., 1].clamp(min=0, max=map_size[0] - 1)
    for (v, s) in zip(v2ds_patch, scales):
        scale_map[v[:, 1], v[:, 0], 0] = 1.
        scale_map[v[:, 1], v[:, 0], 1] = s

    return scale_map
