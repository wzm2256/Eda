import numpy as np
from typing import Tuple, List, Optional, Union, Any


def rotate(points: np.ndarray, rotation: np.ndarray, normals: Optional[np.ndarray] = None):
    points = np.matmul(points, rotation.T)
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    else:
        return points, None


def random_sample_points(points, num_samples_, normals=None, color=None, seed=None):
    r"""Randomly sample points."""
    num_points = points.shape[0]
    # pdb.set_trace()
    if seed is None:
        sel_indices = np.random.permutation(num_points)
    else:
        rng = np.random.RandomState(seed)
        sel_indices = rng.permutation(num_points)

    if num_samples_ < 0:
        num_samples = num_points
    else:
        num_samples = num_samples_

    if num_points >= num_samples:
        sel_indices = sel_indices[:num_samples]
    # else:
        # raise ValueError('Requiring too many points!')
    points = points[sel_indices]
    if color is not None:
        colors = color[sel_indices]
        if normals is not None:
            normals = normals[sel_indices]
            return points, normals, colors
        else:
            return points, colors
    if normals is not None:
        normals = normals[sel_indices]
        return points, normals
    else:
        return points, None


def random_jitter_points(points, scale, noise_magnitude=0.05, seed=None):
    r"""Randomly jitter point cloud."""
    if seed is None:
        noise = np.random.normal(scale=scale, size=points.shape)
    else:
        rng = np.random.RandomState(seed)
        noise = rng.normal(scale=scale, size=points.shape)

    noises = np.clip(noise, a_min=-noise_magnitude, a_max=noise_magnitude)
    points = points + noises
    return points


def random_sample_plane(seed=None):
    r"""Random sample a plane passing the origin and return its normal."""
    if seed is None:
        a = np.random.randn(3) * 100
    else:
        rng = np.random.RandomState(seed)
        a = rng.randn(3) * 100
    norm = np.linalg.norm(a) + 1e-8
    return a / norm


def random_crop_point_cloud_with_plane(points, p_normal=None, keep_ratio=0.7, normals=None, seed=None):
    r"""Random crop a point cloud with a plane and keep num_samples points."""
    num_samples = int(np.floor(points.shape[0] * keep_ratio + 0.5))
    if p_normal is None:
        p_normal = random_sample_plane(seed=seed)  # (3,)
    distances = np.dot(points, p_normal)
    sel_indices = np.argsort(-distances)[:num_samples]  # select the largest K points
    points = points[sel_indices]
    if normals is not None:
        normals = normals[sel_indices]
        return points, normals
    else:
        return points, None


from scipy.spatial.transform import Rotation


def random_sample_rotation(batch: int, rotation_deg: float = 180., rng=None) -> np.ndarray:
    axis_ = rng.randn(batch, 3)
    deg = (rng.rand(batch, 1) * 2 - 1) * rotation_deg / 180. * np.pi

    axis = axis_ / np.linalg.norm(axis_, axis=1, keepdims=True) * deg
    rotation = Rotation.from_rotvec(axis).as_matrix()
    return rotation