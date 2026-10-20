"""Quaternion helpers for native rotation vectors (xyzw convention)."""
import numpy as np

def to_quaternion(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    angles = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.concatenate([vectors * (0.5 * np.sinc(angles / (2 * np.pi))), np.cos(angles / 2)], axis=-1)

def to_rotvec(quaternions):
    q = np.asarray(quaternions, dtype=np.float64).copy()
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q *= np.where(q[..., 3:] < 0, -1, 1)
    sine = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(sine, q[..., 3:])
    return q[..., :3] * np.divide(angle, sine, out=np.full_like(sine, 2), where=sine > 1e-12)

def multiply(a, b):
    return np.concatenate([a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3] +
                           np.cross(a[..., :3], b[..., :3]),
                           a[..., 3:] * b[..., 3:] - np.sum(a[..., :3] * b[..., :3], axis=-1, keepdims=True)], axis=-1)
