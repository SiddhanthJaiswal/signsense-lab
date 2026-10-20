"""Explicit spatial correction for the greeting's second-person pronoun.

ASLU's greeting reference says YOU points at the addressee. In this front-facing
demo the addressee is along native +Z. Preserve the decoded handshape and arm
motion, correcting only the wrist aim during the sign's central hold.
This is an authored constraint, not a claim that SOKE learned this correction.
"""
import numpy as np
from .body import native_poses, rotation_matrices
from .rotations import to_quaternion, to_rotvec, multiply

def point_at_addressee(clip, body):
    body.load()
    result = clip.copy()
    poses = native_poses(clip, body.left_mean, body.right_mean)
    rotations = rotation_matrices(poses)
    world_r, world_p = [], []
    for joint, parent in enumerate(body.parents):
        if parent < 0:
            world_r.append(rotations[:, joint])
            world_p.append(np.broadcast_to(body.joints[joint], (len(clip), 3)))
        else:
            world_r.append(world_r[parent] @ rotations[:, joint])
            world_p.append(world_p[parent] + np.einsum('bij,j->bi', world_r[parent], body.joints[joint]-body.joints[parent]))
    direction = world_p[42] - world_p[40]  # Right index proximal → distal.
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    target = np.broadcast_to([0., 0., 1.], direction.shape)
    # Express both vectors in the wrist parent's (right elbow) coordinate frame.
    inverse = np.swapaxes(world_r[19], -1, -2)
    source = np.einsum('bij,bj->bi', inverse, direction)
    target = np.einsum('bij,bj->bi', inverse, target)
    correction = np.concatenate([np.cross(source, target), 1 + np.sum(source*target, axis=-1, keepdims=True)], axis=-1)
    norm = np.linalg.norm(correction, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise ValueError('Ambiguous opposite pointing direction')
    correction /= norm
    t = np.linspace(0, 1, len(clip))
    envelope = np.minimum(np.clip(t / .2, 0, 1), np.clip((1-t) / .2, 0, 1))
    envelope = envelope*envelope*(3-2*envelope)
    correction = to_quaternion(to_rotvec(correction) * envelope[:, None])
    result[:, 27:30] = to_rotvec(multiply(correction, to_quaternion(clip[:, 27:30])))
    return result
