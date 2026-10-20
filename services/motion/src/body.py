"""Native SMPL-X rig + deformations. No approximate gloss dictionary or rig retargeting."""
import json
from pathlib import Path
import numpy as np

from .paths import BODY_MODEL, RUNTIME

# Same shape used by SOKE's How2Sign visualizer; it is not identity-conditioned generation.
BETAS = np.array([-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
                  0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842], dtype=np.float32)

def rotation_matrices(vectors):
    """Stable Rodrigues, including exactly-zero rotations."""
    vectors = np.asarray(vectors, dtype=np.float32)
    angles = np.linalg.norm(vectors, axis=-1, keepdims=True)
    axes = np.divide(vectors, angles, out=np.zeros_like(vectors), where=angles > 1e-8)
    x, y, z = np.moveaxis(axes, -1, 0)
    zero = np.zeros_like(x)
    skew = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=-1).reshape(*vectors.shape[:-1], 3, 3)
    identity = np.broadcast_to(np.eye(3, dtype=np.float32), skew.shape)
    sin = np.sin(angles)[..., None]
    cos = np.cos(angles)[..., None]
    return identity + sin * skew + (1 - cos) * (skew @ skew)

def native_poses(motion, left_mean, right_mean):
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 133 or not np.isfinite(motion).all():
        raise ValueError("Expected finite [frames, 133] motion")
    poses = np.zeros((len(motion), 55, 3), dtype=np.float32)
    poses[:, 12:22] = motion[:, :30].reshape(-1, 10, 3)
    poses[:, 22] = motion[:, 120:123]
    poses[:, 25:40] = (motion[:, 30:75] + left_mean).reshape(-1, 15, 3)
    poses[:, 40:55] = (motion[:, 75:120] + right_mean).reshape(-1, 15, 3)
    return poses

class BodyModel:
    def __init__(self, path=BODY_MODEL):
        self.path = Path(path)
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        if not self.path.exists():
            raise RuntimeError("The SMPL-X body asset has not finished downloading.")
        # Read numerical arrays only. No object/pickle entries from third-party npz files.
        with np.load(self.path, allow_pickle=False) as data:
            template = np.asarray(data["v_template"], dtype=np.float32)
            shapes = np.asarray(data["shapedirs"], dtype=np.float32)
            self.vertices = template + np.einsum("vck,k->vc", shapes[:, :, :10], BETAS)
            self.expression_basis = shapes[:, :, 300:310].reshape(-1, 10).T.copy()
            if self.expression_basis.shape[0] != 10:
                raise ValueError("SMPL-X asset has no 10-dimensional expression basis.")
            self.pose_basis = np.asarray(data["posedirs"], dtype=np.float32).reshape(-1, 486).T.copy()
            self.joints = np.asarray(data["J_regressor"], dtype=np.float32) @ self.vertices
            self.parents = np.asarray(data["kintree_table"])[0].astype(np.int64)
            self.parents[0] = -1
            self.faces = np.asarray(data["f"], dtype=np.uint32)
            self.weights = np.asarray(data["weights"], dtype=np.float32)
            self.left_mean = np.asarray(data["hands_meanl"], dtype=np.float32)
            self.right_mean = np.asarray(data["hands_meanr"], dtype=np.float32)
        if self.weights.shape != (len(self.vertices), 55):
            raise ValueError("Expected the native 55-joint SMPL-X rig")
        self.loaded = True

    def rig(self):
        self.load()
        return {"version": "smplx-rig-v1", "vertices": self.vertices.reshape(-1).tolist(),
                "faces": self.faces.reshape(-1).tolist(), "joints": self.joints.tolist(),
                "parents": self.parents.tolist(), "weights": self.weights.reshape(-1).tolist(),
                "vertexCount": len(self.vertices), "jointCount": 55, "betas": BETAS.tolist()}

    def motion_diagnostics(self, motion):
        """Measure the reported backward-arm failure in native model coordinates.

        SMPL-X faces +Z. Use the chest's moving frame, rather than camera/world
        coordinates, so turning the avatar cannot change this check. This is a
        narrow anatomy check, not an ASL correctness score.
        """
        self.load()
        rotations = rotation_matrices(native_poses(motion, self.left_mean, self.right_mean))
        world_rot, world_pos = [], []
        for joint, parent in enumerate(self.parents):
            if parent < 0:
                world_rot.append(rotations[:, joint])
                world_pos.append(np.broadcast_to(self.joints[joint], (len(motion), 3)))
            else:
                world_rot.append(world_rot[parent] @ rotations[:, joint])
                offset = self.joints[joint] - self.joints[parent]
                world_pos.append(world_pos[parent] + np.einsum('bij,j->bi', world_rot[parent], offset))
        chest = (world_pos[16] + world_pos[17]) / 2
        forward = world_rot[9][:, :, 2]
        depth = np.stack([np.einsum('bi,bi->b', world_pos[j] - chest, forward) for j in (20, 21)], axis=1)
        return {"bothWristsBehindFraction": float(np.mean(np.all(depth < -0.15, axis=1))),
                "wristDepthMetres": {"min": float(depth.min()), "max": float(depth.max())}}

    def validate_motion(self, motion):
        diagnostics = self.motion_diagnostics(motion)
        if diagnostics["bothWristsBehindFraction"] > 0.8:
            raise ValueError("The model produced backward arm poses (both hands behind the torso). "
                             "Playback was stopped. The current research checkpoint needs correction; "
                             "this sentence has not been translated reliably.")
        return diagnostics

    def prepare(self, motion, directory):
        """Bake expression + pose correctives, retaining all original skinning weights.

        The browser skins with the native 55-joint skeleton and the original complete
        weight table. Facial coefficients drive their actual SMPL-X basis, not a
        guessed ARKit mapping. Files remain local and are excluded from Git.
        """
        self.load()
        poses = native_poses(motion, self.left_mean, self.right_mean)
        matrices = rotation_matrices(poses)
        features = (matrices[:, 1:] - np.eye(3, dtype=np.float32)).reshape(len(motion), -1)
        expression = motion[:, 123:133]
        positions = features @ self.pose_basis + expression @ self.expression_basis
        positions += self.vertices.reshape(1, -1)
        if not np.isfinite(positions).all():
            raise ValueError("Invalid body deformation")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        positions.astype("<f4").tofile(directory / "positions.bin")
        # Keep raw research output available for evaluation and another renderer.
        np.save(directory / "motion133.npy", motion.astype(np.float32))
        rotations = poses.reshape(len(motion), -1).tolist()
        return {"rotations": rotations, "frames": len(motion), "vertexCount": len(self.vertices),
                "positionBytes": int(positions.nbytes), "expressions": "native-smplx-10",
                "poseCorrectives": True, "handMean": "native-smplx", "skinning": "all-55-weights"}
