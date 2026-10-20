import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from services.motion.src.schema import TranslationRequest
from services.motion.src.body import native_poses, rotation_matrices, BodyModel, BETAS
from services.motion.src.paths import BODY_MODEL
from services.motion.src import app as api
from services.motion.src.phrases import phrase_plan
from services.motion.src.soke_provider import blend_motion
from services.motion.src.phrase_constraints import point_at_addressee

def test_sentence_validation_preserves_punctuation_and_rejects_empty_input():
    assert TranslationRequest(text="  Where   is the library? ").text == "Where is the library?"
    for text in [" \n\t", "x" * 401, "word " * 61]:
        with pytest.raises(ValidationError):
            TranslationRequest(text=text)

def test_native_motion_joint_order_and_non_flat_hand_mean_are_preserved():
    motion = np.arange(133, dtype=np.float32)[None] / 100
    left = np.ones(45, dtype=np.float32) * 0.1
    right = np.ones(45, dtype=np.float32) * -0.2
    poses = native_poses(motion, left, right)
    np.testing.assert_array_equal(poses[0, :12], 0)
    np.testing.assert_allclose(poses[0, 12:22].reshape(-1), motion[0, :30])
    np.testing.assert_allclose(poses[0, 22], motion[0, 120:123])
    np.testing.assert_allclose(poses[0, 25:40].reshape(-1), motion[0, 30:75] + left)
    np.testing.assert_allclose(poses[0, 40:55].reshape(-1), motion[0, 75:120] + right)
    with pytest.raises(ValueError):
        native_poses(np.full((1, 133), np.nan), left, right)

def test_rodrigues_zero_and_quarter_turn():
    matrices = rotation_matrices(np.array([[0, 0, 0], [0, 0, np.pi / 2]], dtype=np.float32))
    np.testing.assert_allclose(matrices[0], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(matrices[1] @ [1, 0, 0], [0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(matrices @ np.swapaxes(matrices, -1, -2), np.broadcast_to(np.eye(3), (2, 3, 3)), atol=1e-6)

def test_greeting_handles_original_case_punctuation_without_dropping_other_meanings():
    for text in ['Hi! How are you?', 'Hi,hOW ARE YOU?', 'Hello, how are you?', 'hey how are you']:
        assert phrase_plan(text)['glosses'] == ['hello', 'how', 'you']
    assert phrase_plan('How are you?')['glosses'] == ['how', 'you']
    assert phrase_plan('Hello!')['glosses'] == ['hello']
    for text in ['How are you feeling?', 'How old are you?', 'Hello, my name is Alex.', 'I want water.']:
        assert phrase_plan(text) is None

@pytest.mark.skipif(not BODY_MODEL.exists(), reason="Body asset is not installed")
def test_transition_uses_short_rotation_path_and_preserves_native_hand_means():
    first = np.zeros(133, dtype=np.float32)
    last = first.copy()
    first[2], last[2] = np.deg2rad(179), np.deg2rad(-179)
    last[123:] = 1
    result = blend_motion(first, last, 3)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(rotation_matrices(result[1, :3]) @ [1, 0, 0], [-1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(result[:, 30:120], 0, atol=1e-6)
    np.testing.assert_allclose(result[1, 123:], .5)

@pytest.mark.skipif(not BODY_MODEL.exists(), reason="Body asset is not installed")
def test_you_constraint_points_forward_without_changing_handshape_or_arm_motion():
    body = BodyModel(); body.load()
    clip = np.zeros((40, 133), dtype=np.float32)
    clip[:, 75:120] = -body.right_mean
    corrected = point_at_addressee(clip, body)
    np.testing.assert_allclose(corrected[:, :27], clip[:, :27])
    np.testing.assert_allclose(corrected[:, 30:], clip[:, 30:])
    np.testing.assert_allclose(corrected[[0,-1]], clip[[0,-1]], atol=1e-6)
    rotations = rotation_matrices(native_poses(corrected, body.left_mean, body.right_mean))
    world, points = [], []
    for j,p in enumerate(body.parents):
        world.append(rotations[:,j] if p < 0 else world[p] @ rotations[:,j])
        points.append(np.broadcast_to(body.joints[j], (40,3)) if p < 0 else points[p] + np.einsum('bij,j->bi',world[p],body.joints[j]-body.joints[p]))
    direction = points[42] - points[40]
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    np.testing.assert_allclose(direction[8:32], np.tile([0,0,1],(24,1)), atol=2e-5)

@pytest.mark.skipif(not BODY_MODEL.exists(), reason="Licensed body asset is not installed")
def test_native_body_preserves_facial_channels_and_all_skin_weights(tmp_path):
    body = BodyModel()
    body.load()
    assert body.weights.shape == (10475, 55)
    np.testing.assert_allclose(body.weights.sum(1), 1, atol=1e-6)
    motion = np.zeros((2, 133), dtype=np.float32)
    motion[1, 123] = 1
    result = body.prepare(motion, tmp_path)
    positions = np.fromfile(tmp_path / "positions.bin", dtype="<f4").reshape(2, -1, 3)
    np.testing.assert_allclose(positions[1] - positions[0], body.expression_basis[0].reshape(-1, 3), atol=1e-6)
    assert np.isfinite(positions).all()
    assert result["skinning"] == "all-55-weights"
    assert result["handMean"] == "native-smplx"

def test_api_rejects_cross_origin_invalid_text_and_asset_path_traversal(monkeypatch):
    client = TestClient(api.app)
    assert client.post('/api/translations', json={"text": " "}).status_code == 422
    assert client.post('/api/translations', json={"text": "Hello"}, headers={"Origin": "https://elsewhere.example"}).status_code == 403
    assert client.get('/api/assets/not-a-hash/positions.bin').status_code == 404
    assert client.get('/api/translations/unknown').status_code == 404

def test_job_cache_and_failure_have_explicit_states(tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'CACHE', tmp_path)
    monkeypatch.setattr(api.body, 'validate_motion', lambda motion: {})
    api.jobs.clear()
    api.jobs['test'] = {'id': 'test', 'status': 'queued', 'message': '', 'cacheKey': 'a' * 64}
    cache = tmp_path / ('a' * 64)
    cache.mkdir()
    (cache / 'result.json').write_text(json.dumps({'text': 'Hello', 'cached': False}))
    (cache / 'positions.bin').write_bytes(b'\0' * 4)
    np.save(cache / 'motion133.npy', np.zeros((2, 133), dtype=np.float32))
    api.run_job('test', TranslationRequest(text='Hello'), 'a' * 64)
    assert api.jobs['test']['status'] == 'complete'
    assert api.jobs['test']['result']['cached']
    monkeypatch.setattr(api.provider, 'load', lambda: (_ for _ in ()).throw(RuntimeError('Missing model')))
    api.run_job('test', TranslationRequest(text='Hello'), 'b' * 64)
    assert api.jobs['test']['status'] == 'failed'
    assert api.jobs['test']['message'] == 'Missing model'

@pytest.mark.skipif(not BODY_MODEL.exists(), reason="Licensed body asset is not installed")
def test_backward_arm_failure_is_rejected_but_forward_reach_is_allowed():
    body = BodyModel()
    motion = np.zeros((40, 133), dtype=np.float32)
    # Elbow flexion in opposite directions produces an unambiguous native
    # forward/backward reach without involving a camera or handshape guess.
    motion[:, 19] = np.pi / 2
    motion[:, 22] = -np.pi / 2
    with pytest.raises(ValueError, match="backward arm poses"):
        body.validate_motion(motion)
    motion *= -1
    assert body.validate_motion(motion)['bothWristsBehindFraction'] == 0

def test_invalid_cached_motion_cannot_bypass_playback_check(tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'CACHE', tmp_path)
    api.jobs.clear()
    api.jobs['bad'] = {'id': 'bad', 'status': 'queued', 'message': '', 'cacheKey': 'c' * 64}
    cache = tmp_path / ('c' * 64)
    cache.mkdir()
    (cache / 'result.json').write_text('{}')
    (cache / 'positions.bin').write_bytes(b'\0' * 4)
    np.save(cache / 'motion133.npy', np.zeros((2, 133), dtype=np.float32))
    def reject(motion):
        raise ValueError('backward arm poses')
    monkeypatch.setattr(api.body, 'validate_motion', reject)
    api.run_job('bad', TranslationRequest(text='Hi,hOW ARE YOU?'), 'c' * 64)
    assert api.jobs['bad']['status'] == 'failed'
    assert api.jobs['bad']['result'] is None
    assert api.jobs['bad']['message'] == 'backward arm poses'

@pytest.mark.skipif(not BODY_MODEL.exists(), reason="Licensed body asset is not installed")
def test_native_adapter_matches_original_smplx_reference():
    import torch
    from smplx import SMPLX
    body = BodyModel()
    body.load()
    motion = np.random.default_rng(12).normal(0, 0.12, (2, 133)).astype(np.float32)
    reference = SMPLX(str(body.path), use_pca=False, batch_size=2, num_expression_coeffs=10)
    body_pose = torch.zeros(2, 63)
    body_pose[:, 33:] = torch.from_numpy(motion[:, :30])
    with torch.inference_mode():
        output = reference(betas=torch.tensor(BETAS).repeat(2, 1), body_pose=body_pose,
            left_hand_pose=torch.from_numpy(motion[:, 30:75]), right_hand_pose=torch.from_numpy(motion[:, 75:120]),
            jaw_pose=torch.from_numpy(motion[:, 120:123]), expression=torch.from_numpy(motion[:, 123:]), return_full_pose=True)
    poses = native_poses(motion, body.left_mean, body.right_mean)
    np.testing.assert_allclose(poses.reshape(2, -1), output.full_pose.numpy(), atol=1e-7)
    rotations = rotation_matrices(poses)
    posed = ((rotations[:, 1:] - np.eye(3, dtype=np.float32)).reshape(2, -1) @ body.pose_basis
             + motion[:, 123:] @ body.expression_basis + body.vertices.reshape(1, -1))
    world = []
    for joint in range(55):
        matrix = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4)).copy()
        matrix[:, :3, :3] = rotations[:, joint]
        parent = body.parents[joint]
        matrix[:, :3, 3] = body.joints[joint] if parent < 0 else body.joints[joint] - body.joints[parent]
        world.append(matrix if parent < 0 else world[parent] @ matrix)
    transforms = np.stack(world, axis=1)
    transforms[:, :, :3, 3] -= np.einsum('bjcd,jd->bjc', transforms[:, :, :3, :3], body.joints)
    weighted = np.einsum('vj,bjcd->bvcd', body.weights, transforms)
    homogeneous = np.concatenate([posed.reshape(2, -1, 3), np.ones((2, len(body.vertices), 1), dtype=np.float32)], axis=-1)
    actual = np.einsum('bvcd,bvd->bvc', weighted, homogeneous)[..., :3]
    # Original torch implementation uses epsilon in Rodrigues; native float32
    # implementations should agree within twenty micrometres across the mesh.
    np.testing.assert_allclose(actual, output.vertices.numpy(), atol=2e-5, rtol=0)
