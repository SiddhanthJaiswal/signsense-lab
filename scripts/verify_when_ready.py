"""One-shot setup verification. Wait for downloads, run real inference, then exit."""
import json
from pathlib import Path
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "runtime/setup-verification.json"

def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request("http://127.0.0.1:8000" + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)

def save(value):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(value, indent=2) + "\n")

def main():
    save({"status": "waiting_for_model", "semanticAccuracy": "not_reviewed"})
    deadline = time.monotonic() + 6 * 3600
    while time.monotonic() < deadline:
        health = request('/api/health')
        if health['ready']:
            break
        time.sleep(10)
    else:
        raise RuntimeError('Model download did not finish within six hours.')
    sentence = 'Hi! How are you?'
    save({"status": "running_real_inference", "text": sentence, "semanticAccuracy": "not_reviewed"})
    job = request('/api/translations', {"text": sentence})
    inference_deadline = time.monotonic() + 1800
    while job['status'] not in {'complete', 'failed'} and time.monotonic() < inference_deadline:
        time.sleep(2)
        job = request('/api/translations/' + job['id'])
    if job['status'] != 'complete' or not job.get('result'):
        raise RuntimeError(job['message'])
    result = job['result']
    if result['metadata'].get('model') != 'SOKE' or result['metadata'].get('glosses') != ['HELLO', 'HOW', 'YOU']:
        raise RuntimeError('The greeting used the wrong model or phrase plan')
    with urllib.request.urlopen('http://127.0.0.1:8000' + result['assets']['positions'], timeout=60) as response:
        raw = response.read()
    expected = result['rendering']['frames'] * result['rendering']['vertexCount'] * 3 * 4
    if len(raw) != expected:
        raise RuntimeError('Motion binary size mismatch')
    import numpy as np
    positions = np.frombuffer(raw, dtype='<f4').reshape(result['rendering']['frames'], -1)
    if not np.isfinite(positions).all() or not np.any(np.abs(np.diff(positions, axis=0)) > 1e-6):
        raise RuntimeError('Model output is non-finite or motionless')
    import io, sys
    sys.path.insert(0, str(ROOT))
    from services.motion.src.body import BodyModel, native_poses, rotation_matrices
    with urllib.request.urlopen('http://127.0.0.1:8000' + result['assets']['rawMotion'], timeout=60) as response:
        motion = np.load(io.BytesIO(response.read()), allow_pickle=False)
    body = BodyModel(); body.load()
    diagnostics = body.validate_motion(motion)
    segment = result['metadata']['segments'][-1]
    start, end = round(segment['start'] * result['fps']), round(segment['end'] * result['fps'])
    rotations = rotation_matrices(native_poses(motion[start:end], body.left_mean, body.right_mean))
    world, points = [], []
    for j,p in enumerate(body.parents):
        world.append(rotations[:,j] if p < 0 else world[p] @ rotations[:,j])
        points.append(np.broadcast_to(body.joints[j], (len(rotations),3)) if p < 0 else points[p] +
                      np.einsum('bij,j->bi',world[p],body.joints[j]-body.joints[p]))
    pointing = points[42]-points[40]
    pointing /= np.linalg.norm(pointing, axis=-1, keepdims=True)
    central = pointing[int(.25*len(pointing)):int(.75*len(pointing)),2]
    if central.min() < .99:
        raise RuntimeError('YOU does not point toward the addressee during its hold')
    repeat = request('/api/translations', {'text': sentence})
    if repeat['id'] != job['id']:
        raise RuntimeError('Identical request did not reuse completed job')
    save({"status": "inference_and_native_motion_verified", "text": sentence,
          "jobId": job['id'], "frames": result['rendering']['frames'], "duration": result['duration'],
          "metadata": result['metadata'], "binaryBytes": len(raw), "finite": True, "motionChanges": True,
          "diagnostics": diagnostics, "youForwardDirectionMin": float(central.min()), "jobReuse": True,
          "semanticAccuracy": "not_reviewed", "browserPlayback": "requires inspection of generated sentence"})
    print(f'Real inference verified. Report: {REPORT}', flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save({"status": "failed", "error": str(error), "semanticAccuracy": "not_reviewed"})
        raise
