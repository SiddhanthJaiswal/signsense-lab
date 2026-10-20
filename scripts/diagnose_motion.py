"""Reproduce native wrist-depth diagnostics; optionally test the upstream decoder on CPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor/robostar'))
import numpy as np
from services.motion.src.body import BodyModel
from services.motion.src.paths import CACHE, MODEL_ROOT
from services.motion.src.schema import MODEL_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decode-reference', action='store_true')
    args = parser.parse_args()
    body = BodyModel()
    rows, motions = [], []
    for path in sorted(CACHE.glob('*/result.json')):
        result = json.loads(path.read_text())
        motion = np.load(path.parent / 'motion133.npy', allow_pickle=False)
        rows.append({'text': result['text'], 'frames': len(motion),
                     'cacheKey': path.parent.name, **body.motion_diagnostics(motion)})
        motions.append(motion)
    identical = []
    for i in range(len(rows)):
        for j in range(i):
            a, b = motions[i], motions[j]
            if a.shape == b.shape and np.array_equal(a[:, :30], b[:, :30]):
                identical.append([rows[j]['text'], rows[i]['text']])
    report = {'modelRevision': MODEL_REVISION, 'cachedSentences': rows,
              'identicalBodyMotionPairs': identical, 'aslAccuracy': 'failed observed samples; not validated'}
    if args.decode_reference:
        import torch
        from robostar.tokenizer import RoboSTARTokenizer
        from robostar.representation import denormalize_motion
        tokenizer = RoboSTARTokenizer.from_pretrained(MODEL_ROOT, device='cpu').eval()
        word = json.loads((MODEL_ROOT / 'retrieval/word2code.json').read_text())['WATER']
        codes = {part: torch.tensor([word[key]]) for part, key in [('body', 'body'), ('left', 'lhand'), ('right', 'rhand')]}
        with torch.inference_mode():
            normalized = tokenizer.decode(codes)[0].numpy()
        motion = denormalize_motion(normalized, np.load(MODEL_ROOT / 'stats/mean133.npy'), np.load(MODEL_ROOT / 'stats/std133.npy'))
        report['publishedWaterReferenceOnCPU'] = body.motion_diagnostics(motion)
    output = ROOT / 'runtime/motion-diagnostics.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
