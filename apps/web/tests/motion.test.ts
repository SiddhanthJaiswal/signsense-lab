import test from 'node:test';
import assert from 'node:assert/strict';
import { Quaternion, Vector3 } from 'three';
import { rotationVectorToQuaternion, frameAt } from '../src/avatar/motion.ts';

test('native rotation vectors preserve axis and handedness', () => {
  const q = rotationVectorToQuaternion(0, 0, Math.PI / 2, new Quaternion());
  const result = new Vector3(1, 0, 0).applyQuaternion(q);
  assert.ok(result.distanceTo(new Vector3(0, 1, 0)) < 1e-6);
  assert.deepEqual(rotationVectorToQuaternion(0, 0, 0, q).toArray(), [0, 0, 0, 1]);
});
test('timeline scrubbing clamps safely to available motion frames', () => {
  assert.deepEqual(frameAt(-1, 20, 40), { a: 0, b: 1, alpha: 0 });
  assert.deepEqual(frameAt(0.075, 20, 40), { a: 1, b: 2, alpha: 0.5 });
  assert.deepEqual(frameAt(2, 20, 40), { a: 39, b: 39, alpha: 0 });
});
