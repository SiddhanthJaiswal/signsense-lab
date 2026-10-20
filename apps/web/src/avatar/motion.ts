import { Quaternion, Vector3 } from 'three';

const axis = new Vector3();
export function rotationVectorToQuaternion(x: number, y: number, z: number, target: Quaternion): Quaternion {
  const angle = Math.hypot(x, y, z);
  return angle < 1e-8 ? target.identity() : target.setFromAxisAngle(axis.set(x / angle, y / angle, z / angle), angle);
}

export function frameAt(time: number, fps: number, frameCount: number) {
  const position = Math.max(0, Math.min(frameCount - 1, time * fps));
  const a = Math.floor(position);
  return { a, b: Math.min(a + 1, frameCount - 1), alpha: position - a };
}
