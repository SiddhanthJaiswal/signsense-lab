import * as THREE from 'three';
import type { Rig, Performance } from '../types';
import { frameAt, rotationVectorToQuaternion } from './motion';

/** Native SMPL-X skeleton and full-weight GPU skinning. The numerical mesh asset
 * comes from the local API only; there is no bundled or redistributed body model.
 */
export class NativeAvatar {
  mesh: THREE.SkinnedMesh;
  bones: THREE.Bone[] = [];
  skeleton: THREE.Skeleton;
  private weights: THREE.DataTexture;
  private first = new THREE.Quaternion();
  private second = new THREE.Quaternion();
  private lastNormalFrame = -1;

  constructor(readonly rig: Rig, maxTextureSize: number) {
    if (rig.jointCount !== 55 || rig.weights.length !== rig.vertexCount * 55) throw new Error('The avatar rig is incompatible.');
    if (rig.vertexCount > maxTextureSize) throw new Error('This GPU cannot hold the native skin-weight texture. A desktop WebGL2 GPU is required.');
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(rig.vertices, 3));
    geometry.setIndex(rig.faces);
    geometry.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(new Uint16Array(rig.vertexCount * 4), 4));
    const placeholderWeights = new Float32Array(rig.vertexCount * 4);
    for (let i = 0; i < rig.vertexCount; i++) placeholderWeights[i * 4] = 1;
    geometry.setAttribute('skinWeight', new THREE.Float32BufferAttribute(placeholderWeights, 4));
    const colors = new Float32Array(rig.vertexCount * 3);
    const body = new THREE.Color('#52634e'), skin = new THREE.Color('#c9c5b4'), eyes = new THREE.Color('#434a41');
    const color = new THREE.Color();
    for (let v = 0; v < rig.vertexCount; v++) {
      const eye = rig.weights[v * 55 + 23] + rig.weights[v * 55 + 24];
      const offset = v * 55;
      let skinWeight = rig.weights[offset + 12] + rig.weights[offset + 15];
      for (let j = 20; j < 55; j++) skinWeight += rig.weights[offset + j];
      color.copy(body).lerp(skin, Math.min(1, skinWeight * 1.2));
      if (eye > 0.5) color.copy(eyes);
      colors.set([color.r, color.g, color.b], v * 3);
    }
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.computeVertexNormals();
    // 14 RGBA texels contain all 55 weights for a vertex (last component unused).
    const weightData = new Float32Array(rig.vertexCount * 56);
    for (let v = 0; v < rig.vertexCount; v++) weightData.set(rig.weights.slice(v * 55, (v + 1) * 55), v * 56);
    this.weights = new THREE.DataTexture(weightData, 14, rig.vertexCount, THREE.RGBAFormat, THREE.FloatType);
    this.weights.needsUpdate = true;
    this.weights.minFilter = this.weights.magFilter = THREE.NearestFilter;
    const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.88, metalness: 0.02 });
    material.onBeforeCompile = (shader) => {
      shader.uniforms.nativeWeights = { value: this.weights };
      shader.vertexShader = `uniform sampler2D nativeWeights;\n${shader.vertexShader}`;
      shader.vertexShader = shader.vertexShader.replace('#include <skinbase_vertex>', `
        #ifdef USE_SKINNING
          mat4 nativeSkin = mat4(0.0);
          for (int j = 0; j < 55; j++) {
            vec4 weights = texelFetch(nativeWeights, ivec2(j / 4, gl_VertexID), 0);
            float w = weights[j % 4];
            if (w > 0.0) nativeSkin += w * getBoneMatrix(float(j));
          }
        #endif
      `).replace('#include <skinnormal_vertex>', `
        #ifdef USE_SKINNING
          mat4 normalSkin = bindMatrixInverse * nativeSkin * bindMatrix;
          objectNormal = vec4(normalSkin * vec4(objectNormal, 0.0)).xyz;
          #ifdef USE_TANGENT
            objectTangent = vec4(normalSkin * vec4(objectTangent, 0.0)).xyz;
          #endif
        #endif
      `).replace('#include <skinning_vertex>', `
        #ifdef USE_SKINNING
          transformed = (bindMatrixInverse * nativeSkin * bindMatrix * vec4(transformed, 1.0)).xyz;
        #endif
      `);
    };
    material.customProgramCacheKey = () => 'smplx-full-55-v1';
    this.mesh = new THREE.SkinnedMesh(geometry, material);
    this.mesh.name = 'Native SMPL-X signer';
    this.mesh.frustumCulled = false;
    for (let i = 0; i < 55; i++) {
      const bone = new THREE.Bone();
      bone.name = `smplx_${i}`;
      const parent = rig.parents[i];
      bone.position.fromArray(rig.joints[i]);
      if (parent >= 0) bone.position.sub(new THREE.Vector3().fromArray(rig.joints[parent]));
      this.bones.push(bone);
    }
    this.bones.forEach((bone, i) => rig.parents[i] < 0 ? this.mesh.add(bone) : this.bones[rig.parents[i]].add(bone));
    this.skeleton = new THREE.Skeleton(this.bones);
    this.mesh.bind(this.skeleton);
    this.mesh.updateMatrixWorld(true);
    this.rest();
  }

  rest() {
    // Restore both the pose and deformation after a failed/new generation.
    for (const bone of this.bones) bone.quaternion.identity();
    this.bones[16].rotation.z = -1.15;
    this.bones[17].rotation.z = 1.15;
    const attribute = this.mesh.geometry.getAttribute('position') as THREE.BufferAttribute;
    (attribute.array as Float32Array).set(this.rig.vertices);
    attribute.needsUpdate = true;
    this.mesh.geometry.computeVertexNormals();
    this.lastNormalFrame = -1;
  }

  sample(performance: Performance, time: number) {
    const { result, positions } = performance;
    const { a, b, alpha } = frameAt(time, result.fps, result.rendering.frames);
    const ra = result.rendering.rotations[a], rb = result.rendering.rotations[b];
    for (let j = 0; j < 55; j++) {
      const k = j * 3;
      rotationVectorToQuaternion(ra[k], ra[k + 1], ra[k + 2], this.first);
      rotationVectorToQuaternion(rb[k], rb[k + 1], rb[k + 2], this.second);
      this.bones[j].quaternion.copy(this.first).slerp(this.second, alpha);
    }
    const attribute = this.mesh.geometry.getAttribute('position') as THREE.BufferAttribute;
    const array = attribute.array as Float32Array;
    const count = this.rig.vertexCount * 3;
    const ai = a * count, bi = b * count;
    for (let i = 0; i < count; i++) array[i] = positions[ai + i] * (1 - alpha) + positions[bi + i] * alpha;
    attribute.needsUpdate = true;
    if (a !== this.lastNormalFrame) {
      this.mesh.geometry.computeVertexNormals();
      this.lastNormalFrame = a;
    }
  }

  dispose() {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
    this.skeleton.dispose();
    this.weights.dispose();
  }
}
