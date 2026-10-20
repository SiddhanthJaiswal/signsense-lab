import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { getRig } from '../api';
import type { Performance } from '../types';
import { NativeAvatar } from './NativeAvatar';

type Props = { performance: Performance | null; playing: boolean; speed: number; seek: number;
  seekVersion: number; bodyReady: boolean; resetCamera: number; skeleton: boolean;
  onTime: (time: number) => void; onEnd: () => void; onReady: () => void; onError: (error: string) => void };

export function Stage(props: Props) {
  const mount = useRef<HTMLDivElement>(null);
  const latest = useRef(props);
  latest.current = props;
  const avatarRef = useRef<NativeAvatar | null>(null);
  const timeRef = useRef(0);
  const performanceRef = useRef<Performance | null>(null);
  const seekRef = useRef(-1);
  const cameraResetRef = useRef(-1);

  useEffect(() => {
    if (!mount.current) return;
    let disposed = false, frame = 0, avatar: NativeAvatar | null = null;
    let renderer: THREE.WebGLRenderer;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#e7e9df');
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false }); }
    catch { latest.current.onError('WebGL2 is unavailable. Open this demo in a desktop browser with hardware acceleration.'); return; }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    renderer.debug.onShaderError = () => latest.current.onError('The native avatar shader could not compile on this GPU.');
    mount.current.appendChild(renderer.domElement);
    renderer.domElement.setAttribute('aria-label', 'Interactive rigged ASL signer. Drag to orbit and scroll to zoom.');
    renderer.domElement.setAttribute('role', 'img');
    const camera = new THREE.PerspectiveCamera(33, 1, 0.01, 30);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.minDistance = 1.2;
    controls.maxDistance = 5;
    controls.minPolarAngle = Math.PI * 0.25;
    controls.maxPolarAngle = Math.PI * 0.7;
    const reset = () => { camera.position.set(0, 0.0, 1.9); controls.target.set(0, -0.06, 0); controls.update(); };
    reset();
    scene.add(new THREE.HemisphereLight('#fffdf5', '#8d9b85', 2.3));
    const key = new THREE.DirectionalLight('#fff7e7', 3.2); key.position.set(2, 3, 4); scene.add(key);
    const rim = new THREE.DirectionalLight('#f3f6f1', 1.5); rim.position.set(-3, 1, -2); scene.add(rim);
    let helper: THREE.SkeletonHelper | null = null;
    const resize = () => {
      if (!mount.current) return;
      const { clientWidth: w, clientHeight: h } = mount.current;
      renderer.setSize(w, h, false); camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize); observer.observe(mount.current); resize();
    let last = performance.now(), lastUi = 0;
    const render = (now: number) => {
      if (disposed) return;
      try {
      const dt = Math.min((now - last) / 1000, 0.1); last = now;
      const state = latest.current;
      if (state.resetCamera !== cameraResetRef.current) { reset(); cameraResetRef.current = state.resetCamera; }
      if (state.performance !== performanceRef.current) {
        performanceRef.current = state.performance; timeRef.current = 0;
        if (!state.performance) avatar?.rest();
      }
      if (state.seekVersion !== seekRef.current) { timeRef.current = state.seek; seekRef.current = state.seekVersion; }
      if (state.performance && avatar) {
        if (state.playing) timeRef.current += dt * state.speed;
        if (timeRef.current >= state.performance.result.duration) {
          timeRef.current = state.performance.result.duration;
          if (state.playing) state.onEnd();
        }
        avatar.sample(state.performance, timeRef.current);
        if (now - lastUi > 80) { state.onTime(timeRef.current); lastUi = now; }
      }
      if (helper) { helper.visible = state.skeleton; helper.updateMatrixWorld(true); }
      controls.update(); renderer.render(scene, camera);
      frame = requestAnimationFrame(render);
      } catch (error) {
        latest.current.onError(`Avatar playback failed: ${error instanceof Error ? error.message : String(error)}`);
      }
    };
    frame = requestAnimationFrame(render);
    if (props.bodyReady) {
      getRig().then(rig => {
        if (disposed) return;
        avatar = new NativeAvatar(rig, renderer.capabilities.maxTextureSize);
        avatarRef.current = avatar;
        scene.add(avatar.mesh);
        helper = new THREE.SkeletonHelper(avatar.mesh);
        (helper.material as THREE.LineBasicMaterial).depthTest = false;
        scene.add(helper);
        latest.current.onReady();
      }).catch(error => { if (!disposed) latest.current.onError(error.message); });
    }
    const contextLost = (event: Event) => { event.preventDefault(); latest.current.onError('The GPU context was lost. Reload the page to restore the avatar.'); };
    renderer.domElement.addEventListener('webglcontextlost', contextLost);
    return () => {
      disposed = true; cancelAnimationFrame(frame); observer.disconnect(); controls.dispose(); avatar?.dispose();
      if (helper) { helper.geometry.dispose(); (helper.material as THREE.Material).dispose(); }
      renderer.domElement.removeEventListener('webglcontextlost', contextLost);
      renderer.dispose(); renderer.domElement.remove(); avatarRef.current = null;
    };
  }, [props.bodyReady]);

  return <div className="canvas-stage" ref={mount} />;
}
