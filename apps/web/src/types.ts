export type Rig = {
  version: 'smplx-rig-v1'; vertices: number[]; faces: number[]; joints: number[][];
  parents: number[]; weights: number[]; vertexCount: number; jointCount: number;
};
export type Translation = {
  schema: 'motion133-v1'; text: string; fps: number; duration: number; cached: boolean;
  review: 'unreviewed'; metadata: { generationSeconds: number; device: string; retrievalKeywords: string[]; seed: number; revision: string;
    mode?: 'phrase' | 'research'; glosses?: string[]; segments?: {gloss: string; start: number; end: number}[]; planSource?: string };
  rendering: { rotations: number[][]; frames: number; vertexCount: number; positionBytes: number };
  assets: { positions: string; rawMotion: string };
};
export type Job = { id: string; status: 'queued' | 'loading' | 'generating' | 'preparing' | 'complete' | 'failed'; message: string; result: Translation | null };
export type Health = { ready: boolean; modelLoaded: boolean; bodyReady: boolean; shardsReady: number; downloadedBytes: number; modelBytes: number };
export type Performance = { result: Translation; positions: Float32Array };
