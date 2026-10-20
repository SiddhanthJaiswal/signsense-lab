import { useEffect, useRef, useState } from 'react';
import { Stage } from './avatar/Stage';
import { getHealth, getJob, startTranslation } from './api';
import type { Health, Job, Performance, Translation } from './types';

const examples = ['Hi! How are you?', 'How are you?', 'Hello!'];
const wait = (ms: number, signal: AbortSignal) => new Promise<void>((resolve, reject) => {
  const onAbort = () => { clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')); };
  const timer = setTimeout(() => { signal.removeEventListener('abort', onAbort); resolve(); }, ms);
  signal.addEventListener('abort', onAbort, { once: true });
});
const seconds = (time: number) => `${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, '0')}`;

export function App() {
  const [text, setText] = useState('Hi! How are you?');
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [performance, setPerformance] = useState<Performance | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(0.75);
  const [time, setTime] = useState(0);
  const [seek, setSeek] = useState(0);
  const [seekVersion, setSeekVersion] = useState(0);
  const [camera, setCamera] = useState(0);
  const [skeleton, setSkeleton] = useState(false);
  const [avatarReady, setAvatarReady] = useState(false);
  const [error, setError] = useState('');
  const [avatarError, setAvatarError] = useState('');
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    let pending = false;
    const poll = async () => {
      if (pending) return;
      pending = true;
      try { const next = await getHealth(); if (alive.current) { setHealth(next); setOffline(false); } }
      catch { if (alive.current) setOffline(true); }
      finally { pending = false; }
    };
    void poll();
    const timer = setInterval(poll, 4000);
    return () => { alive.current = false; clearInterval(timer); abort.current?.abort(); };
  }, []);

  const jump = (value: number) => { setSeek(value); setTime(value); setSeekVersion(n => n + 1); };
  const replay = () => { jump(0); setPlaying(true); };
  const loadPerformance = async (result: Translation, signal: AbortSignal) => {
    const response = await fetch(result.assets.positions, { signal });
    if (!response.ok) throw new Error('The motion file could not be loaded. Generate the sentence again.');
    const buffer = await response.arrayBuffer();
    const expected = result.rendering.frames * result.rendering.vertexCount * 3 * 4;
    if (buffer.byteLength !== expected || buffer.byteLength !== result.rendering.positionBytes) throw new Error('The motion file is incomplete.');
    const positions = new Float32Array(buffer);
    if (!positions.every(Number.isFinite)) throw new Error('The motion contains invalid positions.');
    if (result.schema !== 'motion133-v1' || result.rendering.rotations.length !== result.rendering.frames) throw new Error('Unsupported motion format.');
    setPerformance({ result, positions }); jump(0); setPlaying(true);
  };
  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!text.trim() || busy) return;
    abort.current?.abort();
    const controller = new AbortController(); abort.current = controller;
    setError(''); setBusy(true); setPlaying(false); setPerformance(null); jump(0);
    try {
      let next = await startTranslation(text, controller.signal);
      setJob(next);
      while (!['complete', 'failed'].includes(next.status)) {
        await wait(1200, controller.signal);
        next = await getJob(next.id, controller.signal); setJob(next);
      }
      if (next.status === 'failed' || !next.result) throw new Error(next.message);
      await loadPerformance(next.result, controller.signal);
    } catch (e) {
      if ((e as Error).name !== 'AbortError' && alive.current) setError((e as Error).message);
    } finally { if (alive.current) setBusy(false); }
  };

  const ready = health?.ready && avatarReady && !offline && !avatarError;
  const progress = health ? Math.min(100, Math.floor(health.downloadedBytes / health.modelBytes * 100)) : 0;
  const status = offline ? 'Motion service offline' : health?.ready ? 'Research model ready' : health ? `Preparing model · ${progress}%` : 'Connecting to motion service';

  return <div className="app">
    <header className="site-header">
      <a href="/" className="wordmark" aria-label="SignSense Lab home"><span className="brand-symbol" aria-hidden="true">s.</span>signsense<span className="lab">lab</span></a>
      <span className="edition">English → American Sign Language <span className="edition-index">Experiment 01</span></span>
    </header>
    <main>
      <section className="intro" aria-labelledby="title">
        <p className="eyebrow">From a sentence to a performance</p>
        <h1 id="title">Give your words<br /><em>a little movement.</em></h1>
        <p className="intro-copy">Write a sentence. Watch a rigged signer perform the model’s ASL motion, then slow it down and take a closer look.</p>
      </section>
      <section className="workspace" aria-label="Sentence to sign workspace">
        <div className="composer">
          <div className="section-label"><span>01 / Your sentence</span><span>ENGLISH</span></div>
          <form onSubmit={submit}>
            <label className="sr-only" htmlFor="sentence">Enter an English sentence</label>
            <textarea id="sentence" maxLength={400} rows={5} value={text} onChange={e => setText(e.target.value)}
              placeholder="What would you like to say?" aria-describedby="sentence-help" />
            <div className="input-foot"><span id="sentence-help">Keep it short for this first experiment.</span><span>{text.length}/400</span></div>
            <button className="generate" type="submit" disabled={!ready || !text.trim() || busy}>
              <span>{busy ? 'Preparing your performance…' : 'Generate signing'}</span><span aria-hidden="true">↗</span>
            </button>
          </form>
          <div className="examples"><p>Try a sentence</p>{examples.map(example => <button key={example} disabled={busy} onClick={() => setText(example)}>{example}<span aria-hidden="true">↗</span></button>)}</div>
          <div className="service-state" role="status"><span className={`status-dot ${ready ? 'ready' : ''}`} /><span>{status}</span></div>
          {offline && <p className="setup-note">Start the local services with <code>npm run dev</code>.</p>}
          {health && !health.ready && !offline && <p className="setup-note">The first setup downloads the research model and body asset. You can explore the signer as soon as its rig is ready.</p>}
          <p className="research-note">An ASL research experiment. Generated signing has not been evaluated by a fluent ASL signer.</p>
        </div>
        <div className="viewer">
          <div className="stage-header"><span className="section-label">02 / The performance</span><span className="stage-badge">Native 3D rig</span></div>
          <div className="stage-wrap">
            <Stage performance={performance} playing={playing} speed={speed} seek={seek} seekVersion={seekVersion}
              resetCamera={camera} skeleton={skeleton} bodyReady={!!health?.bodyReady} onTime={setTime}
              onEnd={() => setPlaying(false)} onReady={() => setAvatarReady(true)} onError={setAvatarError} />
            {!avatarReady && !avatarError && <div className="stage-placeholder"><span className="loading-ring" aria-hidden="true" /><p>{health?.bodyReady ? 'Loading the rigged signer' : 'The signer is getting ready'}</p><small>The body, hands and face share one native skeleton.</small></div>}
            {avatarError && <div className="stage-placeholder error" role="alert"><p>{avatarError}</p><button onClick={() => window.location.reload()}>Reload avatar</button></div>}
            <div className="stage-tools"><button onClick={() => setSkeleton(v => !v)} aria-pressed={skeleton} disabled={!avatarReady}>{skeleton ? 'Hide rig' : 'Show rig'}</button><button onClick={() => setCamera(v => v + 1)}>Reset view ↺</button></div>
            <span className="camera-hint">Drag to orbit · Scroll to zoom</span>
          </div>
          <div className="playback">
            <button className="play-button" aria-label={playing ? 'Pause signing' : 'Play signing'} disabled={!performance || busy} onClick={() => time >= (performance?.result.duration ?? 0) ? replay() : setPlaying(p => !p)}>{playing ? 'Ⅱ' : '▶'}</button>
            <button className="replay" aria-label="Replay from beginning" disabled={!performance || busy} onClick={replay}>↺</button>
            <input aria-label="Signing timeline" type="range" min={0} max={performance?.result.duration ?? 1} step={0.02} value={time} disabled={!performance || busy} onChange={e => jump(Number(e.target.value))} />
            <span className="time">{seconds(time)}<span> / {seconds(performance?.result.duration ?? 0)}</span></span>
            <label className="sr-only" htmlFor="speed">Playback speed</label><select id="speed" value={speed} onChange={e => setSpeed(Number(e.target.value))}><option value={0.25}>0.25×</option><option value={0.5}>0.5×</option><option value={0.75}>0.75×</option><option value={1}>1×</option></select>
          </div>
          <div className="performance-caption" aria-live="polite">{error ? <p className="error" role="alert">{error}</p> : busy ? <p><span className="pulse" />{job?.message ?? 'Starting the motion model…'}</p> : performance ? <p>“{performance.result.text}” <span>· {performance.result.metadata.mode === 'phrase' ? performance.result.metadata.glosses?.join(' → ') : 'Research generation · Unreviewed'}</span></p> : <p>Your sentence will appear here when the performance is ready.</p>}</div>
        </div>
      </section>
      <details className="pipeline"><summary>Inside this experiment <span>Translation → motion → native rig → Three.js ↗</span></summary>
        <div className="pipeline-content"><p><strong>Phrase planning and learned motion.</strong> Greetings use a phrase-level ASL plan and SOKE’s learned sign clips, with the pointing direction of YOU corrected toward the viewer. Other sentences use its multilingual research generator. Both drive the same native body, finger and facial rig.</p><p><strong>Accuracy needs a human check.</strong> Free-form model output can omit or alter meaning. Compare the handshape, orientation, location, movement, face and sentence meaning with a fluent signer.</p>
          {performance?.result.metadata.planSource && <p><a href={performance.result.metadata.planSource} target="_blank" rel="noreferrer">ASL greeting reference ↗</a></p>}
          {performance && <p>Generated in {performance.result.metadata.generationSeconds.toFixed(1)}s on {performance.result.metadata.device}. Seed {performance.result.metadata.seed}. {performance.result.cached ? 'Loaded from the local cache.' : 'New inference.'} <a href={performance.result.assets.rawMotion} download>Download raw motion ↗</a></p>}
        </div>
      </details>
    </main>
    <footer><span>SignSense Lab / Personal research demo</span><span>SOKE + SMPL-X + Three.js</span></footer>
  </div>;
}
