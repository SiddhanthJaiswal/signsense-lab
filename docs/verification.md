# Verification record

## Current pipeline: SOKE, 2026-10-20

The active provider has been replaced, not merely guarded. The exact input
`Hi! How are you?` completed through the running API on MPS, using the explicit
HELLO → HOW → YOU phrase plan and matching learned sign decoders.

- 304 frames at 20 fps; 15.2 seconds; 38,212,800-byte native deformation payload.
- Both-wrists-behind fraction: 0.0 (previous model: 1.0).
- YOU central index direction: +Z, measured cosine 1.0 after the explicit wrist
  constraint. Learned shoulder, elbow and finger channels remain unchanged.
- Identical requests reuse the completed job. New pipeline/revision keys exclude
  previous RoboSTAR caches and the earlier uncorrected SOKE greeting.
- 12 Python tests and 2 frontend tests pass. TypeScript and production build pass.
- The active model download script verifies the installed bundle successfully.
- After a service restart, the corrected greeting loads from disk with `cached=true`.
- All seven evaluation sentences complete (greeting, request, negation, time,
  two reversed agent/recipient sentences, and a name). Each has zero measured
  backward-arm frames and changing body motion. Different body durations/trajectories
  replace the old identical-output failure; linguistic equivalence is not established.

In Opera GX, generated the exact sentence, observed timeline progression, paused
and scrubbed the animation, and inspected HELLO at 2.6 s, HOW at 6 s and YOU at
14.2 s. The visible hand poses match the exported native motion; the hands no
longer bend behind the torso, and YOU points toward the viewer. Browser console
had no app/shader error; an unrelated extension message-port error was present.

The machine-readable API report is `runtime/setup-verification.json`. That script
intentionally does not self-certify browser playback; the manual observations
above are separate. `runtime/evaluation` contains contrasting sentence outputs
and a review sheet with linguistic scores left blank.

**Scope:** this repairs the model/rig pipeline and the tested greeting path. It
does not prove general English-to-ASL accuracy. Greeting clips are isolated learned
examples joined by transitions; coarticulation, nonmanual grammar and model
reconstruction quality still need fluent ASL review. Other sentences use free-form
SOKE generation and may omit or alter meaning. The YOU constraint is explicitly
authored from the [ASLU greeting reference](https://www.lifeprint.com/asl101/pages-signs/h/how.htm),
not an improvement attributed to the learned model.


## Historical reproduction: inactive RoboSTAR checkpoint

The exact input `Hi,hOW ARE YOU?` was reproduced on the local MPS service. It
produced 32 frames (1.6 seconds), with both wrists 0.33–0.41 metres behind the
shoulder midpoint throughout the clip in the native body's forward direction.
`Hello, how are you?`, `My name is Alex.`, and `I would like a glass of water.`
produced exactly identical first 30 body channels, although their hand channels
were different. These outputs fail the basic signing demonstration requirement.

Decoding the published retrieval entry WATER with the released tokenizer on CPU
also puts both wrists behind the body throughout the clip (0.34–0.44 metres).
This reproduces the orientation problem without the generator, MPS, Three.js,
GPU skinning, camera, or input capitalization. It does not establish whether the
underlying cause is training data conventions, checkpoint compatibility, or a
problem in the released weights. No speculative axis flip has been applied.

Run `.venv/bin/python scripts/diagnose_motion.py --decode-reference` to regenerate
`runtime/motion-diagnostics.json` from the local cached samples and upstream
reference. This requires installed model/body assets and does not download them.

The API now rejects the specific backward-arm failure before preparing playback,
and rechecks existing cached results. Its deliberately narrow check rejects clips
where both wrists are more than 15 cm behind the chest for over 80% of frames.
It does not certify the ASL accuracy of other output, and could reject a deliberate
long behind-the-back pose. Failed requests return the avatar to its resting pose.

This historical failure motivated the provider replacement above. A camera change
alone could not correct it. General translation accuracy still requires independent
linguistic evaluation.

## Renderer and numerical checks

The native body adapter was compared with `smplx==0.1.28` using two random frames
containing body, finger, jaw, and expression changes:

- Full-pose parameters: exact equality.
- Maximum mesh-vertex difference: `8.404255e-06` metres.
- All 55 skin weights retained; row sums within `1.2e-7` of one.

These checks establish agreement with the native mathematical implementation,
not agreement between the research model output and correct ASL. Finite decoder
output, a successful inference request, and a visible avatar are insufficient
acceptance checks. Human review scores remain blank.
