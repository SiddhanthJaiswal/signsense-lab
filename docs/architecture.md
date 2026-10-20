# Architecture: first iteration

The user-facing scope is one text input and one rigged performer. No accounts, microphone,
database, recognition, lesson system, or admin dashboard is included.

## Boundaries

| Part | Responsibility | Does not own |
| --- | --- | --- |
| React UI | Input, job state, playback controls, error messages | Grammar or motion synthesis |
| FastAPI | Request validation, bounded queue, cache, asset URLs | Three.js or guessed translations |
| Phrase planner | Exact greeting variants → HELLO / HOW / YOU; reject partial matches | General ASL grammar |
| SOKE provider | Matching learned codebooks, retrieval memory and multilingual sentence inference | Avatar geometry |
| Phrase constraints | Aim the greeting’s YOU at the addressee; preserve learned fingers and arms | Other sentences |
| Native body adapter | Decode motion133, native hand means, face and pose basis | Language interpretation |
| Three.js player | Native bones, full-weight skinning, interpolation, lighting, camera | Invented signs or spelling fallbacks |

## Motion contract

`motion133-v1` is a versioned numerical contract, distinct from COCO's 133 keypoints:

| Range | Meaning |
| --- | --- |
| 0–29 | Rotation vectors for SMPL-X joints 12–21: neck, collars, head, shoulders, elbows, wrists |
| 30–74 | 15 left-hand rotation vectors, index/middle/pinky/ring/thumb order |
| 75–119 | 15 right-hand rotation vectors in the same order |
| 120–122 | Jaw rotation vector |
| 123–132 | Ten SMPL-X facial-expression coefficients |

The omitted lower body/root rotation channels are zero, following SOKE. Native hand means
are added exactly once, matching its non-flat-hand configuration. Native rotations are
converted to quaternions and interpolated with SLERP in the browser. There is no mirroring
or Euler-angle reinterpretation.

For each frame, the server computes `v_posed = v_shaped + pose_basis(R-I) + expression_basis(e)`.
The browser uses the original 55-joint hierarchy, bind inverses and **all** skin weights
to apply linear blend skinning. A float texture stores the full weights; the standard
four-influence limit is not used. This requires WebGL2 and a maximum texture size of at
least 10,475 on the demo's GPU. Raw motion remains downloadable for comparison with an
independent native renderer.

## API

| Endpoint | Contract |
| --- | --- |
| `GET /api/health` | Local model/body readiness and download progress |
| `GET /api/rig` | Original native mesh, joints, parents and weights; local licensed asset |
| `POST /api/translations` | `{text, seed?}` → job ID and status; max 400 characters / 60 words |
| `GET /api/translations/{id}` | queued/loading/generating/preparing/complete/failed |
| `GET /api/assets/{hash}/positions.bin` | Little-endian float32 `[frame, vertex, xyz]` pose/face-corrected positions |
| `GET /api/assets/{hash}/motion133.npy` | Original model output for evaluation |

A single worker owns the model. Decoding is greedy; the seed is retained in metadata and the request contract. Up to three jobs can be outstanding.
Identical requests reuse jobs and cached results. Cache keys include the sentence, seed,
model revision, phrase/constraint pipeline version, body adapter and schema version. Failed inference is reported explicitly.
No paid service or fabricated motion silently takes over.

## Evaluation and future replacements

1. Prove real inference returns finite, changing native motion and a playable result.
2. Compare native playback to an independent SMPL-X renderer; verify left/right hands,
   rest pose, hand means, face coefficients and timing.
3. Have fluent ASL reviewers judge meaning and grammar on a fixed sentence suite.
4. Record model failures separately from rendering failures.
5. If another generator performs better, implement its provider and convert its output
   into a documented native format; keep the UI and job flow intact.

The model's research status must remain visible until linguistic evaluation exists.
Automatic finite-value checks, test counts and animation smoothness are not measures of
ASL translation accuracy.

## Deployment

Currently loopback-only. A future GPU service could host inference and return compatible
motion assets after the respective licenses permit that use. Browser assets and API
credentials should stay separate, with keys held server-side. Do not deploy the current
licensed model/rig endpoint publicly. A Cloudflare Worker is not a suitable host for the
PyTorch inference process.

## Greeting path

Case and punctuation normalize only for exact phrase matching. Extended sentences such as “How are you feeling?” go to the research generator; they are never truncated to the greeting. Published SOKE code streams decode to native motions with matching VQ weights and normalization. YOU receives an explicit wrist-only aim constraint toward +Z during its central hold, preserving the learned finger shape and shoulder/elbow trajectories. Eight-frame quaternion transitions include the native hand means before interpolation. Metadata lists glosses, time segments, source and constraint; the result remains unreviewed.
