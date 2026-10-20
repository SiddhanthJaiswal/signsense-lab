# SignSense Lab

Local English-to-ASL research demo with a native 55-joint Three.js signer.

The active pipeline uses SOKE. The previous RoboSTAR checkpoint produced backward
arm poses and repeated body motion across unrelated sentences; it is no longer the
active provider. Old cached output is excluded by a new model/pipeline cache key.

`Hi! How are you?`, `Hi,hOW ARE YOU?`, and `Hello, how are you?` use the same explicit
`HELLO → HOW → YOU` phrase plan. The signs are decoded from SOKE's published retrieval
memory with its learned body/hand codebooks. A documented wrist constraint aims YOU
at the front-facing addressee; quaternion transitions join the clips. Other sentences
use SOKE's whole-sentence generator. This limited greeting path is not a general ASL
grammar engine. All output, particularly facial grammar and free-form sentences,
still needs fluent ASL review.

## Run

Requires Node 22.18+, `uv`, `curl`, Python 3.10 (installed by setup), approximately
6 GB free disk space for the new model bundle and range cache, and preferably an
Apple Silicon or CUDA computer with at least 16 GB memory.

```sh
sh scripts/setup.sh
npm run dev
```

Open http://127.0.0.1:5173. API: http://127.0.0.1:8000. The first request loads the
model; subsequent requests reuse it. Inference runs locally and needs no paid API.

To resume or verify the active model download:

```sh
.venv/bin/python scripts/download_soke.py
```

Setup fetches only inference tensor ranges from the pinned community checkpoint,
checks each storage's CRC, then checks SHA-256 of the assembled safetensors and all
support files. It does not execute or unpickle the source checkpoint. Numeric model
manifests live in `scripts/manifests/`; downloaded assets and runtime output are
ignored by Git. Previously downloaded RoboSTAR files are retained but unused.

## Git upload and setup on another computer

Commit the source, `scripts/manifests/`, and dependency lockfiles. `.gitignore`
excludes `models/`, generated motion in `runtime/`, dependencies, and checkpoints.
The manifests contain tensor metadata and checksums, not model weights.

After creating an empty GitHub repository, run from this project folder (replace
the URL with your own):

```sh
git add .
git diff --cached --stat
git commit -m "Add SignSense demo and reproducible model setup"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

On another Mac or Linux computer, install Node and `uv`, then run:

```sh
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
sh scripts/setup.sh
npm run dev
```

Setup installs dependencies and downloads both the SOKE inference bundle and the
SMPL-X body asset into the ignored `models/` directory. Windows users should use
WSL2 for these shell commands; native Windows startup is not configured.

For an existing checkout after a pull:

```sh
git pull
sh scripts/setup.sh
npm run dev
```

Valid downloaded model files are reused. If a download is interrupted, rerun setup;
the large SOKE download resumes from completed chunks. To fetch only the assets
after dependencies are installed:

```sh
.venv/bin/python scripts/download_support.py
.venv/bin/python scripts/download_soke.py
```

## Pipeline

```text
English → validated API job → phrase planner / SOKE sentence generator
        → matching learned VQ codebooks and decoders → native motion133
        → phrase spatial constraint and quaternion transitions (greetings only)
        → SMPL-X hand means + pose/face correctives
        → Three.js native skeleton + all 55 skinning weights → playback
```

There is no Mixamo retargeting, left/right mirroring, or hand-mean subtraction at
render time. The native body adapter is tested against the original SMPL-X library.
A narrow anatomical check also prevents the previous backward-arm failure from
playing, including from disk caches. That check is not an ASL accuracy score.

See [architecture](docs/architecture.md) and [verification](docs/verification.md).

## Sources

- [SOKE official implementation](https://github.com/2000ZRL/SOKE), commit
  `5cbc55d84b5a7cbf05a9cf020c468052e8d94d00`: architecture, vocabulary, normalization,
  and retrieval memory.
- [Community SOKE checkpoint](https://huggingface.co/1-13-am/soke-ckpt), revision
  `1385f54f5324c3cb9abaf0a3b8bfcab97b2a0ef2`, epoch 94. This is a community upload,
  not an assertion of official weight provenance or an unrestricted weight license.
- [RoboSTAR source](https://github.com/zyjOrz/RoboSTAR), commit
  `2371ba3e7cdd57242b4a96709a977eabc298e940`, Apache-2.0, retained under `vendor/robostar`.
  Its generic temporal convolution decoder is reused with SOKE's matching learned
  codebooks; RoboSTAR's FSQ token IDs and generator weights are not used.
- [ASLU greeting reference](https://www.lifeprint.com/asl101/pages-signs/h/how.htm):
  phrase order and second-person pointing direction.
- [SMPL-X](https://smpl-x.is.tue.mpg.de/), local neutral asset from the public
  [Anicrafter research mirror](https://huggingface.co/MyNiuuu/Anicrafter_release/blob/main/pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz).
  Original non-commercial body-model license is retained beside the asset. SHA-256:
  `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`.
- Three.js, React, Vite, FastAPI, PyTorch, Transformers.

Research assets remain local; do not redistribute the body or derived rig. This is
not a hosted deployment. The service binds to loopback and rejects external origins.

## Verification

```sh
npm test
npm run build
.venv/bin/python scripts/verify_when_ready.py # running API required
.venv/bin/python scripts/evaluate.py          # produces a blank human review sheet
```

The automated report checks the exact reported greeting, binary sizes, finite and
changing poses, native hand indexing, forward pointing, and cache reuse. Numerical
checks cannot establish semantic accuracy. Free-form generation remains experimental;
review meaning, handshape, location, movement, orientation and nonmanual markers.
