"""Native body asset used by the active SOKE pipeline."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BODY_SHA256 = "376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992"
BODY_URL = "https://huggingface.co/MyNiuuu/Anicrafter_release/resolve/7f9fcea/pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz"

def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_name(path.name + ".part")
        subprocess.run(["curl", "--http1.1", "--fail", "--location", "--silent", "--show-error", "--retry", "5", "--retry-all-errors",
                        "--continue-at", "-", "--output", str(partial), url], check=True)
        partial.replace(path)
    print(f"Ready: {path.relative_to(ROOT)}", flush=True)

def main():
    body = ROOT / "models/smplx/SMPLX_NEUTRAL.npz"
    download(BODY_URL, body)
    if hashlib.sha256(body.read_bytes()).hexdigest() != BODY_SHA256:
        raise RuntimeError("Body model checksum mismatch")
    download("https://raw.githubusercontent.com/vchoutas/smplx/main/LICENSE", body.parent / "LICENSE.txt")
    (body.parent / "PROVENANCE.json").write_text(json.dumps({"url": BODY_URL, "sha256": BODY_SHA256,
        "license": "SMPL-X non-commercial research license; not the mirror repository's code license", "redistributable": False}, indent=2))

if __name__ == "__main__":
    main()
