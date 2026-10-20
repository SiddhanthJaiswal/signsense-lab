"""Download the pinned, public RoboSTAR artifacts; resume incomplete downloads."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = "Ivystream/RoboSTAR"
REVISION = "bf0ed2f3bf43b8a1d9def85ca8ab51bef5a10b06"

def curl(url, *args):
    return subprocess.check_output(["curl", "--http1.1", "--fail", "--location", "--silent", "--show-error", "--retry", "5", "--retry-all-errors", *args, url])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    target = ROOT / "models" / "robostar"
    target.mkdir(parents=True, exist_ok=True)
    files = json.loads(curl(f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=true"))
    def download(item):
        name = item["path"]
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get("lfs", {}).get("oid")
        if path.exists() and path.stat().st_size == item["size"]:
            if not expected or sha256(path) == expected:
                print(f"Ready: {name}", flush=True)
                return
        partial = path.with_name(path.name + ".part")
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
        if item["size"] > 16 * 1024 * 1024:
            # Bounded range requests survive proxies that cancel long transfers.
            # Existing whole-file prefixes are retained; chunks resume independently.
            prefix = partial.stat().st_size if partial.exists() else 0
            chunk_dir = target / ".chunks" / name
            chunk_dir.mkdir(parents=True, exist_ok=True)
            size = 8 * 1024 * 1024
            ranges = [(start, min(start + size, item["size"]) - 1) for start in range(prefix, item["size"], size)]
            def part(bounds):
                start, end = bounds
                output = chunk_dir / str(start)
                if output.exists() and output.stat().st_size == end - start + 1:
                    return output
                for attempt in range(5):
                    curl(url, "--range", f"{start}-{end}", "--max-time", "180", "--output", str(output))
                    if output.stat().st_size == end - start + 1:
                        return output
                    time.sleep(attempt + 1)
                raise RuntimeError(f"Incomplete range {name}:{start}-{end}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as chunks:
                outputs = list(chunks.map(part, ranges))
            with partial.open("ab") as handle:
                for output in outputs:
                    with output.open("rb") as source:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            handle.write(chunk)
            for output in outputs:
                output.unlink()
        else:
            curl(url, "--output", str(partial))
        if partial.stat().st_size != item["size"]:
            raise RuntimeError(f"Size mismatch: {name}")
        if expected and sha256(partial) != expected:
            raise RuntimeError(f"SHA256 mismatch: {name}")
        partial.replace(path)
        print(f"Downloaded: {name}", flush=True)
    items = sorted([x for x in files if x["type"] == "file" and x["path"] != ".gitattributes"], key=lambda x: x['size'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, items))
    (target / "revision.json").write_text(json.dumps({"repository": REPO, "revision": REVISION}, indent=2) + "\n")

if __name__ == "__main__":
    main()
