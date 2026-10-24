"""Download the author's ONNX model from a pinned Git revision, not a pickle checkpoint."""
import base64
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
EXPECTED_SHA256 = "ddc8cae492f0c28e1ccbadb11560de49b436dd4544f4382a361d3b047f02f38f"
REPO = "alihassanml/Smoking-detection-yolo11"
REVISION = "bbef478fad3d6d18e4a18bf02fadce1a09f55e15"


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "ship-smoking-demo"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def get_file(name):
    item = get_json(f"https://api.github.com/repos/{REPO}/contents/{name}?ref={REVISION}")
    blob = get_json(item["git_url"])
    if blob.get("encoding") != "base64":
        raise RuntimeError("Unexpected GitHub blob encoding")
    data = base64.b64decode(blob["content"], validate=False)
    git_hash = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if git_hash != item["sha"]:
        raise RuntimeError("Downloaded model does not match Git blob hash")
    return data


def main():
    folder = ROOT / "models"
    folder.mkdir(exist_ok=True)
    print("Downloading pinned ONNX model...", flush=True)
    data = get_file("best.onnx")
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("Model SHA-256 verification failed")
    temporary = folder / "smoking.onnx.part"
    temporary.write_bytes(data)
    temporary.replace(folder / "smoking.onnx")
    (folder / "LICENSE.upstream.txt").write_bytes(get_file("LICENSE"))
    provenance = {"repository": f"https://github.com/{REPO}", "revision": REVISION,
                  "original_file": "best.onnx", "sha256": hashlib.sha256(data).hexdigest()}
    (folder / "source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Downloaded {len(data):,} bytes to {folder / 'smoking.onnx'}")


if __name__ == "__main__":
    main()
