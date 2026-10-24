# Model weights

Run `python download_model.py` from the project root to download the pinned upstream model as `models/smoking.onnx`. The downloader verifies the expected SHA-256 and records model provenance in `source.json`.

Do not replace it with a generic COCO person detector: this demo requires the smoking model's `Smooking` class. Model binaries and local download metadata are excluded from Git.
