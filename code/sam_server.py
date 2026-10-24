"""Masks for named objects in a camera frame: OWLv2 boxes from text prompts, then SAM 2.1 masks.

Runs on a GPU and is reached from the laptop through an ssh tunnel. The model calls follow
the perception server of my RoboDojo agent (https://mostlykiguess.github.io/robodojo/).

    python sam_server.py --port 23101

POST /segment {"image_jpeg_base64": str, "prompts": {name: text}, "max_box_fraction": float}
  -> {"objects": {name: {"score", "box", "mask_png_base64"}}, "seconds"}. A name is left out
     when no box within max_box_fraction of the image was found for it.
GET /health -> {"ok": true, "requests": int, "last_request_age_s": float}, for the watchdog.
"""

import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor, Sam2Model, Sam2Processor

DETECTOR = "google/owlv2-large-patch14-ensemble"
SEGMENTER = "facebook/sam2.1-hiera-large"


class Segmenter:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.detector_processor = Owlv2Processor.from_pretrained(DETECTOR)
        self.detector = Owlv2ForObjectDetection.from_pretrained(DETECTOR).to(device).eval()
        self.segmenter_processor = Sam2Processor.from_pretrained(SEGMENTER)
        self.segmenter = Sam2Model.from_pretrained(SEGMENTER).to(device).eval()
        self.lock = threading.Lock()
        self.requests = 0
        self.last_request = time.time()

    def best_boxes(self, image: Image.Image, texts: list[str], max_area: float) -> dict[int, tuple[float, list[float]]]:
        """The highest-scoring box per prompt, among boxes no larger than max_area pixels."""
        inputs = self.detector_processor(text=[texts], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.detector(**inputs)
        # OWLv2 pads the image to a square, so boxes are scaled by the padded side.
        side = max(image.height, image.width)
        result = self.detector_processor.post_process_grounded_object_detection(
            outputs, threshold=0.0, target_sizes=torch.tensor([[side, side]]))[0]
        best: dict[int, tuple[float, list[float]]] = {}
        for score, box, label in zip(result["scores"].cpu(), result["boxes"].cpu(), result["labels"].cpu()):
            box = [float(value) for value in box]
            if (box[2] - box[0]) * (box[3] - box[1]) > max_area:
                continue
            label, score = int(label), float(score)
            if label not in best or score > best[label][0]:
                best[label] = (score, box)
        return best

    def masks(self, image: Image.Image, boxes: list[list[float]]) -> np.ndarray:
        inputs = self.segmenter_processor(images=image, input_boxes=[boxes], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.segmenter(**inputs, multimask_output=False)
        masks = self.segmenter_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        return masks.reshape(len(boxes), image.height, image.width).numpy().astype(np.uint8)

    def segment(self, request: dict) -> dict:
        started = time.time()
        # PIL, not OpenCV: opencv-python pulled NumPy 2 onto the instance and broke its torch.
        image = Image.open(io.BytesIO(base64.b64decode(request["image_jpeg_base64"]))).convert("RGB")
        names = list(request["prompts"])
        texts = [request["prompts"][name] for name in names]
        max_area = float(request.get("max_box_fraction", 0.3)) * image.width * image.height
        with self.lock:
            self.requests += 1
            self.last_request = time.time()
            best = self.best_boxes(image, texts, max_area)
            labels = sorted(best)
            masks = self.masks(image, [best[label][1] for label in labels]) if labels else []
        objects = {}
        for label, mask in zip(labels, masks):
            png = io.BytesIO()
            Image.fromarray((mask * 255).astype(np.uint8)).save(png, format="PNG")
            score, box = best[label]
            objects[names[label]] = {"score": round(score, 4), "box": [round(value, 1) for value in box],
                                     "mask_png_base64": base64.b64encode(png.getvalue()).decode()}
        return {"objects": objects, "seconds": round(time.time() - started, 3)}


def make_handler(segmenter: Segmenter):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path != "/health":
                self._reply(404, {"error": "unknown path"})
                return
            self._reply(200, {"ok": True, "requests": segmenter.requests,
                              "last_request_age_s": round(time.time() - segmenter.last_request, 1)})

        def do_POST(self) -> None:
            if self.path != "/segment":
                self._reply(404, {"error": "unknown path"})
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._reply(200, segmenter.segment(request))
            except Exception as error:
                self._reply(500, {"error": f"{type(error).__name__}: {error}"})

        def log_message(self, *args) -> None:
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=23101)
    args = parser.parse_args()
    segmenter = Segmenter()
    print(f"models loaded, GPU memory {torch.cuda.memory_allocated() / 1e9:.2f} GB, serving on {args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(segmenter)).serve_forever()


if __name__ == "__main__":
    main()
