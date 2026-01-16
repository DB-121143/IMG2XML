import argparse
import base64
import io
import os
from typing import Dict

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
import yaml
import onnxruntime as ort


class RMBGRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded PNG or JPEG image data")


class RMBGResponse(BaseModel):
    image: str = Field(..., description="Base64-encoded RGBA PNG with background removed")


class RMBGInference:
    def __init__(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RMBG model not found: {model_path}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        sel = [p for p in providers if p in available]
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, providers=sel, sess_options=sess_options)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = (1024, 1024)
        # Log actual provider selection and CUDA visibility
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        print(
            f"[RMBG] visible={visible} available={available} selected={self.session.get_providers()} model={model_path}"
        )

    def _preprocess(self, img: np.ndarray):
        h, w = img.shape[:2]
        img_resized = Image.fromarray(img).resize(self.input_size, Image.BILINEAR)
        arr = np.array(img_resized).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)
        return arr, (h, w)

    def _postprocess(self, pred: np.ndarray, original_size):
        alpha = pred[0, 0, :, :]
        alpha_resized = Image.fromarray((alpha * 255).astype(np.uint8)).resize((original_size[1], original_size[0]), Image.BILINEAR)
        return np.array(alpha_resized)

    def remove_background(self, pil_img: Image.Image) -> Image.Image:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img_np = np.array(pil_img)
        inp, orig_size = self._preprocess(img_np)
        pred = self.session.run([self.output_name], {self.input_name: inp})[0]
        alpha = self._postprocess(pred, orig_size)
        img_rgba = np.dstack([img_np, alpha])
        return Image.fromarray(img_rgba)


def create_app(model_path: str) -> FastAPI:
    infer = RMBGInference(model_path)
    app = FastAPI(title="RMBG Service", version="1.0.0")

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/remove", response_model=RMBGResponse)
    def remove_bg(payload: RMBGRequest) -> RMBGResponse:
        """Sync handler so FastAPI runs it in a threadpool, avoiding event-loop starvation."""
        try:
            img_bytes = base64.b64decode(payload.image)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            rgba = infer.remove_background(pil_img)
            buf = io.BytesIO()
            rgba.save(buf, format="PNG")
            buf.seek(0)
            out_b64 = base64.b64encode(buf.read()).decode("ascii")
            return RMBGResponse(image=out_b64)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start an RMBG HTTP service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=9101, help="Port to bind")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"),
        help="Path to config.yaml",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "rmbg", "model.onnx")
    if "rmbg" in cfg and cfg["rmbg"].get("model_path"):
        model_path = cfg["rmbg"]["model_path"]
    app = create_app(model_path)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
