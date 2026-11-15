import asyncio
import io
import os
import numpy as np
import onnxruntime as ort
import requests
import streamlit as st
from PIL import Image, ImageFilter
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

MODEL_URL = "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model_quantized.onnx"
MODEL_PATH = "/tmp/rmbg_1_4_quantized.onnx"
_SESSION = None


def get_api_key() -> str:
    try:
        return st.secrets.get("API_KEY", os.environ.get("API_KEY", "akhil-rmbg-secret"))
    except Exception:
        return os.environ.get("API_KEY", "akhil-rmbg-secret")


def get_session():
    global _SESSION
    if _SESSION is None:
        if not os.path.exists(MODEL_PATH):
            with requests.get(MODEL_URL, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        _SESSION = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _SESSION


def process_rmbg(image_bytes: bytes, erode_px: int = 1, feather_px: float = 0.5) -> bytes:
    sess = get_session()
    input_name = sess.get_inputs()[0].name

    orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = orig.size

    resized = orig.resize((1024, 1024), Image.Resampling.BILINEAR)
    arr = (np.asarray(resized, dtype=np.float32) / 255.0) - 0.5
    inp = np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)

    pred = sess.run(None, {input_name: inp})[0][0, 0]
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    mask = Image.fromarray((pred * 255).astype(np.uint8), mode="L")
    mask = mask.resize((orig_w, orig_h), Image.Resampling.BILINEAR)

    if erode_px > 0:
        mask = mask.filter(ImageFilter.MinFilter(erode_px * 2 + 1))
    if feather_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_px))

    result = orig.convert("RGBA")
    result.putalpha(mask)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


async def remove_bg_endpoint(request: Request):
    if request.headers.get("X-API-Key") != get_api_key():
        return JSONResponse({"error": "Invalid API Key"}, status_code=401)

    form = await request.form()
    upload = form.get("file")
    if upload is None:
        return JSONResponse({"error": "Send image in 'file' form field"}, status_code=400)

    img_bytes = await upload.read()
    erode_px = int(form.get("erode", 1))
    feather_px = float(form.get("feather", 0.5))

    # Run CPU inference in a worker thread so Streamlit health checks stay responsive
    png_bytes = await asyncio.to_thread(process_rmbg, img_bytes, erode_px, feather_px)
    return Response(content=png_bytes, media_type="image/png")


app = st.App(
    "ui.py",
    routes=[Route("/api/remove-bg", remove_bg_endpoint, methods=["POST"])],
)
