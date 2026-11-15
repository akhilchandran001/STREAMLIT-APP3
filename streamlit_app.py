import io
import json
import os
import threading
import time
import numpy as np
import onnxruntime as ort
import requests
import streamlit as st
from PIL import Image, ImageFilter, ImageOps

RELAY_TOPIC = "akhil-rmbg-bridge-8976"

MODEL_URL = "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model_quantized.onnx"
MODEL_PATH = "/tmp/rmbg_1_4_quantized.onnx"
_SESSION = None


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


def process_rmbg_mask(image_bytes: bytes, erode_px: int = 1, feather_px: float = 0.5) -> tuple[bytes, str]:
    sess = get_session()
    input_name = sess.get_inputs()[0].name

    orig = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    resized = orig.resize((1024, 1024), Image.Resampling.BILINEAR)
    arr = (np.asarray(resized, dtype=np.float32) / 255.0) - 0.5
    inp = np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)

    pred = sess.run(None, {input_name: inp})[0][0, 0]
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    mask = Image.fromarray((pred * 255).astype(np.uint8), mode="L")

    if erode_px > 0:
        mask = mask.filter(ImageFilter.MinFilter(erode_px * 2 + 1))
    if feather_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_px))

    # Encode 1-channel mask (~40-120 KB) so ntfy.sh never rejects the return payload
    buf = io.BytesIO()
    mask.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()
    if len(data) > 400_000:
        buf = io.BytesIO()
        mask.save(buf, format="JPEG", quality=95)
        return buf.getvalue(), "jpg"
    return data, "png"


def relay_worker(topic: str):
    seen_jobs = set()
    url = f"https://ntfy.sh/{topic}/json"

    while True:
        try:
            with requests.get(url, stream=True, timeout=75) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line.decode("utf-8"))
                    if event.get("event") != "message":
                        continue

                    title = event.get("title", "")
                    attachment = event.get("attachment") or {}
                    file_url = attachment.get("url")

                    if not title.startswith("req:") or not file_url:
                        continue

                    job_id = title.split(":", 1)[1]
                    if job_id in seen_jobs:
                        continue
                    seen_jobs.add(job_id)

                    img_bytes = requests.get(file_url, timeout=60).content
                    mask_bytes, ext = process_rmbg_mask(img_bytes)

                    put_res = requests.put(
                        f"https://ntfy.sh/{topic}",
                        data=mask_bytes,
                        headers={
                            "Filename": f"mask_{job_id}.{ext}",
                            "Title": f"done:{job_id}",
                        },
                        timeout=60,
                    )
                    put_res.raise_for_status()
        except Exception as e:
            print("Relay reconnect:", e)
            time.sleep(2)


@st.cache_resource
def start_worker(topic: str):
    t = threading.Thread(target=relay_worker, args=(topic,), daemon=True)
    t.start()
    return True


st.title("RMBG-1.4 HTTP Relay Worker")
start_worker(RELAY_TOPIC)
st.success("Relay worker active! Ready for rb.py requests.")
