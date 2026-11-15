import io
import os
import threading
import time
import numpy as np
import onnxruntime as ort
import requests
import streamlit as st
from PIL import Image, ImageFilter, ImageOps

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


def process_rmbg(image_bytes: bytes, erode_px: int = 1, feather_px: float = 0.5) -> bytes:
    sess = get_session()
    input_name = sess.get_inputs()[0].name

    orig = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
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


def bot_worker(token: str):
    base = f"https://api.telegram.org/bot{token}"
    offset = None
    while True:
        try:
            params = {"timeout": 25, "allowed_updates": ["message"]}
            if offset is not None:
                params["offset"] = offset
            res = requests.get(f"{base}/getUpdates", params=params, timeout=35).json()
            if not res.get("ok"):
                time.sleep(3)
                continue

            for upd in res.get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or {}
                chat_id = msg.get("chat", {}).get("id")
                if not chat_id:
                    continue

                file_id = None
                if msg.get("photo"):
                    file_id = msg["photo"][-1]["file_id"]
                elif msg.get("document") and str(msg["document"].get("mime_type", "")).startswith("image/"):
                    file_id = msg["document"]["file_id"]

                if not file_id:
                    continue

                requests.post(f"{base}/sendChatAction", json={"chat_id": chat_id, "action": "upload_document"})
                file_info = requests.get(f"{base}/getFile", params={"file_id": file_id}, timeout=20).json()
                file_path = file_info["result"]["file_path"]
                img_bytes = requests.get(f"https://api.telegram.org/file/bot{token}/{file_path}", timeout=60).content

                png_bytes = process_rmbg(img_bytes)

                # Must use sendDocument so Telegram preserves the transparent PNG alpha channel
                requests.post(
                    f"{base}/sendDocument",
                    data={"chat_id": chat_id, "caption": "HD Transparent PNG (RMBG-1.4)"},
                    files={"document": ("nobg.png", png_bytes, "image/png")},
                    timeout=60,
                )
        except Exception as e:
            print("Worker error:", e)
            time.sleep(3)


@st.cache_resource
def start_background_worker(token: str):
    t = threading.Thread(target=bot_worker, args=(token,), daemon=True)
    t.start()
    return True


st.title("RMBG-1.4 Cloud Worker")
token = st.secrets.get("BOT_TOKEN", "")
if token:
    start_background_worker(token)
    st.success("Worker active! Send any photo or image file to your Telegram bot.")
else:
    st.error("Add BOT_TOKEN to Streamlit Secrets to start the worker.")
