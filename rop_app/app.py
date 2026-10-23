"""ROP screening app: upload retinal photos of one eye and get a prediction.

Run from the project folder:
    streamlit run rop_app/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find model_utils however the app is launched
from model_utils import decode_image, load_config, load_model, predict, preprocess

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "rop_detection_model.pth"
DEFAULT_CONFIG = ROOT / "models" / "rop_model_config.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="ROP Screening", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_model(weights_path, modified_time):
    # modified_time is part of the cache key, so a newly downloaded model file is picked up
    return load_model(weights_path, DEVICE)


st.title("ROP Screening")
st.caption("Upload retinal photos of **one eye from one visit**. The model checks each photo for signs of "
           "Retinopathy of Prematurity (ROP).")
st.warning("Research prototype, not a medical device. Every result must be reviewed by an ophthalmologist.")

# --- Sidebar: settings ----------------------------------------------------------------------
config = load_config(DEFAULT_CONFIG)
with st.sidebar:
    st.header("Settings")
    weights_path = Path(st.text_input("Model file", str(DEFAULT_WEIGHTS)))
    threshold = st.slider(
        "ROP threshold", 0.0, 1.0, config["threshold"], 0.01,
        help="A photo scoring at or above this value counts as ROP. Lower = catches more ROP but more false alarms.",
    )
    if config["from_file"]:
        st.caption(f"Default threshold {config['threshold']:.3f} and image size {config['img_size']} "
                   f"loaded from `{DEFAULT_CONFIG.name}`.")
    else:
        st.caption(f"`{DEFAULT_CONFIG.name}` not found: using threshold {config['threshold']} "
                   f"and image size {config['img_size']}.")
    show_processed = st.checkbox("Show preprocessed images", value=False,
                                 help="The enhanced version of each photo that the model actually sees.")
    st.caption(f"Running on: {DEVICE.upper()}")

if not weights_path.exists():
    st.error(f"Model file not found: `{weights_path}`\n\n"
             "Download `rop_detection_model.pth` and `rop_model_config.json` from your Kaggle notebook's "
             "**Output** panel and put them in the `models` folder (or enter the location in the sidebar).")
    st.stop()

try:
    model = get_model(str(weights_path), weights_path.stat().st_mtime)
except Exception as exc:  # wrong or corrupted file
    st.error(f"Could not load the model from `{weights_path}`: {exc}")
    st.stop()

# --- Upload ---------------------------------------------------------------------------------
uploads = st.file_uploader("Retinal photos", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                           accept_multiple_files=True)
if not uploads:
    st.info("Add one or more photos to get a prediction.")
    st.stop()

names, originals, processed = [], [], []
for upload in uploads:
    img = decode_image(upload.getvalue())
    if img is None:
        st.warning(f"Skipped `{upload.name}`: not a readable image.")
        continue
    names.append(upload.name)
    originals.append(img[:, :, ::-1])  # BGR -> RGB for display
    processed.append(preprocess(img, config["img_size"]))

if not names:
    st.stop()

with st.spinner(f"Analysing {len(names)} photo(s)..."):
    probs = predict(model, processed, DEVICE)

# --- Overall result (same logic as the notebook's exam-level evaluation) -------------------
flagged = [p >= threshold for p in probs]
n_flagged = sum(flagged)
highest = max(probs)

st.subheader("Result for this eye")
if n_flagged:
    st.error(f"**ROP suspected**: {n_flagged} of {len(probs)} photo(s) show signs of ROP "
             f"(highest score {highest:.2f}). Refer to an ophthalmologist.")
else:
    st.success(f"**No ROP detected** in {len(probs)} photo(s) (highest score {highest:.2f}, threshold {threshold:.2f}).")
st.caption("The eye counts as ROP if **any** photo scores at or above the threshold. Mild ROP is often only "
           "visible at the edge of the retina, so upload all photos from the visit.")

# --- Per-photo details ----------------------------------------------------------------------
st.subheader("Per photo")
table = pd.DataFrame({
    "Photo": names,
    "ROP score": probs,
    "Result": ["ROP" if f else "Normal" for f in flagged],
}).sort_values("ROP score", ascending=False)
st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={"ROP score": st.column_config.ProgressColumn("ROP score", min_value=0.0, max_value=1.0, format="%.2f")},
)

cols = st.columns(3)
for i, (name, prob, is_rop) in enumerate(zip(names, probs, flagged)):
    with cols[i % 3]:
        image = processed[i] if show_processed else originals[i]
        st.image(image, caption=f"{name}: {'ROP' if is_rop else 'Normal'} ({prob:.2f})", width="stretch")
