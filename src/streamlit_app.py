"""
Fashion Search Engine — FashionCLIP + FAISS
"""

import json
import time
import io
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image

import torch
import open_clip
import faiss
from datasets import load_dataset

st.set_page_config(
    page_title="Fashion Search",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def find_artifacts() -> Path:
    candidates = [
        Path("/app/artifacts"),
        Path(__file__).resolve().parent.parent / "artifacts",
        Path(__file__).resolve().parent / "artifacts",
        Path.cwd() / "artifacts",
        Path.cwd().parent / "artifacts",
        Path("/content/artifacts"),
    ]
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_dir() and (resolved / "fashion.index").exists():
            return resolved
    st.error("❌ artifacts/ not found. Checked:")
    for p in candidates:
        st.code(str(p.resolve()))
    st.stop()

ARTIFACTS = find_artifacts()

@st.cache_resource(show_spinner="Loading FashionCLIP model…")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionCLIP')
    tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionCLIP')
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer, device

@st.cache_resource(show_spinner="Loading FAISS index & metadata…")
def load_artifacts_data():
    index      = faiss.read_index(str(ARTIFACTS / "fashion.index"))
    embeddings = np.load(str(ARTIFACTS / "embeddings.npy"))
    with open(ARTIFACTS / "metadata.json") as f:
        metadata = json.load(f)
    return index, embeddings, metadata

@st.cache_resource(show_spinner="Loading fashion dataset…")
def load_hf_dataset():
    return load_dataset("ceyda/fashion-products-small", split="train")

@st.cache_data(show_spinner=False)
def compute_score_calibration(_embeddings: np.ndarray, n_samples: int = 400):
    rng     = np.random.default_rng(42)
    idx     = rng.choice(len(_embeddings), size=min(n_samples, len(_embeddings)), replace=False)
    sample  = _embeddings[idx].astype(np.float32)
    scores  = (sample @ sample.T).flatten()
    p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
    return float(p5), float(p95)

MODALITY_GAP_RATIO = 0.55

def encode_text(text, model, tokenizer, device):
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        vec = model.encode_text(tokens).cpu().numpy().astype(np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return vec

def encode_image(pil_img, model, preprocess, device):
    tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(tensor).cpu().numpy().astype(np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return vec

def calibrate_scores(raw_scores, p5, p95, temperature, is_text=False):
    """
    temperature (0.1–0.5):
      low  (0.1) → sharp peaks, only very close matches score high
      high (0.5) → softer curve, more results in a similar range
    """
    scores = raw_scores.copy()
    if is_text:
        scores /= MODALITY_GAP_RATIO
    # Apply temperature scaling before percentile mapping
    scores = scores / temperature
    p5_t   = p5 / temperature
    p95_t  = p95 / temperature
    return np.clip((scores - p5_t) / (p95_t - p5_t + 1e-8) * 100, 0, 100)

def search_text(query, model, tokenizer, device, index, embeddings, p5, p95, temperature, top_k):
    vec        = encode_text(query, model, tokenizer, device)
    _, I       = index.search(vec, top_k * 3)
    raw_scores = (embeddings[I[0]] @ vec.T).flatten()
    scores     = calibrate_scores(raw_scores, p5, p95, temperature, is_text=True)
    return list(zip(I[0].tolist(), scores.tolist()))

def search_image_query(pil_img, model, preprocess, device, index, embeddings, p5, p95, temperature, top_k):
    vec        = encode_image(pil_img, model, preprocess, device)
    _, I       = index.search(vec, top_k * 3)
    raw_scores = (embeddings[I[0]] @ vec.T).flatten()
    scores     = calibrate_scores(raw_scores, p5, p95, temperature, is_text=False)
    return list(zip(I[0].tolist(), scores.tolist()))

def search_combined(pil_img, text_modifier, model, preprocess, tokenizer, device,
                    index, embeddings, p5, p95, temperature, alpha, top_k):
    img_vec  = encode_image(pil_img, model, preprocess, device)
    txt_vec  = encode_text(text_modifier, model, tokenizer, device)
    blended  = (1 - alpha) * img_vec + alpha * txt_vec
    blended /= np.linalg.norm(blended)
    _, I     = index.search(blended.astype(np.float32), top_k * 5)
    cand     = embeddings[I[0]]
    img_s    = calibrate_scores((cand @ img_vec.T).flatten(), p5, p95, temperature, is_text=False)
    txt_s    = calibrate_scores((cand @ txt_vec.T).flatten(), p5, p95, temperature, is_text=True)
    final    = (1 - alpha) * img_s + alpha * txt_s
    order    = np.argsort(-final)
    return [(int(I[0][i]), float(final[i])) for i in order]

def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()

def fetch_result_items(results, dataset, metadata, top_k):
    items = []
    for idx, score in results:
        if len(items) >= top_k:
            break
        try:
            meta        = metadata[int(idx)]
            dataset_idx = meta.get("dataset_idx", int(idx))
            img         = dataset[dataset_idx]["image"]
            caption     = meta.get("caption", "")
            colour      = meta.get("colour", "")
            article     = meta.get("articleType", "")
            items.append({
                "img_bytes": pil_to_bytes(img),
                "score":     score,
                "caption":   caption,
                "colour":    colour,
                "article":   article,
            })
        except Exception:
            continue
    return items

def score_badge(score):
    if score >= 70: return f"🟢 {score:.0f}%"
    if score >= 40: return f"🟡 {score:.0f}%"
    return f"🔴 {score:.0f}%"

def render_grid(items, n_cols=4):
    rows = [items[i:i+n_cols] for i in range(0, len(items), n_cols)]
    for row in rows:
        cols = st.columns(n_cols)
        for col, item in zip(cols, row):
            with col:
                st.image(item["img_bytes"], width=300)   # fixed width — prevents shaking
                badge  = score_badge(item["score"])
                colour = item["colour"]
                art    = item["article"]
                label  = f"{colour} {art}".strip() if (colour or art) else item["caption"][:40]
                st.caption(f"{badge}  {label}")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main-title  { font-family: 'DM Serif Display', serif; font-size: 2.8rem; letter-spacing: -0.5px; margin-bottom: 0; }
    .sub-title   { font-size: 1rem; color: #888; margin-top: 0; margin-bottom: 2rem; }
    div[data-testid="stImage"] img { border-radius: 8px; }
    .stButton > button { background: #111; color: #fff; border: none; border-radius: 6px; padding: 0.5rem 2rem; font-weight: 500; }
    .stButton > button:hover { background: #333; }
    .result-count { font-size: 0.85rem; color: #aaa; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Load everything ───────────────────────────────────────────────────────────
model, preprocess, tokenizer, device = load_model()
index, embeddings, metadata          = load_artifacts_data()
p5, p95                              = compute_score_calibration(embeddings)
dataset                              = load_hf_dataset()

for key, default in [("result_items", []), ("result_time", 0.0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar — temperature control ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Search Settings")
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=0.5,
        value=0.25,
        step=0.05,
        help=(
            "Controls how strict the scoring is.\n\n"
            "**Low (0.1)** — only very close matches score high, results are precise but fewer.\n\n"
            "**High (0.5)** — softer curve, more results score similarly, better for exploration."
        ),
    )
    st.caption(f"Current: `{temperature}`")
    st.divider()
    st.markdown("**Tips**")
    st.caption("🔤 Text: describe color, pattern, and type — *'blue floral midi dress'*")
    st.caption("🖼️ Image: upload a clean product photo on white background")
    st.caption("🔀 Combined: use image for category, text for color/style tweak")

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">👗 Fashion Search</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Semantic search across 44,000+ fashion products · FashionCLIP + FAISS</p>', unsafe_allow_html=True)

mode = st.radio("Mode", ["🔤 Text", "🖼️ Image", "🔀 Combined"],
                horizontal=True, label_visibility="collapsed")
st.divider()

# ── Search modes ──────────────────────────────────────────────────────────────
if mode == "🔤 Text":
    query = st.text_input("Describe what you're looking for",
                          placeholder="e.g. blue floral dress, black leather handbag, white sneakers")
    top_k = st.slider("Results", 4, 32, 8, 4)
    if st.button("Search", key="text_btn") and query.strip():
        with st.spinner("Searching…"):
            t0      = time.time()
            results = search_text(query, model, tokenizer, device,
                                  index, embeddings, p5, p95, temperature, top_k)
            st.session_state.result_items = fetch_result_items(results, dataset, metadata, top_k)
            st.session_state.result_time  = time.time() - t0

elif mode == "🖼️ Image":
    top_k    = st.slider("Results", 4, 32, 8, 4)
    uploaded = st.file_uploader("Upload a product photo", type=["jpg","jpeg","png","webp"])
    if uploaded is not None:
        img_bytes = uploaded.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.image(img_bytes, width=300)
        if st.button("Search", key="img_btn"):
            with st.spinner("Searching…"):
                t0      = time.time()
                results = search_image_query(pil_img, model, preprocess, device,
                                             index, embeddings, p5, p95, temperature, top_k)
                st.session_state.result_items = fetch_result_items(results, dataset, metadata, top_k)
                st.session_state.result_time  = time.time() - t0

elif mode == "🔀 Combined":
    col_a, col_b = st.columns([1, 2])
    pil_img = None
    with col_a:
        uploaded = st.file_uploader("Reference image", type=["jpg","jpeg","png","webp"])
        if uploaded is not None:
            img_bytes = uploaded.read()
            pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            st.image(img_bytes, width=300)
    with col_b:
        modifier = st.text_input("Text modifier", placeholder='"in red color", "with floral print"')
        alpha    = st.slider("Image ↔ Text weight", 0.0, 1.0, 0.3, 0.05,
                             help="0 = image only · 1 = text only")
        top_k    = st.slider("Results", 4, 32, 8, 4)
    if st.button("Search", key="comb_btn"):
        if pil_img is None:
            st.warning("Please upload a reference image.")
        elif not modifier.strip():
            st.warning("Please enter a text modifier.")
        else:
            with st.spinner("Searching…"):
                t0      = time.time()
                results = search_combined(pil_img, modifier, model, preprocess, tokenizer, device,
                                          index, embeddings, p5, p95, temperature, alpha, top_k)
                st.session_state.result_items = fetch_result_items(results, dataset, metadata, top_k)
                st.session_state.result_time  = time.time() - t0

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.result_items:
    n = len(st.session_state.result_items)
    t = st.session_state.result_time
    st.markdown(f'<p class="result-count">Top {n} results · {t:.2f}s</p>', unsafe_allow_html=True)
    render_grid(st.session_state.result_items)

st.divider()
st.markdown(
    "<small style='color:#aaa'>Marqo FashionCLIP · FAISS IVFFlat · 44k items · ceyda/fashion-products-small</small>",
    unsafe_allow_html=True,
)