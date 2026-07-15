<div align="center">

# 👗 Fashion Search Engine

**Multimodal semantic search over 44,000 fashion products**

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow?style=for-the-badge)](https://huggingface.co/spaces/Ai-WhizKid/Fashion-Similarity_Search)
[![HF Space](https://img.shields.io/badge/Space-Ai--WhizKid%2FFashion--Similarity__Search-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/Ai-WhizKid/Fashion-Similarity_Search)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.44+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Deployed-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<br/>

Search by text description, upload an image, or combine both —
powered by **Marqo FashionCLIP** and **Facebook FAISS**.

**🔗 [Try the live demo →](https://huggingface.co/spaces/Ai-WhizKid/Fashion-Similarity_Search)**

</div>

---

## ✨ Features

- 🔤 **Text Search** — natural language queries like *"blue floral midi dress"* or *"black leather ankle boots"*
- 🖼️ **Image Search** — upload any product photo and find visually similar items
- 🔀 **Combined Search** — use an image for category + text to shift color or style (*"in red color"*)
- 🌡️ **Temperature control** — tune result sharpness from precise (0.1) to exploratory (0.5)
- ⚡ **~0.3s query time** — FAISS IVFFlat index with 128 clusters, nprobe=32
- 🎯 **0–100% match scores** — unified scale across text and image queries via percentile calibration

---

## 🏗️ Architecture

```
Query (text / image / both)
         │
         ▼
 Marqo FashionCLIP          512-d L2-normalised vector
 (open-clip-torch)    ──►   fine-tuned on fashion data
         │
         ▼
 FAISS IVFFlat index        top-K nearest neighbours
 128 clusters · nprobe=32   cosine similarity
         │
         ▼
 Temperature-scaled         unified 0–100% score
 score calibration    ──►   text + image comparable
         │
         ▼
 Streamlit UI               Docker · HuggingFace Spaces
                            free CPU tier
```

### Why FashionCLIP over generic CLIP?
Generic CLIP ignores color and garment category in fashion queries. [`Marqo/marqo-fashionCLIP`](https://huggingface.co/Marqo/marqo-fashionCLIP) is fine-tuned on fashion data — *"blue floral dress"* returns blue floral dresses, not random blue items.

### Score calibration — modality gap fix
Raw CLIP dot products differ by modality (image-image ~0.8, text-image ~0.3). Fixed via percentile calibration with a modality gap ratio of 0.55, mapping both to a unified 0–100% scale.

### Combined search — category-anchored re-ranking
```
blended_vec  = (1 - α) × image_vec + α × text_vec
final_score  = (1 - α) × image_score + α × text_score
```
Default α = 0.3 keeps the image dominant — uploading a handbag + *"in red"* returns red handbags, not red dresses.

### Temperature scaling
```
scaled_score = raw_dot_product / temperature
final_%      = clip((scaled_score − p5) / (p95 − p5) × 100, 0, 100)
```
Low temperature (0.1) → sharp peaks, precise results. High temperature (0.5) → softer curve, better for exploration.

---

## 📦 Stack

| Component | Technology |
|---|---|
| Vision-Language Model | [`Marqo/marqo-fashionCLIP`](https://huggingface.co/Marqo/marqo-fashionCLIP) via `open-clip-torch` |
| Vector Index | `faiss-cpu` — IVFFlat, 128 clusters, nprobe=32 |
| Dataset | [`ceyda/fashion-products-small`](https://huggingface.co/datasets/ceyda/fashion-products-small) — 44k images @ 512px |
| Frontend | Streamlit ≥ 1.44 |
| Deployment | Docker on [HuggingFace Spaces](https://huggingface.co/spaces/Ai-WhizKid/Fashion-Similarity_Search) (free CPU tier) |
| Artifact storage | Git LFS (~250 MB — instant ~30s startup) |
| Index build | Google Colab T4 GPU (~20 min) |

---

## 🗂️ Repo Structure

```
├── src/
│   └── streamlit_app.py        # UI + search logic + score calibration
├── build_index_colab.py        # full ML pipeline — encode + index + push
├── Dockerfile
└── requirements.txt
```

> Artifact files (`fashion.index`, `embeddings.npy`, `metadata.json`) are hosted on HuggingFace Spaces via Git LFS and not included in this repo.

---

## 🚀 Run Locally

```bash
# 1. Clone HF Space (includes artifacts via LFS)
git clone https://huggingface.co/spaces/Ai-WhizKid/Fashion-Similarity_Search
cd Fashion-Similarity_Search
git lfs pull

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run src/streamlit_app.py
```

---

## 🔁 Rebuild the Index

Run `build_index_colab.py` on a **T4 GPU** in Google Colab (~20 min for 44k images):

```python
# What the script does:
ds = load_dataset("ceyda/fashion-products-small", split="train")  # 44k rows
# → encodes in batches of 64 (peak RAM ~200 MB vs ~15 GB naive)
# → stores dataset_idx in metadata (correct image fetch at query time)
# → trains IVFFlat(nlist=128), adds vectors, pushes artifacts via Git LFS
```

The script pushes **only the 3 artifact files** — `streamlit_app.py`, `Dockerfile`, and `requirements.txt` are never touched.

---

## 🌡️ Temperature

| Setting | Effect | Best for |
|---|---|---|
| **0.1** | Sharp peaks — only very close matches score high | Precise queries: *"blue floral midi dress"* |
| **0.25** *(default)* | Balanced | Most searches |
| **0.5** | Soft curve — more results score similarly | Exploration, vague queries |

---

## 📊 Score Guide

| Badge | Range | Meaning |
|---|---|---|
| 🟢 | 70–100% | Strong match |
| 🟡 | 40–69% | Decent match |
| 🔴 | 0–39% | Weak match |

---

## 🐛 Notable Bugs Fixed

| Bug | Root Cause | Fix |
|---|---|---|
| All results were dresses | `ds.select(range(50_000))` grabbed first 50k rows — 100% dresses | Balanced category sampling |
| Wrong images displayed | FAISS position ≠ full dataset row index after subset selection | Stored `dataset_idx` in metadata, always fetch using it |
| Blurry images | Source dataset stored images at only 60px | Switched to `ceyda` dataset (up to 512px) |
| Random garbage results | Index built with generic CLIP, app querying with FashionCLIP — mismatched vector spaces | Rebuilt index with FashionCLIP |
| Image grid shaking | `use_container_width=True` triggers layout recalc on every Streamlit rerender | Changed to fixed `width=300` |
| 403 on file upload | Streamlit XSRF protection incompatible with HF Spaces free tier proxy | `--server.enableXsrfProtection=false` in Dockerfile ENTRYPOINT |
| `artifacts/` not found in Space | `COPY artifacts/` missing from Dockerfile | Added explicit `COPY artifacts/ ./artifacts/` |
| RAM crash during index build | Loading 44k PIL images at once = ~15 GB RAM | Stream encode in batches of 64, discard after each batch |

---

## 🔮 Planned Enhancements

- [ ] Product name display on result cards
- [ ] Filter sidebar — gender, category, colour
- [ ] Similar items button (uses stored embeddings, no re-encoding needed)
- [ ] Pagination — load more without re-running search
- [ ] Larger dataset — DeepFashion2 (800k images)
- [ ] Cross-encoder re-ranking on top-20 FAISS results
- [ ] Multilingual support via query-time translation
- [ ] Outfit completion — suggest complementary items across categories

---

## 📝 License

MIT © 2025 [Ai-WhizKid](https://huggingface.co/Ai-WhizKid)
