---
title: Fashion-Similarity Search
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
license: mit
---

---
title: Fashion Similarity Search
emoji: 👗
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
license: mit
---

# 👗 Fashion Search Engine — FashionCLIP + FAISS

A multimodal semantic search engine for fashion e-commerce. Search 44,000+ products by text description, image upload, or both combined. Built end-to-end from data pipeline to deployed UI.

[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/Ai-WhizKid/Fashion-Similarity_Search)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.44+-red)
![FAISS](https://img.shields.io/badge/FAISS-IVFFlat-green)
![FashionCLIP](https://img.shields.io/badge/Model-FashionCLIP-purple)

---

## 🔍 Search Modes

| Mode | How it works |
|---|---|
| **🔤 Text** | Natural language query — *"blue floral midi dress"*, *"black leather handbag"* |
| **🖼️ Image** | Upload a product photo, find visually similar items |
| **🔀 Combined** | Reference image + text modifier — upload a bag, type *"in red color"* |

---

## 🏗️ Architecture

```
Query (text / image / both)
         │
         ▼
Marqo FashionCLIP encoder  →  512-d L2-normalised vector
         │
         ▼
FAISS IVFFlat index        →  top-K nearest neighbours
(128 clusters, nprobe=32)       (cosine similarity)
         │
         ▼
Temperature-scaled score calibration  →  0–100% unified scale
         │
         ▼
Streamlit UI (HuggingFace Spaces · Docker · free CPU tier)
```

---

## 🌡️ Temperature — How It Works

Temperature is a scaling factor applied to raw cosine similarity scores before the 0–100% percentile mapping. It controls how *sharp or soft* the score distribution is.

**The math:**
```
scaled_score = raw_dot_product / temperature
final_%      = clip((scaled_score - p5) / (p95 - p5) × 100, 0, 100)
```

| Temperature | Effect | Best for |
|---|---|---|
| **0.1** | Sharp peaks — only very close matches score high. Clear separation between good and bad results. | Precise queries: *"blue floral dress"* |
| **0.25** *(default)* | Balanced — good discrimination without being too strict | Most searches |
| **0.5** | Soft curve — more results score similarly, broader spread | Exploration, vague queries |

The temperature slider lives in the sidebar and applies in real time without any reindexing.

---

## 📦 Dataset

**[ceyda/fashion-products-small](https://huggingface.co/datasets/ceyda/fashion-products-small)**
- 44,441 product images at up to 512px — clean white background, e-commerce style
- Source: Myntra fashion catalog
- Categories: Topwear, Bottomwear, Footwear, Bags, Watches, Jewellery, Eyewear, Wallets, Belts, and more (45 subcategories)
- Fields used: `masterCategory`, `subCategory`, `articleType`, `productDisplayName`, `baseColour`, `gender`, `usage`

---

## ⚙️ Tech Stack

| Component | Library / Tool |
|---|---|
| Vision-Language Model | [`Marqo/marqo-fashionCLIP`](https://huggingface.co/Marqo/marqo-fashionCLIP) via `open-clip-torch` |
| Vector Index | `faiss-cpu` — IVFFlat, 128 clusters, nprobe=32 |
| Dataset | `ceyda/fashion-products-small` via HuggingFace Datasets |
| UI | Streamlit ≥ 1.44 |
| Deployment | Docker on HuggingFace Spaces (free CPU tier) |
| Artifact storage | Git LFS (~250 MB total) |
| Development environment | Google Colab (T4 GPU for index build) |

---

## 🗂️ Repo Structure

```
repo/
├── src/
│   └── streamlit_app.py       ← Streamlit UI + all search logic
├── artifacts/                 ← Git LFS tracked (~250 MB total)
│   ├── fashion.index          ← FAISS IVFFlat index
│   ├── embeddings.npy         ← Raw FashionCLIP embeddings (float32)
│   └── metadata.json          ← Captions, colours, categories, dataset_idx
├── Dockerfile
└── requirements.txt
```

---

## 🔧 Key Design Decisions

### Model — FashionCLIP over generic CLIP
Generic CLIP ViT-B/32 was tried first. It returned wrong categories and ignored color — *"blue floral dress"* returned shoes and bags. Switched to [`Marqo/marqo-fashionCLIP`](https://huggingface.co/Marqo/marqo-fashionCLIP), fine-tuned specifically on fashion data. Color, pattern, and category matching improved dramatically.

### Index — IVFFlat over FlatIP
FlatIP is brute-force (checks every vector). IVFFlat clusters into 128 groups and searches only the nearest 32 — ~4× faster at 44k items with less than 1% recall drop.

### Score calibration — modality gap fix
Raw CLIP dot products differ across modalities — image-image ~0.8, text-image ~0.3. Fixed via percentile calibration:
- Sample 400 random image vectors as proxy queries
- Compute p5 and p95 of their score distribution
- Scale text scores by empirical modality gap ratio (0.55 for FashionCLIP)
- Map both to 0–100% via temperature scaling

### Combined search — image-anchored re-ranking
Without anchoring, searching *"in red color"* with a handbag image returns red dresses instead. Fix — two-step search:
1. Blended embedding (`α × text + (1-α) × image`) fetches a wide candidate pool
2. Re-rank by `(1-α) × image_score + α × text_score`
3. Default α = 0.3 — image dominates, keeps the category anchored

### Memory-safe batch encoding
Encoding 44k images naively loads everything into RAM (~15 GB). Instead, images are streamed in batches of 64, encoded, then immediately discarded. Peak RAM stays at ~200 MB — fits comfortably in Colab's free tier.

### dataset_idx mapping — critical correctness fix
FAISS stores vectors for a filtered/reordered subset of the dataset. Without tracking the original row position, FAISS result position 5 fetches `dataset[5]` (a random unrelated item) instead of the correct image. Every metadata entry stores `dataset_idx` — the original full-dataset row index — so image fetches are always correct.

### Git LFS for instant startup
All three artifact files tracked with Git LFS. Startup time ~30s (model load) vs ~20 min (index rebuild from scratch) on every Space restart.

### Session state image caching
Search results stored as JPEG bytes in `st.session_state`. Streamlit rerenders read from memory and never re-fetch from the dataset — prevents image shaking on every widget interaction.

### Fixed image width
Grid renders at `width=300` (not `use_container_width=True`). Fixed width prevents Streamlit from recalculating layout on every widget interaction, which was causing the entire result grid to shake.

---

## 🐛 Bugs Encountered & Fixed

| Bug | Cause | Fix |
|---|---|---|
| Images showing wrong items | FAISS position ≠ full dataset row index — balanced subset reorders rows | Added `dataset_idx` to metadata, fetch using it |
| Blurry images | `ashraq/fashion-product-images-small` stores images at 60px | Switched to `ceyda/fashion-products-small` (up to 512px) |
| Wrong categories in results | Generic CLIP ignores fashion-specific attributes | Switched to `Marqo/marqo-fashionCLIP` |
| Text search returning shoes for "blue floral dress" | Index built with generic CLIP, app querying with FashionCLIP — mismatched vector spaces | Rebuilt index with FashionCLIP |
| Image grid shaking on every interaction | `use_container_width=True` triggers layout recalc on every Streamlit rerender | Changed to fixed `width=300` |
| 403 error on file upload | Streamlit XSRF protection blocks uploads on HF Spaces free tier | Added `--server.enableXsrfProtection=false` to Dockerfile ENTRYPOINT |
| `artifacts/` not found in Space | `COPY artifacts/` missing from Dockerfile | Added `COPY artifacts/ ./artifacts/` |
| Session crashed during index build | Loading 44k PIL images at once = ~15 GB RAM | Stream encode in batches of 64 |
| App overwriting updated files on push | Build script was copying local `streamlit_app.py` into repo before pushing | Push artifacts only — app files untouched |
| `use_column_width` deprecation | Old Streamlit parameter removed | Changed to `use_container_width=True`, then `width=300` |
| Images loading slowly | `streaming=True` with `skip().take()` iterates the whole stream per image | Non-streaming `load_dataset()` with direct `dataset[int(idx)]` access |

---

## 📊 Score Guide

| Badge | Range | Meaning |
|---|---|---|
| 🟢 | 70–100% | Strong match |
| 🟡 | 40–69% | Decent match |
| 🔴 | 0–39% | Weak match |

Scores are normalised to 0–100% using percentile calibration with temperature scaling. Text and image scores are directly comparable.

---

## 🚀 Run Locally

```bash
git clone https://huggingface.co/spaces/Ai-WhizKid/Fashion-Similarity_Search
cd Fashion-Similarity_Search
git lfs pull          # pulls the 3 artifact files
pip install -r requirements.txt
streamlit run src/streamlit_app.py
```

---

## 🔁 Rebuild the Index

Run `build_index_colab.py` on a **T4 GPU** in Google Colab (~20 min):

1. Loads full `ceyda/fashion-products-small` dataset
2. Encodes all images with FashionCLIP in batches of 64
3. Trains FAISS IVFFlat index
4. Saves 3 artifact files with correct `dataset_idx` mapping
5. Pushes artifacts via Git LFS — app files untouched

---

## 🧾 Resume Talking Points

- Built a **multimodal fashion search engine** indexing 44,000 products using Marqo FashionCLIP and Facebook FAISS — supports text, image, and combined search modes
- Solved the **CLIP modality gap** using percentile-based score calibration with temperature scaling — unified text and image scores to a 0–100% scale without dataset-specific tuning
- Designed **category-anchored re-ranking** for combined search — prevents embedding drift across product categories when mixing image and text queries
- Engineered **memory-safe batch encoding** processing 44k images in ~200 MB RAM using streaming iteration (vs ~15 GB naive approach)
- Diagnosed and fixed a **dataset index mapping bug** where balanced subset reordering caused FAISS result positions to fetch completely wrong images from the full dataset
- Deployed on **HuggingFace Spaces free tier** using Docker + Git LFS for instant ~30s startup — no paid infrastructure
- Debugged and resolved 12+ production issues including Docker COPY omissions, Streamlit XSRF 403 errors, image grid shaking from rerender loops, dataset streaming bottlenecks, mismatched model/index vector spaces, and Git LFS pointer vs real file issues
- Full stack: data pipeline (HuggingFace Datasets) → vector indexing (FAISS) → encoder (FashionCLIP via open-clip-torch) → UI (Streamlit) → deployment (Docker + HF Spaces)

---

## 📝 License

MIT
