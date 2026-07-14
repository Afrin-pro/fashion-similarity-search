%pip install -q torch torchvision numpy Pillow faiss-cpu \
    ftfy regex tqdm datasets huggingface_hub open-clip-torch
print("✅ Done")

#cell 2
import gc, json, os, shutil, time
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
import open_clip
import faiss
from datasets import load_dataset

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
SAVE_DIR   = Path("artifacts")
SAVE_DIR.mkdir(exist_ok=True)

print(f"Device : {DEVICE}")
if DEVICE == "cpu":
    print("⚠️  No GPU! Runtime → Change runtime type → T4 GPU")

#cell 3
model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionCLIP')
tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionCLIP')
model = model.to(DEVICE)
model.eval()

with torch.no_grad():
    dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)
    dim = model.encode_image(dummy).shape[-1]
print(f"✅ Marqo FashionCLIP on {DEVICE}  |  dim: {dim}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — LOAD FULL DATASET, NO CAP
# Takes ALL ~44k items from ceyda/fashion-products-small
# selected_indices maps FAISS position → full dataset row index (critical fix)
# ─────────────────────────────────────────────────────────────────────────────
#cell 4
print("Loading dataset…")
ds_full = load_dataset("ceyda/fashion-products-small", split="train")
print(f"Full dataset size: {len(ds_full):,} rows")

# Filter out rows with missing images or captions, keep everything else
print("\nScanning dataset…")
selected_indices = []  # positions in ds_full → used to fetch images at query time
category_counts  = defaultdict(int)

for i in range(len(ds_full)):
    row = ds_full[i]
    # Skip rows with no useful category
    cat = (row.get("masterCategory") or "").strip()
    sub = (row.get("subCategory") or "").strip()
    if not cat and not sub:
        continue
    selected_indices.append(i)
    category_counts[(row.get("masterCategory") or "unknown").strip().lower()] += 1

print(f"\nMasterCategory distribution:")
for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat:<35} {count:>6,}")

print(f"\nTotal selected: {len(selected_indices):,} items")
ds = ds_full.select(selected_indices)
print(f"✅ Dataset ready: {len(ds):,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — ENCODE
# KEY FIX: metadata stores dataset_idx = selected_indices[local_pos]
# This is the original full-dataset row index used at query time to fetch images
# Without this, FAISS result idx 5 fetches ds_full[5] instead of the correct row
# ─────────────────────────────────────────────────────────────────────────────
#cell 5
def encode_streaming(ds, selected_indices, batch_size=BATCH_SIZE):
    all_emb  = []
    metadata = []
    skipped  = 0
    n        = len(ds)
    t0       = time.time()

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = ds[start:end]

        imgs, valid_local = [], []
        for j, img in enumerate(batch["image"]):
            if img is None:
                skipped += 1
                continue
            try:
                imgs.append(preprocess(img.convert("RGB")))
                valid_local.append(start + j)   # local position in ds
            except Exception:
                skipped += 1

        if not imgs:
            continue

        tensors = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            e = model.encode_image(tensors)
            e = e / e.norm(dim=-1, keepdim=True)
        all_emb.append(e.cpu().numpy().astype("float16"))

        for local_pos in valid_local:
            s = ds[local_pos]
            parts = [
                s.get("masterCategory", ""),
                s.get("subCategory", ""),
                s.get("articleType", ""),
                s.get("productDisplayName", ""),
                s.get("baseColour", ""),
                s.get("usage", ""),
            ]
            cap = " | ".join(p.strip() for p in parts if p and p.strip())
            metadata.append({
                "id":           len(metadata),
                "dataset_idx":  selected_indices[local_pos],  # ← CRITICAL: full dataset row index
                "caption":      cap,
                "category":     s.get("masterCategory", "unknown"),
                "subCategory":  s.get("subCategory", ""),
                "articleType":  s.get("articleType", ""),
                "productName":  s.get("productDisplayName", ""),
                "colour":       s.get("baseColour", ""),
                "gender":       s.get("gender", ""),
                "usage":        s.get("usage", ""),
            })

        del tensors, e, imgs, batch
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        done = end
        if done % 5000 == 0 or done == n:
            elapsed = time.time() - t0
            eta     = (elapsed / done) * (n - done)
            print(f"  {done:>6,}/{n:,}  |  "
                  f"{done*100/n:5.1f}%  |  "
                  f"elapsed {elapsed/60:.1f}m  |  "
                  f"ETA {eta/60:.1f}m  |  "
                  f"skipped {skipped}")

    embeddings = np.vstack(all_emb).astype("float32")
    print(f"\n✅ Encoded {len(metadata):,} images  |  skipped {skipped}")
    print(f"   Embeddings: {embeddings.shape}  ({embeddings.nbytes/1e6:.0f} MB)")
    return embeddings, metadata

print("Encoding dataset…")
embeddings, metadata = encode_streaming(ds, selected_indices)

#cell 6
DIM   = embeddings.shape[1]
n     = len(metadata)
NLIST = 256 if n >= 40_000 else 128

quantizer = faiss.IndexFlatIP(DIM)
index     = faiss.IndexIVFFlat(quantizer, DIM, NLIST, faiss.METRIC_INNER_PRODUCT)

print(f"Training IVFFlat index ({NLIST} clusters) on {n:,} vectors…")
index.train(embeddings)
index.add(embeddings)
index.nprobe = 32

print(f"✅ FAISS IVFFlat  |  vectors: {index.ntotal:,}  |  dim: {DIM}")

#cell 7
faiss.write_index(index, str(SAVE_DIR / "fashion.index"))
np.save(str(SAVE_DIR / "embeddings.npy"), embeddings)
json.dump(metadata, open(SAVE_DIR / "metadata.json", "w"))

print("✅ Saved:")
for f in ["fashion.index", "embeddings.npy", "metadata.json"]:
    mb = os.path.getsize(SAVE_DIR / f) / 1e6
    print(f"   {f:<25} {mb:>7.1f} MB")

# Verify dataset_idx is correct
sample = metadata[100]
print(f"\nVerification — metadata[100]:")
print(f"  dataset_idx : {sample['dataset_idx']}")
print(f"  caption     : {sample['caption'][:80]}")
fetched = ds_full[sample['dataset_idx']]
print(f"  ds_full row : {fetched.get('productDisplayName','')}")
print("  ✅ Match!" if sample['productName'] in fetched.get('productDisplayName','') else "  ❌ Mismatch — check selected_indices logic")

#cell 8 — sanity check
def quick_test(query):
    tokens = tokenizer([query]).to(DEVICE)
    with torch.no_grad():
        e = model.encode_text(tokens)
        e = e / e.norm(dim=-1, keepdim=True)
    emb = e.cpu().numpy().astype("float32")
    index.nprobe = 32
    scores, ids = index.search(emb, 5)
    print(f'\n"{query}"')
    for s, i in zip(scores[0], ids[0]):
        m = metadata[i]
        print(f"  [{s:.3f}]  {m['colour']:<12}  {m['articleType']:<20}  {m['productName'][:40]}")

quick_test("blue floral dress")
quick_test("black leather handbag")
quick_test("white sneakers women")
quick_test("red sports shoes")
quick_test("gold watch men")
print("\n✅ Check: colors and categories should match queries!")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — PUSH ARTIFACTS ONLY
# ─────────────────────────────────────────────────────────────────────────────
#cell 10
HF_USERNAME = "Ai-WhizKid"
SPACE_NAME  = "Fashion-Similarity_Search"
HF_TOKEN    = "hf........"   # huggingface.co/settings/tokens → Write

import subprocess
from pathlib import Path

SAVE_DIR = Path("artifacts")
REPO_DIR = Path("/content/hf_space")
REPO_URL = f"https://{HF_USERNAME}:{HF_TOKEN}@huggingface.co/spaces/{HF_USERNAME}/{SPACE_NAME}"

def run(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.stdout.strip():
        print(r.stdout.strip()[-600:])
    return r.returncode == 0

print("── Credentials ──")
run(f'git config --global user.email "{HF_USERNAME}@users.noreply.huggingface.co"')
run(f'git config --global user.name  "{HF_USERNAME}"')
run( 'git config --global credential.helper store')
Path("/root/.git-credentials").write_text(
    f"https://{HF_USERNAME}:{HF_TOKEN}@huggingface.co\n"
)

print("\n── Clone ──")
if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
if not run(f"GIT_TERMINAL_PROMPT=0 git clone {REPO_URL} {REPO_DIR}"):
    raise SystemExit("Clone failed — check token and space name")

print("\n── Git LFS ──")
run("git lfs install", cwd=REPO_DIR)
run('git lfs track "artifacts/fashion.index"',  cwd=REPO_DIR)
run('git lfs track "artifacts/embeddings.npy"', cwd=REPO_DIR)
run('git lfs track "artifacts/metadata.json"',  cwd=REPO_DIR)
run("git add .gitattributes", cwd=REPO_DIR)
print("  All 3 artifact files tracked by LFS ✅")

print("\n── Copy artifacts only (app/Dockerfile untouched) ──")
art_dst = REPO_DIR / "artifacts"
art_dst.mkdir(exist_ok=True)
for f in ["fashion.index", "embeddings.npy", "metadata.json"]:
    shutil.copy(SAVE_DIR / f, art_dst / f)
    mb = os.path.getsize(art_dst / f) / 1e6
    print(f"  {f:<25} {mb:>7.1f} MB")

print("\n── Commit ──")
run("git add artifacts/", cwd=REPO_DIR)
run(f'git commit -m "rebuild: FashionCLIP + full ceyda dataset + dataset_idx fix ({len(metadata):,} items)"', cwd=REPO_DIR)

print("\n── Push (LFS upload ~4-5 min) ──")
ok = run("git push", cwd=REPO_DIR)

if ok:
    print()
    print("=" * 60)
    print("✅  Live at:")
    print(f"   https://huggingface.co/spaces/{HF_USERNAME}/{SPACE_NAME}")
    print("=" * 60)
else:
    print("\n❌ Push failed — debug:")
    print(f"r = subprocess.run('git push', shell=True, cwd='{REPO_DIR}', capture_output=True, text=True)")
    print("print(r.stdout + r.stderr)")
