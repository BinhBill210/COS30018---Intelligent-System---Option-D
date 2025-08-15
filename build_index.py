# build_index.py
"""
Build a FAISS index from your review files (raw_messages/*.json or data/reviews.jsonl).
Saves:
  - /mnt/data/index_demo/reviews.index
  - /mnt/data/index_demo/id_map.json
  - /mnt/data/index_demo/meta.json

Usage:
  conda activate review-index
  python build_index.py
"""
import os, json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

OUTPUT_DIR = Path("C:/Users/user/Documents/LLMtest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_reviews():
    reviews = []
    raw_dir = Path("raw_messages")
    if raw_dir.exists():
        for p in sorted(raw_dir.glob("*.json")):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                # attempt common fields for readable text
                raw = obj.get("raw", {})
                text = None
                if isinstance(raw, dict):
                    for key in ("text", "message", "summary", "content", "body", "title"):
                        if raw.get(key):
                            text = raw.get(key); break
                    text = text or raw.get("summary") or raw.get("title") or obj.get("id")
                else:
                    text = raw or obj.get("summary") or obj.get("id")
                rid = obj.get("id") or p.stem
                reviews.append({"id": str(rid), "text": str(text), "source": obj.get("source", "raw")})
            except Exception:
                continue
    # fallback to data/reviews.jsonl or reviews.jsonl
    if not reviews:
        for candidate in ["data/reviews.jsonl", "reviews.jsonl", "data/reviews.json"]:
            fp = Path(candidate)
            if fp.exists():
                with fp.open("r", encoding="utf-8") as f:
                    for i,line in enumerate(f):
                        if not line.strip(): continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            try:
                                obj = json.loads(f.read())
                            except Exception:
                                continue
                        text = obj.get("text") or obj.get("review") or obj.get("summary") or str(obj)
                        rid = obj.get("id") or obj.get("review_id") or f"r{i}"
                        reviews.append({"id":str(rid), "text":str(text), "source":"dataset"})
                break
    # if still empty -> create tiny synthetic set for testing
    if not reviews:
        synth = [
            "The delivery was late and the box was damaged when it arrived.",
            "Customer support was incredibly helpful and resolved my issue quickly.",
            "I found the price to be high compared to competitors.",
            "The product quality is outstanding — exceeded my expectations.",
            "App crashes constantly on startup. Very frustrating experience.",
            "I like the new interface, it's clean and intuitive.",
            "Shipping took two weeks longer than promised, unacceptable.",
            "Returned item process was smooth and refund arrived fast.",
            "Battery life is poor; it barely lasts a day with normal use.",
            "The features are great but the documentation is lacking."
        ]
        for i,t in enumerate(synth):
            reviews.append({"id":f"synthetic_{i}", "text":t, "source":"synthetic"})
    return reviews

def build_and_save_index(reviews, model_name="all-MiniLM-L6-v2"):
    texts = [r["text"] for r in reviews]
    print("Loading model:", model_name)
    model = SentenceTransformer(model_name)
    print("Encoding", len(texts), "texts (batched)...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)   # use inner-product on normalized vectors => cosine
    index.add(embeddings)
    print("Index built. n_vectors =", index.ntotal)

    idx_path = OUTPUT_DIR / "reviews.index"
    faiss.write_index(index, str(idx_path))
    print("Saved index to:", idx_path)

    id_map = [r["id"] for r in reviews]
    meta = {r["id"]: {"text": r["text"], "source": r.get("source")} for r in reviews}
    with open(OUTPUT_DIR / "id_map.json", "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Saved id_map.json and meta.json to:", OUTPUT_DIR)

if __name__ == "__main__":
    print("Loading reviews...")
    reviews = load_reviews()
    print(f"Loaded {len(reviews)} reviews. Example:", reviews[:2])
    build_and_save_index(reviews)
    print("Done. Files are in", OUTPUT_DIR.resolve())
