# load_and_query_example.py
"""
Example of loading the saved FAISS index and performing a top-K query.
Usage:
  conda activate review-index
  python load_and_query_example.py "late delivery" 5
"""
import sys, json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

OUTPUT_DIR = Path("C:/Users/user/Documents/LLMtest")
MODEL_NAME = "all-MiniLM-L6-v2"

def load_index():
    idx_path = OUTPUT_DIR / "reviews.index"
    id_map_path = OUTPUT_DIR / "id_map.json"
    meta_path = OUTPUT_DIR / "meta.json"
    index = faiss.read_index(str(idx_path))
    id_map = json.load(open(id_map_path, "r", encoding="utf-8"))
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    return index, id_map, meta

def query_top_k(query, k=5):
    model = SentenceTransformer(MODEL_NAME)
    index, id_map, meta = load_index()
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    scores, idxs = index.search(q_emb, k)
    out = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1: continue
        rid = id_map[idx]
        out.append({"id": rid, "score": float(score), "text": meta[rid]["text"]})
    return out

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "late delivery"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    res = query_top_k(q, k=k)
    print("Query:", q)
    for r in res:
        print(f"- [{r['score']:.3f}] {r['id']}: {r['text']}")
