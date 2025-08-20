"""
load_and_query.py
Usage: python load_and_query.py "late delivery" 5
"""
import sys, json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

OUTDIR = Path("index_demo")
MODEL = "all-MiniLM-L6-v2"

def load_index(outdir=OUTDIR):
    index = faiss.read_index(str(outdir / "reviews.index"))
    id_map = json.load(open(outdir / "id_map.json", "r", encoding="utf-8"))
    meta = json.load(open(outdir / "meta.json", "r", encoding="utf-8"))
    return index, id_map, meta

def query_top_k(q, k=5):
    model = SentenceTransformer(MODEL)
    index, id_map, meta = load_index()
    q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, k)
    out = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1: continue
        rid = id_map[idx]
        out.append({"id": rid, "score": float(score), "text": meta[rid]["text"], "orig_id": meta[rid].get("orig_id")})
    return out

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv)>1 else "delivery late"
    k = int(sys.argv[2]) if len(sys.argv)>2 else 5
    results = query_top_k(q, k=k)
    for r in results:
        print(f"- [{r['score']:.3f}] {r['id']} (orig: {r['orig_id']}): {r['text'][:200]}")
