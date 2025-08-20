#!/usr/bin/env python3
"""
build_index_from_csv.py (GPU-aware + parquet support + robust JSON metadata)

Usage examples:
  # CPU (default)
  python build_index_from_csv.py --input data/review_cleaned.csv --outdir index_demo

  # Use CUDA GPU if available
  python build_index_from_csv.py --input cleaned_data/review_cleaned.parquet --outdir index_demo --device cuda --batch-size 256

  # Input can be a directory containing many parquet/csv files
  python build_index_from_csv.py --input data/ --outdir index_demo --device cuda

Outputs (in outdir):
  - reviews.index
  - id_map.json
  - meta.json
  - index_metadata.json
"""
import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import torch

# ---------- JSON serialization helper ----------
def make_json_serializable(v):
    """Convert pandas/numpy/datetime types into JSON-serializable Python primitives."""
    # None / NaN
    try:
        if v is None:
            return None
    except Exception:
        pass
    # pandas NA / numpy nan
    if pd.isna(v):
        return None

    # pandas Timestamp or datetime-like
    try:
        if isinstance(v, pd.Timestamp):
            return v.isoformat()
    except Exception:
        pass

    # numpy scalar types
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)

    # python datetime/date
    try:
        import datetime
        if isinstance(v, (datetime.datetime, datetime.date)):
            return v.isoformat()
    except Exception:
        pass

    # dict / list recursion
    if isinstance(v, dict):
        return {str(k): make_json_serializable(val) for k, val in v.items()}
    if isinstance(v, list):
        return [make_json_serializable(x) for x in v]

    # basic python primitives (str, int, float, bool)
    if isinstance(v, (str, int, float, bool)):
        return v

    # fallback: try string conversion
    try:
        return str(v)
    except Exception:
        return None

# ---------- File handling / reading ----------
def find_input_files(path: str):
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = list(p.glob("*.parquet")) + list(p.glob("*.parq")) + list(p.glob("*.par")) + list(p.glob("*.csv"))
        return sorted(files)
    raise FileNotFoundError(f"No file or directory found at {path}")

def read_file_to_df(path: Path):
    ext = path.suffix.lower()
    if ext in [".parquet", ".parq", ".par"]:
        return pd.read_parquet(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError("Unsupported file type: " + str(path))

def prepare_records_from_dataframe(df: pd.DataFrame, id_col="review_id", text_col="text", max_rows: int = None):
    records = []
    if max_rows:
        df = df.head(max_rows)
    for idx, row in df.iterrows():
        rid = None
        if id_col in df.columns:
            rid = row.get(id_col)
        if pd.isna(rid) or rid is None:
            rid = f"r_{idx}"
        rid = str(rid)
        text = row.get(text_col, "")
        if pd.isna(text):
            text = ""
        # collect some metadata (serialize later)
        raw_meta = {}
        for k in ("date", "stars", "business_id", "user_id"):
            if k in df.columns:
                raw_meta[k] = row.get(k)
        # convert to JSON-serializable
        serial_meta = {k: make_json_serializable(v) for k, v in raw_meta.items()}
        records.append({"id": rid, "text": str(text), "meta": serial_meta})
    return records

# ---------- chunking ----------
def chunk_text(text: str, chunk_size:int=0, stride:int=50) -> List[str]:
    """Split text into overlapping word-based chunks. chunk_size=0 => no chunking."""
    if not text:
        return [""]
    if chunk_size <= 0:
        return [text]
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - stride
    return chunks

# ---------- encoding & index building ----------
def encode_and_build_index(records, outdir="index_demo", model_name="all-MiniLM-L6-v2",
                           chunk_size:int=0, stride:int=50, batch_size:int=128, device:str="cpu"):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    # prepare chunks and metadata
    texts = []
    id_map = []
    meta = {}
    for r in records:
        chunks = chunk_text(r["text"], chunk_size=chunk_size, stride=stride)
        for i, ch in enumerate(chunks):
            chunk_id = f"{r['id']}__c{i}" if chunk_size>0 else r["id"]
            id_map.append(chunk_id)
            texts.append(ch)
            # ensure metadata values are JSON-friendly
            row_meta = r.get("meta") or {}
            serial_row_meta = {k: make_json_serializable(v) for k, v in row_meta.items()}
            meta[chunk_id] = {"orig_id": r["id"], "text": ch, **serial_row_meta}

    n = len(texts)
    if n == 0:
        raise ValueError("No texts found to index.")

    print(f"[index] Encoding {n} items with model={model_name} on device={device} (batch_size={batch_size})")

    # Load model (sentence-transformers handles device placement)
    model = SentenceTransformer(model_name, device=device)

    # We'll encode in batches on the chosen device. Use convert_to_tensor to keep tensors on device.
    all_embeddings = None
    n_batches = math.ceil(n / batch_size)
    for i in range(n_batches):
        start = i * batch_size
        end = min((i+1) * batch_size, n)
        batch_texts = texts[start:end]
        # convert_to_tensor keeps tensors on device if device != cpu
        emb_tensor = model.encode(batch_texts, convert_to_numpy=False, convert_to_tensor=True, show_progress_bar=False)
        # If on CUDA, optionally cast to fp16 for memory/speed. Keep fallback to fp32 if cast fails.
        if str(device).startswith("cuda") and emb_tensor.dtype == torch.float32:
            try:
                emb_tensor = emb_tensor.half()
            except Exception:
                pass
        # normalize on device
        try:
            emb_tensor = torch.nn.functional.normalize(emb_tensor, p=2, dim=1)
        except Exception:
            # fallback if emb_tensor is numpy (shouldn't be here)
            pass
        # move to CPU numpy float32 for FAISS (FAISS expects float32)
        emb_cpu = emb_tensor.cpu().detach().float().numpy()
        if all_embeddings is None:
            all_embeddings = emb_cpu
        else:
            all_embeddings = np.vstack([all_embeddings, emb_cpu])
        print(f"  encoded batch {i+1}/{n_batches}: shapes {emb_cpu.shape}")

    all_embeddings = np.asarray(all_embeddings, dtype="float32")
    dim = all_embeddings.shape[1]
    print(f"[index] Building FAISS IndexFlatIP with dim={dim} and adding {all_embeddings.shape[0]} vectors ...")
    index = faiss.IndexFlatIP(dim)
    index.add(all_embeddings)
    print("[index] index.ntotal =", index.ntotal)

    # Persist index and maps
    idx_path = outdir / "reviews.index"
    faiss.write_index(index, str(idx_path))
    with open(outdir / "id_map.json", "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)
    with open(outdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    index_meta = {
        "embedding_model": model_name,
        "index_type": "IndexFlatIP",
        "n_vectors": int(index.ntotal),
        "chunk_size": int(chunk_size),
        "stride": int(stride),
        "device_used": device,
        "batch_size": int(batch_size)
    }
    with open(outdir / "index_metadata.json", "w", encoding="utf-8") as f:
        json.dump(index_meta, f, ensure_ascii=False, indent=2)

    print(f"[index] Saved index and metadata to {outdir.resolve()}")

# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to csv/parquet file or directory")
    p.add_argument("--outdir", default="index_demo")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--chunk-size", type=int, default=0, help="words per chunk; 0 = no chunking")
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default=None, help="device: 'cuda', 'mps', 'cpu', or 'auto' (default 'auto')")
    p.add_argument("--max-rows", type=int, default=0)
    args = p.parse_args()

    # device selection
    device = args.device or "auto"
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            # try MPS (Apple) if available
            if hasattr(torch, "has_mps") and torch.has_mps:
                device = "mps"
            else:
                device = "cpu"
    print("[main] chosen device:", device)

    files = find_input_files(args.input)
    print(f"[main] found {len(files)} input files:")
    for f in files:
        print("   ", f)

    records = []
    for f in files:
        print("[main] reading", f)
        df = read_file_to_df(f)
        recs = prepare_records_from_dataframe(df, id_col="review_id", text_col="text", max_rows=(args.max_rows or None))
        records.extend(recs)
    print(f"[main] total records loaded: {len(records)}")

    encode_and_build_index(records, outdir=args.outdir, model_name=args.model,
                           chunk_size=args.chunk_size, stride=args.stride,
                           batch_size=args.batch_size, device=device)

if __name__ == "__main__":
    main()
