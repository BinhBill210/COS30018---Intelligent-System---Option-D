#!/usr/bin/env python3
"""
migrate_business_to_chromadb.py

Script to embed business data and store in ChromaDB vector store.
"""
import pandas as pd
from chromadb_integration import ChromaDBVectorStore
from pathlib import Path

BUSINESS_DATA_PATH = "data/processed/business_cleaned.parquet"
CHROMA_PATH = "./business_chroma_db"
COLLECTION_NAME = "yelp_businesses"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    print(f"Loading business data from {BUSINESS_DATA_PATH}...")
    if BUSINESS_DATA_PATH.endswith('.parquet'):
        df = pd.read_parquet(BUSINESS_DATA_PATH)
    else:
        df = pd.read_csv(BUSINESS_DATA_PATH)
    print(f"Loaded {len(df)} businesses.")

    # Prepare documents for embedding
    documents = []
    for idx, row in df.iterrows():
        # Concatenate relevant fields for semantic search
        text = f"{row['name']}, {row['address']}, {row['city']}, {row['state']}, {row['categories']}"
        # Always include these fields in metadata, even if empty
        metadata = {
            "business_id": row.get("business_id", ""),
            "name": row.get("name", ""),
            "address": row.get("address", ""),
            "city": row.get("city", ""),
            "state": row.get("state", ""),
            "postal_code": row.get("postal_code", ""),
            "stars": row.get("stars", ""),
            "categories": row.get("categories", ""),
        }
        # Add all other fields, converting dict/list to str
        for k in df.columns:
            if k not in metadata and pd.notna(row[k]):
                v = row[k]
                if isinstance(v, (dict, list)):
                    metadata[k] = str(v)
                else:
                    metadata[k] = v
        documents.append({
            "page_content": text,
            "metadata": metadata
        })

    print(f"Prepared {len(documents)} documents for embedding.")

    # Initialize ChromaDB vector store
    vector_store = ChromaDBVectorStore(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_model=EMBEDDING_MODEL
    )

    # Add documents to ChromaDB
    from langchain.docstore.document import Document
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in documents]
    print("Adding documents to ChromaDB...")
    batch_size = 5000  # or 5000, or any value <= 5461
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        vector_store.vectorstore.add_documents(batch)
        print(f"Added batch {i//batch_size + 1}/{(len(docs)//batch_size) + 1}")
    print("Business embedding migration complete!")

if __name__ == "__main__":
    main()
