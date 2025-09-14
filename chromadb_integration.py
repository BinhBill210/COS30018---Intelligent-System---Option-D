# chromadb_integration.py
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

class ChromaDBVectorStore:
    def __init__(self, 
                 collection_name: str = "yelp_reviews",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        # Initialize embeddings (same model you're using)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize LangChain Chroma wrapper
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            client=self.client,
            persist_directory=persist_directory
        )
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
    def load_and_index_reviews(self, data_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """Load reviews from CSV/Parquet and create vector index"""
        
        print(f"Loading data from {data_path}...")
        
        # Read data (supporting both CSV and Parquet like your current system)
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        
        print(f"Loaded {len(df)} reviews")
        
        # Convert to LangChain Documents with metadata
        documents = []
        for idx, row in df.iterrows():
            # Create text content
            text = str(row.get('text', ''))
            if not text or pd.isna(text) or text.strip() == '':
                continue
                
            # Prepare metadata (all columns except text)
            metadata = {}
            for k, v in row.to_dict().items():
                if k != 'text' and pd.notna(v):
                    # Convert to JSON-serializable types
                    if isinstance(v, (pd.Timestamp,)):
                        metadata[k] = v.isoformat()
                    elif isinstance(v, (int, float, str, bool)):
                        metadata[k] = v
                    else:
                        metadata[k] = str(v)
            
            # Create Document
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            documents.append(doc)
        
        print(f"Created {len(documents)} valid documents")
        
        # Use LangChain's text splitter for chunking (replaces your custom chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]  # Better than word-based chunking
        )
        
        # Split documents
        splits = text_splitter.split_documents(documents)
        print(f"Split into {len(splits)} chunks")
        
        # Add to vector store in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i+batch_size]
            self.vectorstore.add_documents(batch)
            print(f"Indexed batch {i//batch_size + 1}/{(len(splits)//batch_size) + 1}")
        
        print(f"Successfully indexed {len(splits)} document chunks from {len(documents)} reviews")
        
        return len(splits)
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None):
        """Search for similar documents"""
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )
    
    def as_retriever(self, **kwargs):
        """Return as LangChain retriever for agent integration"""
        return self.vectorstore.as_retriever(**kwargs)
    
    def get_collection_info(self):
        """Get information about the collection"""
        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            return {"error": str(e)}
