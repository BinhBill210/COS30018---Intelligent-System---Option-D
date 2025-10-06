#!/usr/bin/env python3
"""
ChromaDB Connection Diagnostic Tool
"""

import chromadb
import os
from pathlib import Path

def test_chromadb_connections():
    print("ChromaDB Connection Diagnostic")
    print("=" * 50)
    
    # Test 1: Local persistent client
    print("\n1. Testing Local Persistent ChromaDB...")
    try:
        client = chromadb.PersistentClient(path='./chroma_db')
        collections = client.list_collections()
        print(f"Local ChromaDB works - Collections: {[c.name for c in collections]}")
        
        for collection_name in ['yelp_reviews', 'yelp_businesses']:
            try:
                collection = client.get_collection(collection_name)
                count = collection.count()
                print(f"   {collection_name}: {count:,} items")
            except Exception as e2:
                print(f"   {collection_name}: Not found - {e2}")
    except Exception as e:
        print(f"❌ Local ChromaDB failed: {e}")
    
    # Test 2: Check for running Chrome servers
    print("\n2. Testing HTTP ChromaDB Servers...")
    
    # Common ports and hosts
    test_configs = [
        ('localhost', 8000),
        ('localhost', 8001),
        ('127.0.0.1', 8000),
        ('127.0.0.1', 8001),
        ('172.24.104.210', 8000),
        ('172.24.104.210', 8001),
    ]
    
    working_servers = []
    
    for host, port in test_configs:
        try:
            print(f"   Testing {host}:{port}...")
            client = chromadb.HttpClient(host=host, port=port)
            client.heartbeat()
            collections = client.list_collections()
            print(f"   {host}:{port} works - Collections: {[c.name for c in collections]}")
            working_servers.append((host, port))
            
            # Test collections
            for collection_name in ['yelp_reviews', 'yelp_businesses']:
                try:
                    collection = client.get_collection(collection_name)
                    count = collection.count()
                    print(f"      {collection_name}: {count:,} items")
                except Exception as e2:
                    print(f"      {collection_name}: Not found")
                    
        except Exception as e:
            print(f"   {host}:{port} failed: Connection refused")
    
    # Test 3: Check environment variables
    print("\n3. Checking Environment Configuration...")
    chroma_host = os.environ.get("CHROMA_HOST", "localhost")
    print(f"   CHROMA_HOST environment variable: {chroma_host}")
    
    # Test 4: Check if ChromaDB directories exist
    print("\n4. Checking ChromaDB Data Directories...")
    directories = ['./chroma_db', './business_chroma_db']
    for dir_path in directories:
        if Path(dir_path).exists():
            print(f"   {dir_path} exists")
            # List contents
            contents = list(Path(dir_path).iterdir())
            print(f"      Contents: {[f.name for f in contents[:5]]}")
        else:
            print(f"   {dir_path} not found")
    
    # Summary
    print("\n" + "="*50)
    print("DIAGNOSTIC SUMMARY:")
    if working_servers:
        print(f"Working ChromaDB servers: {working_servers}")
    else:
        print("No working ChromaDB HTTP servers found")
        print("   -> Solution: Use local persistent client or start ChromaDB server")
    
    return working_servers

if __name__ == "__main__":
    working_servers = test_chromadb_connections()
