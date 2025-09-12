"""
Script to start ChromaDB servers for both business and review databases.
"""
import subprocess
import os
import sys
from pathlib import Path

# Get absolute paths to the database directories
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = Path(current_dir).parent
business_db_path = os.path.join(repo_root, "business_chroma_db")
reviews_db_path = os.path.join(repo_root, "chroma_db")

# Start ChromaDB server for business database (port 8000)
def start_business_db():
    print(f"Starting ChromaDB server for business database on port 8000...")
    print(f"Database path: {business_db_path}")
    cmd = f"chroma run --path {business_db_path} --host 0.0.0.0 --port 8000"
    print(f"Running command: {cmd}")
    
    # Start the process
    process = subprocess.Popen(cmd, shell=True)
    print(f"Business ChromaDB server started with PID: {process.pid}")
    return process

# Start ChromaDB server for reviews database (port 8001)
def start_reviews_db():
    print(f"Starting ChromaDB server for reviews database on port 8001...")
    print(f"Database path: {reviews_db_path}")
    cmd = f"chroma run --path {reviews_db_path} --host 0.0.0.0 --port 8001"
    print(f"Running command: {cmd}")
    
    # Start the process
    process = subprocess.Popen(cmd, shell=True)
    print(f"Reviews ChromaDB server started with PID: {process.pid}")
    return process

if __name__ == "__main__":
    print("Starting ChromaDB servers...")
    
    # Start both servers
    business_process = start_business_db()
    reviews_process = start_reviews_db()
    
    print("\nBoth ChromaDB servers are running!")
    print("- Business database: http://localhost:8000")
    print("- Reviews database: http://localhost:8001")
    print("\nPress Ctrl+C to stop the servers.")
    
    try:
        # Keep the script running
        business_process.wait()
        reviews_process.wait()
    except KeyboardInterrupt:
        print("\nStopping ChromaDB servers...")
        business_process.terminate()
        reviews_process.terminate()
        print("ChromaDB servers stopped.")