# test_search_tool.py
from tools import ReviewSearchTool
import torch

def test_search_tool():
    print("Testing ReviewSearchTool...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # Try with CPU first
        search_tool = ReviewSearchTool("index_demo")
        results = search_tool("service quality", k=3)
        print("✓ Search tool works on CPU")
        print(f"Found {len(results)} results")
        
        # Try to move to GPU if available
        if torch.cuda.is_available():
            # Move the embedder to GPU
            search_tool.embedder = search_tool.embedder.to('cuda')
            print("✓ Moved embedder to GPU")
            
            # Test again
            results = search_tool("service quality", k=3)
            print("✓ Search tool works on GPU")
            print(f"Found {len(results)} results")
            
        return True
    except Exception as e:
        print(f"✗ Search tool failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_search_tool()
    if success:
        print("\nSearch tool test passed!")
    else:
        print("\nSearch tool test failed.")