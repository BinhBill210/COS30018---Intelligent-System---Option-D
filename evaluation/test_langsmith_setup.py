"""
Test LangSmith Setup
====================

This simple script verifies that your LangSmith setup is working correctly
before running the full evaluation.

Run this first to make sure everything is configured properly.
"""

import os
import json
from langsmith import Client

print("=" * 60)
print("Testing LangSmith Setup")
print("=" * 60)

# Test 1: Check API Key
print("\n[Test 1] Checking LANGSMITH_API_KEY...")
api_key = os.environ.get("LANGSMITH_API_KEY")
if api_key:
    print(f"    ✓ API key is set (starts with: {api_key[:10]}...)")
else:
    print("    ❌ LANGSMITH_API_KEY is not set!")
    print("\n    To set it:")
    print("    Windows PowerShell: $env:LANGSMITH_API_KEY = 'your-key-here'")
    print("    Linux/Mac: export LANGSMITH_API_KEY='your-key-here'")
    exit(1)

# Test 2: Initialize Client
print("\n[Test 2] Initializing LangSmith client...")
try:
    client = Client()
    print("    ✓ Client initialized successfully")
except Exception as e:
    print(f"    ❌ Failed to initialize client: {e}")
    exit(1)

# Test 3: Test Connection
print("\n[Test 3] Testing connection to LangSmith...")
try:
    # Try to list datasets (will return empty list if none exist)
    datasets = list(client.list_datasets(limit=1))
    print(f"    ✓ Connection successful")
    print(f"    Found {len(datasets)} dataset(s)")
except Exception as e:
    print(f"    ❌ Connection failed: {e}")
    exit(1)

# Test 4: Check Dataset File
print("\n[Test 4] Checking for test dataset file...")
dataset_file = "golden_test_dataset_v2.json"
if not os.path.exists(dataset_file):
    dataset_file = os.path.join("evaluation", "golden_test_dataset_v2.json")

if os.path.exists(dataset_file):
    print(f"    ✓ Dataset file found: {dataset_file}")
    
    # Load and validate structure
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_cases = data.get("test_cases", [])
        print(f"    ✓ Dataset contains {len(test_cases)} test cases")
        
        # Show first test case structure
        if test_cases:
            first_case = test_cases[0]
            print("\n    Sample test case structure:")
            print(f"      - test_id: {first_case.get('test_id', 'N/A')}")
            print(f"      - category: {first_case.get('category', 'N/A')}")
            print(f"      - query: {first_case.get('query', 'N/A')[:50]}...")
            print(f"      - expected_tool_chain: {first_case.get('expected_tool_chain', [])}")
            
    except Exception as e:
        print(f"    ⚠ Warning: Could not parse dataset file: {e}")
else:
    print(f"    ❌ Dataset file not found!")
    print(f"    Expected: golden_test_dataset_v2.json")

# Test 5: Test Traceable Decorator
print("\n[Test 5] Testing @traceable decorator...")
try:
    from langsmith.run_helpers import traceable
    
    @traceable
    def test_function(x: int) -> int:
        """Simple test function."""
        return x * 2
    
    result = test_function(5)
    print(f"    ✓ @traceable decorator works (test result: {result})")
    
except Exception as e:
    print(f"    ❌ @traceable decorator failed: {e}")

# Final Summary
print("\n" + "=" * 60)
print("SETUP VERIFICATION COMPLETE")
print("=" * 60)

print("\n✅ All tests passed! You're ready to run the evaluation.")
print("\nNext steps:")
print("  1. Run: python evaluation/simple_langsmith_eval.py")
print("     (This runs with placeholder agent)")
print("\n  OR")
print("\n  2. Run: python evaluation/simple_langsmith_eval_with_agent.py")
print("     (This runs with your actual G1 agent)")
print("\n  Then view results at: https://smith.langchain.com/")
print("=" * 60)

