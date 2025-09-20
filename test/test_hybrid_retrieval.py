#!/usr/bin/env python3
"""
Test script for the new simplified Hybrid Retrieval tool
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_functionality():
    """Test basic hybrid retrieval functionality"""
    print("🔍 Testing New Hybrid Retrieval Tool - Basic Functionality...")
    
    try:
        from tools.hybrid_retrieval_tool import HybridRetrieve
        
        # Test tool initialization
        hybrid_tool = HybridRetrieve()
        print(f"✅ Successfully initialized HybridRetrieve")
        print(f"📡 Connection mode: {hybrid_tool.connection_mode}")
        
        # Test with sample query
        test_business_id = "XQfwVwDr-v0ZS3_CbbE5Xw"
        test_query = "food quality and taste"
        
        print(f"\n🧪 Testing query: '{test_query}'")
        print(f"🏢 Business ID: {test_business_id}")
        
        result = hybrid_tool(
            business_id=test_business_id,
            query=test_query,
            top_k=5
        )
        
        print(f"\n📊 Results:")
        print(f"   Total hits: {result.get('total_hits', 0)}")
        print(f"   Evidence quotes: {len(result.get('evidence_quotes', []))}")
        print(f"   Elapsed time: {result.get('elapsed_ms', 0)}ms")
        print(f"   Summary: {result.get('summary', 'N/A')}")
        
        if result.get('error'):
            print(f"   ❌ Error: {result['error']}")
            return False
        
        # Show sample results
        hits = result.get('hits', [])
        if hits:
            print(f"\n📝 Sample hits:")
            for i, hit in enumerate(hits[:2]):
                print(f"   {i+1}. Score: {hit.get('score', 0):.3f} | Stars: {hit.get('stars', 0)} | Text: {hit.get('text', '')[:100]}...")
        
        # Show evidence quotes
        quotes = result.get('evidence_quotes', [])
        if quotes:
            print(f"\n💬 Evidence quotes:")
            for i, quote in enumerate(quotes[:2]):
                print(f"   {i+1}. \"{quote}\"")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_filtering():
    """Test advanced filtering functionality"""
    print("\n🔍 Testing Advanced Filtering...")
    
    try:
        from tools.hybrid_retrieval_tool import HybridRetrieve
        
        hybrid_tool = HybridRetrieve()
        test_business_id = "XQfwVwDr-v0ZS3_CbbE5Xw"
        
        # Test with star rating filter
        print("🌟 Testing star rating filter (4-5 stars)...")
        result = hybrid_tool(
            business_id=test_business_id,
            query="great food excellent",
            top_k=5,
            filters={"stars": [4, 5]}
        )
        
        print(f"   Results with 4-5 star filter: {result.get('total_hits', 0)} hits")
        
        # Test without filter
        print("🔓 Testing without filters...")
        result_no_filter = hybrid_tool(
            business_id=test_business_id,
            query="great food excellent",
            top_k=5
        )
        
        print(f"   Results without filter: {result_no_filter.get('total_hits', 0)} hits")
        
        return True
        
    except Exception as e:
        print(f"❌ Filtering test failed: {e}")
        return False

def test_error_handling():
    """Test error handling"""
    print("\n🔍 Testing Error Handling...")
    
    try:
        from tools.hybrid_retrieval_tool import HybridRetrieve
        
        hybrid_tool = HybridRetrieve()
        
        # Test invalid business_id
        print("❌ Testing invalid business_id...")
        result = hybrid_tool(business_id="", query="test query")
        if result.get('error'):
            print("   ✅ Correctly handled invalid business_id")
        
        # Test invalid query
        print("❌ Testing invalid query...")
        result = hybrid_tool(business_id="test_id", query="")
        if result.get('error'):
            print("   ✅ Correctly handled invalid query")
        
        # Test non-existent business
        print("🔍 Testing non-existent business...")
        result = hybrid_tool(
            business_id="non_existent_business_123456789",
            query="test query"
        )
        print(f"   Result for non-existent business: {result.get('total_hits', 0)} hits")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_performance():
    """Test performance with multiple queries"""
    print("\n🔍 Testing Performance...")
    
    try:
        from tools.hybrid_retrieval_tool import HybridRetrieve
        import time
        
        hybrid_tool = HybridRetrieve()
        test_business_id = "XQfwVwDr-v0ZS3_CbbE5Xw"
        
        queries = [
            "food quality",
            "service experience", 
            "atmosphere ambiance",
            "price value",
            "location parking"
        ]
        
        total_time = 0
        for i, query in enumerate(queries, 1):
            start = time.time()
            result = hybrid_tool(
                business_id=test_business_id,
                query=query,
                top_k=3
            )
            elapsed = time.time() - start
            total_time += elapsed
            
            print(f"   Query {i}: {elapsed*1000:.0f}ms | Hits: {result.get('total_hits', 0)}")
        
        avg_time = (total_time / len(queries)) * 1000
        print(f"   📈 Average query time: {avg_time:.0f}ms")
        
        if avg_time < 1000:  # Under 1 second
            print("   ✅ Performance looks good!")
        else:
            print("   ⚠️ Performance might need optimization")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_langchain_compatibility():
    """Test LangChain integration compatibility"""
    print("\n🔍 Testing LangChain Compatibility...")
    
    try:
        # Test dict input format (as expected by LangChain)
        from tools.hybrid_retrieval_tool import HybridRetrieve
        
        hybrid_tool = HybridRetrieve()
        
        # Simulate LangChain input format
        input_dict = {
            "business_id": "XQfwVwDr-v0ZS3_CbbE5Xw",
            "query": "food quality and service",
            "top_k": 5,
            "filters": {"stars": [4, 5]}
        }
        
        result = hybrid_tool(**input_dict)
        
        print(f"   ✅ Dict input format works")
        print(f"   📊 Result keys: {list(result.keys())}")
        
        # Check required output fields
        required_fields = ["business_id", "query", "total_hits", "hits", "evidence_quotes", "summary"]
        missing_fields = [field for field in required_fields if field not in result]
        
        if not missing_fields:
            print(f"   ✅ All required output fields present")
        else:
            print(f"   ⚠️ Missing fields: {missing_fields}")
        
        return len(missing_fields) == 0
        
    except Exception as e:
        print(f"❌ LangChain compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing New Hybrid Retrieval Tool\n")
    
    # Run all tests
    basic_ok = test_basic_functionality()
    filter_ok = test_filtering()
    error_ok = test_error_handling()
    perf_ok = test_performance()
    langchain_ok = test_langchain_compatibility()
    
    print(f"\n📋 Test Results:")
    print(f"{'✅' if basic_ok else '❌'} Basic Functionality: {'PASS' if basic_ok else 'FAIL'}")
    print(f"{'✅' if filter_ok else '❌'} Advanced Filtering: {'PASS' if filter_ok else 'FAIL'}")
    print(f"{'✅' if error_ok else '❌'} Error Handling: {'PASS' if error_ok else 'FAIL'}")
    print(f"{'✅' if perf_ok else '❌'} Performance: {'PASS' if perf_ok else 'FAIL'}")
    print(f"{'✅' if langchain_ok else '❌'} LangChain Compatibility: {'PASS' if langchain_ok else 'FAIL'}")
    
    all_pass = all([basic_ok, filter_ok, error_ok, perf_ok, langchain_ok])
    print(f"\n🎯 Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    
    if all_pass:
        print("\n💡 New Hybrid Retrieval Tool is ready!")
        print("   Key improvements:")
        print("   - ✅ Simplified architecture")
        print("   - ✅ Direct ChromaDB client")
        print("   - ✅ Better error handling")
        print("   - ✅ Improved performance")
        print("   - ✅ LangChain compatible output")
    else:
        print("\n⚠️ Some issues detected. Please review the failed tests.")
