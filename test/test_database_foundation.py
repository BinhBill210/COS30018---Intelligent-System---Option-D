from dotenv import load_dotenv
load_dotenv()
#!/usr/bin/env python3
"""
Test script for the database foundation
Verifies that the DuckDB setup works correctly
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_database_basics():
    """Test basic database functionality"""
    print("🧪 Testing Database Foundation")
    print("=" * 50)
    
    try:
        # Test 1: Import database manager
        print("1. Testing database manager import...")
        from database.db_manager import get_db_manager
        db_manager = get_db_manager()
        print("   ✅ Database manager imported successfully")
        
        # Test 2: Test basic query (this will create the database file)
        print("2. Testing basic database connection...")
        result = db_manager.execute_query("SELECT 1 as test")
        assert not result.empty
        assert result.iloc[0, 0] == 1
        print("   ✅ Database connection working")
        
        # Test 3: Test table creation
        print("3. Testing table creation...")
        db_manager.execute_update("""
        CREATE TABLE IF NOT EXISTS test_table (
            id INTEGER PRIMARY KEY,
            name VARCHAR(100)
        )
        """)
        print("   ✅ Table creation working")
        
        # Test 4: Test data insertion
        print("4. Testing data operations...")
        import pandas as pd
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Test1', 'Test2', 'Test3']
        })
        db_manager.batch_insert("test_table", test_data)
        
        result = db_manager.execute_query("SELECT COUNT(*) as count FROM test_table")
        assert result.iloc[0, 0] == 3
        print("   ✅ Data insertion working")
        
        # Test 5: Test performance stats
        print("5. Testing performance tracking...")
        stats = db_manager.get_performance_stats()
        assert 'total_queries' in stats
        assert stats['total_queries'] > 0
        print("   ✅ Performance tracking working")
        
        # Cleanup
        db_manager.execute_update("DROP TABLE IF EXISTS test_table")
        
        print("\n✅ All database foundation tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Database test failed: {e}")
        return False

def test_business_search_integration():
    """Test BusinessSearchTool integration"""
    print("\n🏢 Testing BusinessSearchTool Integration")
    print("=" * 50)
    
    try:
        # Test BusinessSearchTool import
        print("1. Testing BusinessSearchTool import...")
        from tools.business_search_tool import BusinessSearchTool
        tool = BusinessSearchTool()
        print("   ✅ BusinessSearchTool imported successfully")
        
        # Test database availability
        print("2. Testing database availability in tool...")
        if tool.db_available:
            print("   ✅ BusinessSearchTool can access DuckDB")
        else:
            print("   ⚠️  BusinessSearchTool cannot access DuckDB (database not set up)")
        
        # Test graceful degradation
        print("3. Testing graceful degradation...")
        result = tool.get_business_id("NonExistentBusiness")
        # Should return None without crashing
        print("   ✅ Tool handles missing data gracefully")
        
        print("\n✅ BusinessSearchTool integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ BusinessSearchTool integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Database Foundation Test Suite")
    print("=" * 60)
    
    success1 = test_database_basics()
    success2 = test_business_search_integration()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 All tests passed! Database foundation is working correctly.")
        print("\nNext steps:")
        print("1. Run setup: python migration/setup_database.py")
        print("2. Test with real data: python run_langchain_chat.py")
        print("3. Try Streamlit: streamlit run streamlit_agent.py")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
