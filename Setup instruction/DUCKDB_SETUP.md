# DuckDB Database Foundation

This project now includes a DuckDB foundation that provides fast data access for any tool. The database foundation is designed to be extensible - any future tool can easily leverage it for high-performance data operations.

## Quick Start

### 1. Install DuckDB
```bash
pip install duckdb
```

### 2. Test the Foundation
```bash
python test/test_database_foundation.py
```

### 3. Set Up the Database
```bash
python migration/setup_database.py
```

### 4. Test with BusinessSearchTool
```bash
python -c "from tools.business_search_tool import BusinessSearchTool; tool = BusinessSearchTool(); print('DB Available:', tool.db_available)"
```

## What This Provides

### For Current Tools
- **BusinessSearchTool**: Now uses DuckDB for lightning-fast business lookups instead of loading parquet files
- **Streamlit Interface**: Shows database status and performance metrics

### For Future Tools
- **Ready-to-use database connection**: Just import `from database.db_manager import get_db_manager`
- **Extensible schema**: Add new tables easily by updating `schema/base_schema.sql`
- **Performance tracking**: Built-in query performance monitoring
- **Safe operations**: Thread-safe database operations with proper error handling

## Database Structure

```sql
-- Current tables
businesses (business_id, name, city, state, categories, stars, etc.)
reviews (review_id, business_id, text, stars, date, etc.) -- optional

-- Easy to extend with more tables as needed
```

## How to Use in Your Tools

### Simple Query Example
```python
from database.db_manager import get_db_manager

db_manager = get_db_manager()

# Simple query
businesses = db_manager.execute_query(
    "SELECT * FROM businesses WHERE city = ?", 
    params=['Philadelphia']
)

# Insert data
import pandas as pd
new_data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
db_manager.batch_insert('your_table', new_data)
```

### Check if Database is Available
```python
try:
    from database.db_manager import get_db_manager
    db_manager = get_db_manager()
    result = db_manager.execute_query("SELECT 1")
    db_available = True
except:
    db_available = False
```

## Benefits

1. **Performance**: SQL queries are much faster than loading/filtering CSV/parquet files
2. **Memory Efficient**: Only load the data you need
3. **Extensible**: Easy to add new tables and capabilities
4. **Compatible**: Existing tools continue to work, enhanced tools get better performance
5. **Future-Ready**: Foundation for advanced analytics, caching, and data management

## Architecture

```
Tools Layer:          BusinessSearchTool, FutureDataTool, FutureSentimentTool
                                    ↓
Database Layer:       db_manager.py (thread-safe, performance tracking)
                                    ↓
Storage Layer:        business_analytics.duckdb (single file, portable)
```

This foundation makes it easy to add new tools that need fast data access while keeping the existing CSV/parquet-based tools working exactly as before.
