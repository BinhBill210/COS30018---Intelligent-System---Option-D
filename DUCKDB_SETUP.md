# DuckDB Analytics Database Setup

The DuckDB database file (`business_analytics.duckdb`) is excluded from git due to its size (~200MB+), but can be quickly recreated using the setup script.

## 🚀 Quick Setup

```bash
# 1. Make sure you have the processed data files
ls data/processed/  # Should show business_cleaned.parquet and review_cleaned.parquet

# 2. Run the automated setup script
python setup_duckdb_database.py
```

That's it! The script will:
- ✅ Install required dependencies (duckdb, pandas)
- ✅ Create optimized database with indexes
- ✅ Load 150K+ businesses and 1.4M+ reviews
- ✅ Validate with performance tests

## 📋 Prerequisites

1. **Python 3.8+** with conda/pip
2. **Processed data files** in `data/processed/`:
   - `business_cleaned.parquet` (or `.csv`)
   - `review_cleaned.parquet` (or `.csv`)
3. **At least 2GB RAM** and 1GB disk space

## 📂 Getting the Data Files

If you don't have the processed data files:

### Option 1: Run Preprocessing (Recommended)
```bash
python scripts/run_preprocessing.py
```

### Option 2: Download Processed Data
Download from the shared drive:
```
https://drive.google.com/drive/folders/1enrB0_dKmCJG62NjTBqRG_pZF76Xv4z9
```
Place files in `data/processed/` directory.

## 🔧 Manual Setup (Alternative)

If the automated script doesn't work:

```python
# 1. Install DuckDB
pip install duckdb pandas

# 2. Create database manually
python load_data_to_duckdb.py

# 3. Test the setup
python demo_duckdb_implementation.py
```

## ⚡ What You Get

After setup, you'll have:

- **📊 High-Performance Analytics**: 10x+ faster than Pandas
- **🔍 Complex Queries**: Millisecond aggregations on 1.4M+ reviews  
- **📈 Business Intelligence**: Trend analysis, competitive insights
- **🤖 Enhanced Agent**: LangChain agent with DuckDB tools

## 🧪 Testing Your Setup

```python
# Test basic functionality
from duckdb_integration import DuckDBAnalytics
analytics = DuckDBAnalytics("business_analytics.duckdb")
info = analytics.get_database_info()
print(f"Reviews: {info['statistics']['reviews']:,}")

# Test analytics tools
from tools.duckdb_analytics_tools import BusinessPerformanceTool
perf_tool = BusinessPerformanceTool()
overview = perf_tool(analysis_type="overview")
print(f"Businesses: {overview['total_businesses']:,}")

# Run full demonstration
python demo_duckdb_implementation.py
```

## 📊 Performance Benchmarks

On a typical system, you should see:
- **Database creation**: 2-5 minutes
- **Monthly aggregation queries**: <100ms
- **Complex analytics**: <500ms
- **Database size**: ~200MB

## 🐛 Troubleshooting

### Error: "No data files found"
```bash
# Check if data files exist
ls -la data/processed/

# If missing, run preprocessing
python scripts/run_preprocessing.py
```

### Error: "Module not found"
```bash
# Install dependencies
pip install duckdb pandas numpy

# Or use conda
conda install duckdb pandas numpy
```

### Error: "Permission denied"
```bash
# Make sure DuckDB file isn't open in another process
# Close any Python scripts using the database
# Try running setup script again
```

### Error: "Out of memory"
```bash
# Reduce chunk size in duckdb_integration.py
# Or use a machine with more RAM (2GB+ recommended)
```

## 🔄 Updating the Database

If you get new data or need to refresh:

```bash
# Remove old database
rm business_analytics.duckdb

# Recreate with new data
python setup_duckdb_database.py
```

## 📚 Files Created

After successful setup:
- ✅ `business_analytics.duckdb` - Main analytics database
- ✅ All DuckDB analytics tools work
- ✅ Enhanced LangChain agent ready
- ✅ Hybrid ChromaDB+DuckDB system ready

## 🎯 Next Steps

1. **Run the enhanced agent**: `python langchain_agent_duckdb.py`
2. **Try the demo**: `python demo_duckdb_implementation.py`
3. **Use in your project**: Import tools from `tools/duckdb_analytics_tools.py`

---

**📞 Need Help?**

If you encounter issues:
1. Check the error messages carefully
2. Ensure you have the latest code version
3. Verify data files are in the correct location
4. Try the manual setup steps

The DuckDB integration provides 10x+ performance improvements for analytics queries while maintaining all existing functionality!
