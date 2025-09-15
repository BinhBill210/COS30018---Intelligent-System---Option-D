"""
General-purpose DuckDB Database Manager
Provides a foundation that any tool can use for fast data access
"""

import duckdb
import pandas as pd
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)

class DuckDBManager:
    """
    General-purpose DuckDB connection and query manager
    Designed to be used by any tool that needs fast data access
    """
    
    def __init__(self, db_path: str = "business_analytics.duckdb"):
        self.db_path = db_path
        self._connection = None
        self._lock = threading.Lock()
        self.query_stats = []
        
        # Performance tracking
        self.total_queries = 0
        
    def get_connection(self):
        """Get or create DuckDB connection"""
        if self._connection is None:
            self._connection = duckdb.connect(self.db_path)
            # Configure for good performance
            self._connection.execute("SET memory_limit='2GB'")
            self._connection.execute("SET threads=4")
            
        return self._connection
    
    def execute_query(self, query: str, params: Optional[List] = None) -> pd.DataFrame:
        """
        Execute query and return results as DataFrame
        
        Args:
            query: SQL query string
            params: Optional parameters for prepared statements
            
        Returns:
            Query results as pandas DataFrame
        """
        start_time = time.time()
        self.total_queries += 1
        
        try:
            with self._lock:
                conn = self.get_connection()
                
                if params:
                    result = conn.execute(query, params).df()
                else:
                    result = conn.execute(query).df()
                
                execution_time = time.time() - start_time
                logger.debug(f"Query executed in {execution_time:.3f}s, {len(result)} rows returned")
                
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query failed after {execution_time:.3f}s: {str(e)}")
            logger.error(f"Query: {query[:200]}...")
            raise
            
    def execute_update(self, query: str, params: Optional[List] = None) -> None:
        """Execute UPDATE/INSERT/DELETE/CREATE queries"""
        start_time = time.time()
        
        try:
            with self._lock:
                conn = self.get_connection()
                
                if params:
                    conn.execute(query, params)
                else:
                    conn.execute(query)
                
                execution_time = time.time() - start_time
                logger.debug(f"Update query executed in {execution_time:.3f}s")
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Update query failed after {execution_time:.3f}s: {str(e)}")
            raise
            
    def batch_insert(self, table_name: str, df: pd.DataFrame) -> None:
        """
        High-performance batch insert using DuckDB's pandas integration
        """
        start_time = time.time()
        total_rows = len(df)
        
        try:
            with self._lock:
                conn = self.get_connection()
                # Use DuckDB's fast pandas integration
                conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
                
                execution_time = time.time() - start_time
                logger.info(f"Batch inserted {total_rows} rows into {table_name} "
                           f"in {execution_time:.3f}s ({total_rows/execution_time:.0f} rows/sec)")
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Batch insert failed after {execution_time:.3f}s: {str(e)}")
            raise
            
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        try:
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
            result = self.execute_query(query, [table_name])
            return not result.empty
        except:
            return False
            
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get table information"""
        try:
            # Get basic table info
            schema_info = self.execute_query(f"DESCRIBE {table_name}")
            row_count = self.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
            
            return {
                'schema': schema_info.to_dict('records') if not schema_info.empty else [],
                'row_count': int(row_count.iloc[0, 0]) if not row_count.empty else 0
            }
        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {e}")
            return {'schema': [], 'row_count': 0}
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'total_queries': self.total_queries,
            'database_path': self.db_path,
            'database_exists': Path(self.db_path).exists(),
            'database_size_mb': round(Path(self.db_path).stat().st_size / (1024*1024), 2) if Path(self.db_path).exists() else 0
        }
        
    def close(self):
        """Close database connection"""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")

# Global instance for easy access
_db_manager: Optional[DuckDBManager] = None

def get_db_manager() -> DuckDBManager:
    """Get global DuckDB manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DuckDBManager()
    return _db_manager

def close_db_manager():
    """Close global database manager"""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None
