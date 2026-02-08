"""
Data management module.

Provides two data handler implementations:
- DataHandler: DuckDB-based, high-performance queries
- DataHandlerF: File-based (CSV/Parquet), supports multi-process loading
"""

from .data_handler import DataHandler
from .data_handler_f import DataHandlerF

__all__ = ['DataHandler', 'DataHandlerF']
