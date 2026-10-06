"""
table_metadata.py
Efficient metadata storage for table statistics and categorical distributions
"""

import pickle
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Any, Tuple


class TableMetadata:
    """
    Stores metadata for a single table efficiently
    """
    
    def __init__(self, table_name: str):
        """
        Initialize metadata for a table
        
        Args:
            table_name: Name of the table
        """
        self.table_name = table_name
        self.num_records = 0
        self.num_columns = 0
        self.column_names = []
        
        # For categorical columns: {column_name: {category: count}}
        self.categorical_distributions = {}
        
        # Store which columns are categorical
        self.categorical_columns = set()
        
        # Store column indices for quick lookup
        self._column_index = {}
    
    def _resolve_column_identifier(self, column_identifier):
        """
        Resolve column identifier (int or str) to column name
        
        Args:
            column_identifier: Either column index (int) or column name (str)
        
        Returns:
            str: Column name, or None if not found
        """
        if isinstance(column_identifier, int):
            # Column identifier is an index
            if 0 <= column_identifier < len(self.column_names):
                return self.column_names[column_identifier]
            return None
        elif isinstance(column_identifier, str):
            # Column identifier is already a name
            if column_identifier in self.column_names:
                return column_identifier
            return None
        return None
    
    def add_categorical_column(self, column_identifier, distribution: Dict[str, int]):
        """
        Add categorical column distribution
        
        Args:
            column_identifier: Column index (int) or column name (str)
            distribution: Dictionary of {category_value: count}
        """
        column_name = self._resolve_column_identifier(column_identifier)
        if column_name is None:
            raise ValueError(f"Invalid column identifier: {column_identifier}")
        
        self.categorical_columns.add(column_name)
        self.categorical_distributions[column_name] = distribution
    
    def get_category_count(self, column_identifier, category_value: str) -> int:
        """
        Get count for a specific category in a column
        
        Args:
            column_identifier: Column index (int) or column name (str)
            category_value: The category value
            
        Returns:
            Count of records with that category, or 0 if not found
        """
        column_name = self._resolve_column_identifier(column_identifier)
        if column_name is None or column_name not in self.categorical_distributions:
            return 0
        return self.categorical_distributions[column_name].get(category_value, 0)
    
    def get_column_distribution(self, column_identifier) -> Optional[Dict[str, int]]:
        """
        Get full distribution for a column
        
        Args:
            column_identifier: Column index (int) or column name (str)
            
        Returns:
            Dictionary of {category: count} or None if not categorical
        """
        column_name = self._resolve_column_identifier(column_identifier)
        if column_name is None:
            return None
        return self.categorical_distributions.get(column_name)
    
    def is_categorical(self, column_identifier) -> bool:
        """
        Check if a column is categorical
        
        Args:
            column_identifier: Column index (int) or column name (str)
        """
        column_name = self._resolve_column_identifier(column_identifier)
        if column_name is None:
            return False
        return column_name in self.categorical_columns
    
    def get_column_proportions(self, column_identifier) -> Optional[Dict[str, float]]:
        """
        Get proportions for each category in a column
        
        Args:
            column_identifier: Column index (int) or column name (str)
        
        Returns:
            Dictionary of {category: proportion} or None if not categorical
        """
        column_name = self._resolve_column_identifier(column_identifier)
        if column_name is None or column_name not in self.categorical_distributions:
            return None
        
        dist = self.categorical_distributions[column_name]
        total = sum(dist.values())
        if total == 0:
            return None
        
        return {cat: count / total for cat, count in dist.items()}
    
    def get_column_name(self, column_index: int) -> Optional[str]:
        """
        Get column name by index
        
        Args:
            column_index: Column index
        
        Returns:
            Column name or None if index out of range
        """
        if 0 <= column_index < len(self.column_names):
            return self.column_names[column_index]
        return None
    
    def get_column_index(self, column_name: str) -> Optional[int]:
        """
        Get column index by name
        
        Args:
            column_name: Column name
        
        Returns:
            Column index or None if not found
        """
        try:
            return self.column_names.index(column_name)
        except ValueError:
            return None
    
    def __repr__(self):
        return (f"TableMetadata(name={self.table_name}, "
                f"records={self.num_records}, "
                f"columns={self.num_columns}, "
                f"categorical={len(self.categorical_columns)})")
class MetadataStore:
    """
    Efficient storage and retrieval of metadata for multiple tables
    Uses memory-efficient structures and provides fast lookups
    Stores value distributions for ALL columns.
    """
    
    def __init__(self):
        """
        Initialize metadata store
        Distributions are computed for all columns (no categorical threshold).
        """
        self._metadata = {}  # {table_name: TableMetadata}
        self._global_stats = {
            'total_tables': 0,
            'total_records': 0,
            'total_columns_with_distributions': 0
        }
    
    def build_metadata_from_csv(self, folder_path: str, 
                                file_extension: str = '.csv',
                                max_files: Optional[int] = None) -> None:
        """
        Build metadata from CSV files in a folder
        
        Args:
            folder_path: Path to folder containing CSV files
            file_extension: File extension to look for (default: '.csv')
            max_files: Maximum number of files to process (None = all)
        """
        import os
        import csv
        
        files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
        if max_files:
            files = files[:max_files]
        
        print(f"Building metadata for {len(files)} files...")
        
        for filename in files:
            filepath = os.path.join(folder_path, filename)
            self._process_csv_file(filepath, filename)
        
        self._update_global_stats()
        print(f"✓ Metadata built for {len(self._metadata)} tables")
    
    def build_metadata_from_tables(self, tables: Dict[str, List[List[str]]]) -> None:
        """
        Build metadata from pre-loaded table dictionary
        
        Args:
            tables: Dictionary of {table_name: [[col1_values], [col2_values], ...]}
                   where each inner list is a column with header as first element
        """
        print(f"Building metadata for {len(tables)} tables...")
        
        for table_name, columns in tables.items():
            self._process_table_columns(table_name, columns)
        
        self._update_global_stats()
        print(f"✓ Metadata built for {len(self._metadata)} tables")
    
    def _process_csv_file(self, filepath: str, table_name: str) -> None:
        """Process a single CSV file and extract metadata"""
        import csv
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if not rows:
            return
        
        # Create metadata object
        metadata = TableMetadata(table_name)
        headers = rows[0]
        data_rows = rows[1:]
        
        metadata.num_records = len(data_rows)
        metadata.num_columns = len(headers)
        metadata.column_names = headers
        metadata._column_index = {name: i for i, name in enumerate(headers)}
        
        # Process each column - compute distribution for ALL columns
        for col_idx, col_name in enumerate(headers):
            # Extract column values
            col_values = [row[col_idx] if col_idx < len(row) else '' 
                         for row in data_rows]
            
            # Compute distribution for all columns
            if len(col_values) > 0:
                distribution = Counter(col_values)
                metadata.add_categorical_column(col_name, dict(distribution))
        
        self._metadata[table_name] = metadata
    
    def _process_table_columns(self, table_name: str, columns: List[List[str]]) -> None:
        """Process pre-loaded table columns"""
        
        metadata = TableMetadata(table_name)
        
        if not columns or not columns[0]:
            return
        
        # First element of each column is the header
        headers = [col[0] if col else '' for col in columns]
        metadata.column_names = headers
        metadata.num_columns = len(columns)
        metadata.num_records = max(len(col) - 1 for col in columns) if columns else 0
        metadata._column_index = {name: i for i, name in enumerate(headers)}
        
        # Process each column - compute distribution for ALL columns
        for col_idx, column in enumerate(columns):
            col_name = headers[col_idx]
            col_values = column[1:]  # Skip header
            
            # Compute distribution for all columns
            if len(col_values) > 0:
                distribution = Counter(col_values)
                metadata.add_categorical_column(col_name, dict(distribution))
        
        self._metadata[table_name] = metadata
    
    def get_metadata(self, table_name: str) -> Optional[TableMetadata]:
        """
        Get metadata for a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            TableMetadata object or None if not found
        """
        return self._metadata.get(table_name)
    def get_category_count(self, table_name: str, column_identifier, 
                        category_value: str) -> int:
        """
        Fast lookup: Get count for a specific category
        
        Args:
            table_name: Name of the table
            column_identifier: Column index (int) or column name (str)
            category_value: The category value
            
        Returns:
            Count of records with that category, or 0 if not found
        """
        metadata = self._metadata.get(table_name)
        if not metadata:
            return 0
        return metadata.get_category_count(column_identifier, category_value)
    
    def get_num_records(self, table_name: str) -> int:
        """Get number of records in a table"""
        metadata = self._metadata.get(table_name)
        return metadata.num_records if metadata else 0
    
    def get_distribution(self, table_name: str, column_identifier) -> Optional[Dict[str, int]]:
        """
        Get distribution for a column
        
        Args:
            table_name: Name of the table
            column_identifier: Column index (int) or column name (str)
        """
        metadata = self._metadata.get(table_name)
        if not metadata:
            return None
        return metadata.get_column_distribution(column_identifier)
    

    def get_proportions(self, table_name: str, column_identifier) -> Optional[Dict[str, float]]:
        """
        Get proportions for a column
        
        Args:
            table_name: Name of the table
            column_identifier: Column index (int) or column name (str)
        """
        metadata = self._metadata.get(table_name)
        if not metadata:
            return None
        return metadata.get_column_proportions(column_identifier)
    
    
    def compute_union_distribution(self, table_names: List[str], 
                                column_identifier) -> Optional[Dict[str, int]]:
        """
        Compute combined distribution across multiple tables for a column
        
        Args:
            table_names: List of table names
            column_identifier: Column index (int) or column name (str)
            
        Returns:
            Combined distribution or None if column not categorical in any table
        """
        combined = Counter()
        found_any = False
        
        for table_name in table_names:
            dist = self.get_distribution(table_name, column_identifier)
            if dist:
                combined.update(dist)
                found_any = True
        
        return dict(combined) if found_any else None
    
    def is_categorical(self, table_name: str, column_identifier) -> bool:
        """
        Check if a column is categorical in a table
        
        Args:
            table_name: Name of the table
            column_identifier: Column index (int) or column name (str)
        """
        metadata = self._metadata.get(table_name)
        if not metadata:
            return False
        return metadata.is_categorical(column_identifier)
    
    def get_categorical_columns(self, table_name: str) -> Set[str]:
        """Get all categorical columns in a table"""
        metadata = self._metadata.get(table_name)
        if not metadata:
            return set()
        return metadata.categorical_columns.copy()
    
    def _update_global_stats(self):
        """Update global statistics"""
        self._global_stats['total_tables'] = len(self._metadata)
        self._global_stats['total_records'] = sum(
            m.num_records for m in self._metadata.values()
        )
        self._global_stats['total_columns_with_distributions'] = sum(
            len(m.categorical_columns) for m in self._metadata.values()
        )
    
    def save(self, filepath: str) -> None:
        """Save metadata store to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Metadata saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'MetadataStore':
        """Load metadata store from disk"""
        with open(filepath, 'rb') as f:
            store = pickle.load(f)
        print(f"✓ Metadata loaded from {filepath}")
        return store
    
    def get_stats(self) -> Dict[str, Any]:
        """Get global statistics"""
        return self._global_stats.copy()
    
    def __len__(self):
        return len(self._metadata)
    
    def __repr__(self):
        return (f"MetadataStore(tables={len(self._metadata)}, "
                f"records={self._global_stats['total_records']}, "
                f"cols_with_distributions={self._global_stats['total_columns_with_distributions']})")


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Build from CSV files
    print("="*60)
    print("EXAMPLE 1: Build from CSV folder")
    print("="*60)
    
    store = MetadataStore()  # Distributions computed for all columns
    # store.build_metadata_from_csv('/path/to/csv/folder')
    
    # Example 2: Build from pre-loaded tables
    print("\n" + "="*60)
    print("EXAMPLE 2: Build from dictionary")
    print("="*60)
    
    # Sample data: {table_name: [[col1_data], [col2_data], ...]}
    sample_tables = {
        'customers.csv': [
            ['customer_id', '1', '2', '3', '4', '5'],
            ['gender', 'M', 'F', 'M', 'F', 'M'],
            ['age_group', 'young', 'young', 'old', 'old', 'young']
        ],
        'orders.csv': [
            ['order_id', '101', '102', '103'],
            ['status', 'completed', 'pending', 'completed']
        ]
    }
    
    store.build_metadata_from_tables(sample_tables)
    
    # Query metadata
    print("\n" + "-"*60)
    print("QUERIES:")
    print("-"*60)
    
    # Get number of records
    print(f"Records in 'customers.csv': {store.get_num_records('customers.csv')}")
    
    # Get category count
    count = store.get_category_count('customers.csv', 'gender', 'M')
    print(f"Males in customers: {count}")
    
    # Get distribution
    dist = store.get_distribution('customers.csv', 'gender')
    print(f"Gender distribution: {dist}")
    
    # Get proportions
    props = store.get_proportions('customers.csv', 'gender')
    print(f"Gender proportions: {props}")
    
    # Check if categorical
    print(f"Is 'gender' categorical? {store.is_categorical('customers.csv', 'gender')}")
    
    # Union distribution
    union_dist = store.compute_union_distribution(
        ['customers.csv', 'orders.csv'], 'status'
    )
    print(f"Union distribution for 'status': {union_dist}")
    
    # Global stats
    print(f"\nGlobal stats: {store.get_stats()}")
    print(f"Store: {store}")
    
    # Save and load
    # store.save('metadata_store.pkl')
    # loaded_store = MetadataStore.load('metadata_store.pkl')