"""
utility.py
Helper functions for data processing and file operations
"""

import os
import csv
import pickle
import numpy as np
from typing import Dict, List, Any, Optional

# Rounding for fairness metrics (F, delta, thresholds).
# F uses 4 dp; delta/threshold use 2 dp so values like 0.101 vs limit 0.1
# do not fail due to floating-point noise.
FAIRNESS_F_DECIMALS = 4
FAIRNESS_DELTA_DECIMALS = 2


def round_fairness_f(value):
    return round(float(value), FAIRNESS_F_DECIMALS)


def round_fairness_delta(value):
    return round(float(value), FAIRNESS_DELTA_DECIMALS)


def round_fairness_threshold(value):
    return round(float(value), FAIRNESS_DELTA_DECIMALS)


def compute_fairness_delta(target_fairness, protected_proportion):
    target = round_fairness_f(target_fairness)
    f_score = round_fairness_f(protected_proportion)
    return round_fairness_delta(target - f_score)


def fairness_delta_exceeds(delta, threshold):
    return round_fairness_delta(delta) > round_fairness_threshold(threshold)


def fairness_delta_within(delta, threshold):
    return round_fairness_delta(delta) <= round_fairness_threshold(threshold)


def average_or_na(values):
    """Mean of numeric values, or 'NA' when the list is empty."""
    if not values:
        return 'NA'
    return sum(values) / len(values)


def fairness_from_delta(target_fairness, delta):
    """Reconstruct F from target_fairness and delta (F = target - delta)."""
    return target_fairness - delta


def collect_final_fairness_from_qres(qres, target_fairness, all_values, successful_values):
    """Append per-query final fairness to all/successful lists (fair pipeline)."""
    if not isinstance(qres, dict):
        return
    final_delta = qres.get('delta')
    if not isinstance(final_delta, (int, float)):
        return
    final_f = fairness_from_delta(target_fairness, final_delta)
    all_values.append(final_f)
    if qres.get('success', False):
        successful_values.append(final_f)


def assert_average_final_fairness_ordering(avg_all, avg_successful, *, tol=1e-9):
    """Ensure average_final_fairness <= average_final_fairness_successful_queries."""
    if avg_all == 'NA' or avg_successful == 'NA':
        return
    if avg_all > avg_successful + tol:
        raise ValueError(
            f"average_final_fairness ({avg_all}) must be <= "
            f"average_final_fairness_successful_queries ({avg_successful})"
        )


class Utility:
    """Collection of utility functions for data processing"""
    
    @staticmethod
    def read_csv_files_to_dict(folder_path):
        """
        Reads all CSV files from the specified folder and creates a dictionary
        mapping file names to lists of columns (including headers) from each file.

        Parameters:
            folder_path (str): The path to the folder containing CSV files.

        Returns:
            dict: A dictionary where keys are file names and values are lists of columns.
        """
        data_dict = {}
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, mode='r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    rows = list(reader)
                    if not rows:
                        continue  # Skip empty files
                    headers = rows[0]
                    # Initialize a list for each column, starting with the header
                    columns = [[header] for header in headers]
                    for row in rows[1:]:
                        for i, value in enumerate(row):
                            # Ensure we don't run into index errors if rows are uneven
                            if i < len(columns):
                                columns[i].append(value)
                            else:
                                # Handle missing columns by appending empty strings
                                columns.append([''] * (len(columns[0]) - 1) + [value])
                    data_dict[filename] = columns
        print(f"Loaded {len(data_dict.keys())} CSV files")
        return data_dict
    
    @staticmethod
    def save_pickle(data: Any, filepath: str) -> None:
        """
        Save data to a pickle file
        
        Parameters:
            data: Data to save
            filepath (str): Path to save the pickle file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved pickle to {filepath}")
    
    @staticmethod
    def load_pickle(filepath: str) -> Any:
        """
        Load data from a pickle file
        
        Parameters:
            filepath (str): Path to the pickle file
            
        Returns:
            The loaded data
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded pickle from {filepath}")
        return data
    
    @staticmethod
    def ensure_dir_exists(directory: str) -> None:
        """
        Create directory if it doesn't exist
        
        Parameters:
            directory (str): Path to directory
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    @staticmethod
    def get_file_count(folder_path: str, extension: Optional[str] = None) -> int:
        """
        Count files in a folder, optionally filtering by extension
        
        Parameters:
            folder_path (str): Path to folder
            extension (str, optional): File extension to filter (e.g., '.csv')
            
        Returns:
            int: Number of files
        """
        if not os.path.exists(folder_path):
            return 0
        
        files = os.listdir(folder_path)
        if extension:
            files = [f for f in files if f.endswith(extension)]
        return len(files)
    
    @staticmethod
    def read_csv_to_dataframe(filepath: str, **kwargs):
        """
        Read a CSV file to pandas DataFrame (if pandas is available)
        
        Parameters:
            filepath (str): Path to CSV file
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            DataFrame or None if pandas not available
        """
        try:
            import pandas as pd
            return pd.read_csv(filepath, **kwargs)
        except ImportError:
            print("Pandas not available. Cannot read CSV to DataFrame.")
            return None
    
    @staticmethod
    def print_dict_summary(data_dict: Dict, max_items: int = 10) -> None:
        """
        Print a summary of a dictionary
        
        Parameters:
            data_dict (dict): Dictionary to summarize
            max_items (int): Maximum number of items to show
        """
        print(f"\nDictionary Summary:")
        print(f"  Total keys: {len(data_dict)}")
        print(f"  First {min(max_items, len(data_dict))} keys:")
        for i, key in enumerate(list(data_dict.keys())[:max_items]):
            value = data_dict[key]
            if isinstance(value, list):
                print(f"    {key}: list with {len(value)} items")
            elif isinstance(value, np.ndarray):
                print(f"    {key}: array with shape {value.shape}")
            else:
                print(f"    {key}: {type(value).__name__}")
    
    @staticmethod
    def get_csv_column_names(filepath: str) -> List[str]:
        """
        Get column names from a CSV file without loading entire file
        
        Parameters:
            filepath (str): Path to CSV file
            
        Returns:
            List[str]: List of column names
        """
        with open(filepath, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader, [])
        return headers
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Parameters:
            vec1 (np.ndarray): First vector
            vec2 (np.ndarray): Second vector
            
        Returns:
            float: Cosine similarity score
        """
        from numpy.linalg import norm
        if norm(vec1) == 0 or norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    
    @staticmethod
    def batch_process(items: List[Any], batch_size: int, process_func, 
                     verbose: bool = True):
        """
        Process items in batches
        
        Parameters:
            items (list): Items to process
            batch_size (int): Size of each batch
            process_func: Function to apply to each batch
            verbose (bool): Print progress
            
        Returns:
            list: Results from processing each batch
        """
        results = []
        total_batches = (len(items) + batch_size - 1) // batch_size
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            if verbose:
                batch_num = i // batch_size + 1
                print(f"Processing batch {batch_num}/{total_batches}...")
            result = process_func(batch)
            results.append(result)
        
        return results
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """
        Format seconds into human-readable time string
        
        Parameters:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.2f}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {mins}m {secs:.2f}s"


# Standalone helper functions (non-class methods)

def load_csv_dict(folder_path: str) -> Dict[str, List[List[str]]]:
    """
    Convenience function to load CSV files to dictionary
    Alias for Utility.read_csv_files_to_dict
    """
    return Utility.read_csv_files_to_dict(folder_path)


def quick_save(data: Any, filename: str) -> None:
    """Quick save to pickle"""
    Utility.save_pickle(data, filename)


def quick_load(filename: str) -> Any:
    """Quick load from pickle"""
    return Utility.load_pickle(filename)


# Example usage
if __name__ == "__main__":
    import sys
    
    print("Utility Functions Demo")
    print("=" * 60)
    
    # Example 1: Read CSV files
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        print(f"\nReading CSV files from: {folder}")
        csv_data = Utility.read_csv_files_to_dict(folder)
        Utility.print_dict_summary(csv_data)
    else:
        print("\nTo test CSV reading, run:")
        print("  python utility.py /path/to/csv/folder")
    
    # Example 2: Time formatting
    print("\n" + "=" * 60)
    print("Time Formatting Examples:")
    print(f"  45.3 seconds = {Utility.format_time(45.3)}")
    print(f"  125.5 seconds = {Utility.format_time(125.5)}")
    print(f"  3725.2 seconds = {Utility.format_time(3725.2)}")
    
    # Example 3: Cosine similarity
    print("\n" + "=" * 60)
    print("Cosine Similarity Example:")
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([2.0, 4.0, 6.0])
    v3 = np.array([1.0, 0.0, -1.0])
    print(f"  v1: {v1}")
    print(f"  v2: {v2}")
    print(f"  v3: {v3}")
    print(f"  sim(v1, v2) = {Utility.cosine_similarity(v1, v2):.4f}")
    print(f"  sim(v1, v3) = {Utility.cosine_similarity(v1, v3):.4f}")