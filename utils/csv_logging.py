"""
CSV Logging Utility for Coordinate Checks

Adapted from nanoGPT-mup (authored by Gavia Gray)
Provides efficient CSV logging with correct config JSON writing.
Used for muP and CompleteP coordinate check experiments.

Example usage:
  csv_logger = CSVLogWrapper(out_dir='path/to/output', config=your_config)
  
  # in train loop
  csv_logger.log({"train/loss": 0.5, "iter": 1, "attn_act_abs_mean": 0.1})
  csv_logger.step()
  
  # at the end
  csv_logger.close()
"""

import re
import os
import csv
import json
import atexit


def exists(x): 
    return x is not None


def transform_format_string(s):
    """
    Transforms a string containing f-string-like expressions to a format
    compatible with str.format().
    """
    pattern = r'\{(\w+)=(:.[^}]*)?\}'
    return re.sub(pattern, lambda m: f"{m.group(1)}={{{m.group(1)}{m.group(2) or ''}}}", s)


class CSVLogWrapper:
    """
    CSV logging wrapper for coordinate check experiments.
    Logs metrics to CSV files for later analysis and plotting.
    """
    
    def __init__(self, logf=None, config=None, out_dir=None):
        """
        Initialize the CSV logger.
        
        Args:
            logf: Optional logging function (e.g., wandb.log)
            config: Dictionary of configuration parameters to save
            out_dir: Output directory for CSV and config files
        """
        self.logf = logf
        self.config = config or {}
        self.log_dict = {}
        self.out_dir = out_dir
        self.csv_data_file = None
        self.csv_header_file = None
        self.csv_writer = None
        self.step_count = 0
        self.ordered_keys = []
        self.header_updated = False
        self.is_finalized = False
        self.no_sync_keyword = 'no_sync'  # Keyword to prevent syncing to wandb

        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            self.setup_csv_writer()
            self.write_config()

        atexit.register(self.close)

    def setup_csv_writer(self):
        """Set up CSV writer with temporary files."""
        self.csv_data_path = os.path.join(self.out_dir, 'log_data.csv.tmp')
        self.csv_header_path = os.path.join(self.out_dir, 'log_header.csv.tmp')
        self.csv_data_file = open(self.csv_data_path, 'w', newline='')
        self.csv_header_file = open(self.csv_header_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_data_file)

    def write_config(self):
        """Write configuration to JSON file."""
        if self.config:
            config_path = os.path.join(self.out_dir, 'config.json')
            with open(config_path, 'w') as f:
                # Convert config to dict if it's not already
                config_dict = dict(self.config) if hasattr(self.config, 'items') else self.config
                json.dump(config_dict, f, indent=2, default=str)

    def log(self, data):
        """
        Log data to be written on the next step.
        
        Args:
            data: Dictionary of metrics to log
        """
        self.log_dict.update(data)
        for key in data:
            if key not in self.ordered_keys:
                self.ordered_keys.append(key)
                self.header_updated = True

    def update_header(self):
        """Update the CSV header file if new keys were added."""
        if self.header_updated:
            header = ['step'] + self.ordered_keys
            with open(self.csv_header_path, 'w', newline='') as header_file:
                csv.writer(header_file).writerow(header)
            self.header_updated = False

    def print(self, format_string, prefix=None):
        """
        Print formatted log data.
        
        Args:
            format_string: Format string with {var=} or {var=:format} syntax
            prefix: Optional prefix to filter keys
        """
        format_string = transform_format_string(format_string)

        if prefix:
            filtered_dict = {k.replace(prefix, ''): v for k, v in self.log_dict.items() if k.startswith(prefix)}
        else:
            filtered_dict = self.log_dict
        filtered_dict = {k.replace('/', '_'): v for k, v in filtered_dict.items()}

        try:
            print(format_string.format(**filtered_dict))
        except KeyError as e:
            print(f"KeyError: {e}. Available keys: {', '.join(filtered_dict.keys())}")
            raise e

    def step(self):
        """
        Commit the current log data to CSV and optionally to wandb.
        Call this at the end of each training iteration.
        """
        if exists(self.logf) and self.log_dict:
            self.logf({k: v for k, v in self.log_dict.items() if self.no_sync_keyword not in k})

        if self.csv_writer and self.log_dict:
            self.update_header()
            row_data = [self.step_count] + [self.log_dict.get(key, '') for key in self.ordered_keys]
            self.csv_writer.writerow(row_data)
            self.csv_data_file.flush()

        self.step_count += 1
        self.log_dict.clear()

    def close(self):
        """Close CSV files and finalize the log."""
        if self.csv_data_file:
            self.csv_data_file.close()
        self.finalize_csv()

    def finalize_csv(self):
        """Merge header and data into final CSV file."""
        if self.is_finalized:
            return
        
        if not self.out_dir:
            return

        csv_final_path = os.path.join(self.out_dir, 'log.csv')

        try:
            with open(csv_final_path, 'w', newline='') as final_csv:
                # Copy header
                if os.path.exists(self.csv_header_path):
                    with open(self.csv_header_path, 'r') as header_file:
                        final_csv.write(header_file.read())

                # Copy data
                if os.path.exists(self.csv_data_path):
                    with open(self.csv_data_path, 'r') as data_file:
                        final_csv.write(data_file.read())
            
            self.is_finalized = True

            # Remove temporary files
            if os.path.exists(self.csv_header_path):
                os.remove(self.csv_header_path)
            if os.path.exists(self.csv_data_path):
                os.remove(self.csv_data_path)
        except Exception as e:
            print(f"Warning: Failed to finalize CSV: {e}")


def create_coord_check_logger(out_dir, config):
    """
    Convenience function to create a CSV logger for coordinate checks.
    
    Args:
        out_dir: Output directory for logs
        config: Configuration dictionary
        
    Returns:
        CSVLogWrapper instance
    """
    return CSVLogWrapper(out_dir=out_dir, config=config)

