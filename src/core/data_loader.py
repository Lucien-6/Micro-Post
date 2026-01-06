"""
Data loading and validation module.

This module handles searching for Excel trajectory files,
loading parameters and object data, and validating parameter consistency.
"""

import os
import pandas as pd
from typing import Dict, List, Tuple, Optional

from src.utils.logger import get_logger


class DataLoader:
    """
    Handles loading and validation of trajectory data from Excel files.

    Attributes:
        root_folder: Root folder path selected by user.
        excel_files: List of discovered Excel file paths.
        parameters: Parameters from the first file (reference).
        excluded_params: Parameter names excluded from comparison.
    """

    # Parameters excluded from consistency check
    EXCLUDED_PARAMS = [
        "Video Path",
        "Mask Directory",
        "Exclude Object IDs",
        "Total Objects",
        "Passed Objects",
        "Filtered Objects"
    ]

    def __init__(self, root_folder: str):
        """
        Initialize DataLoader with root folder path.

        Args:
            root_folder: Path to the root folder containing trajectory data.
        """
        self.logger = get_logger()
        self.root_folder = root_folder
        self.excel_files: List[str] = []
        self.parameters: Dict[str, any] = {}
        self.excluded_params = self.EXCLUDED_PARAMS

    def search_excel_files(self) -> List[str]:
        """
        Search for trajectory Excel files in root folder and subfolders.

        Searches up to 3 levels of subdirectories for files matching
        the pattern 'Trajectories_Results_*.xlsx'.

        Returns:
            List of absolute paths to discovered Excel files.
        """
        self.excel_files = []

        def search_directory(directory: str, level: int):
            """Recursively search directory up to specified level."""
            if level > 3:
                return

            try:
                for entry in os.scandir(directory):
                    if entry.is_file() and entry.name.startswith(
                        "Trajectories_Results_"
                    ) and entry.name.endswith(".xlsx"):
                        self.excel_files.append(entry.path)
                        self.logger.debug(f"Found file: {entry.path}")
                    elif entry.is_dir():
                        search_directory(entry.path, level + 1)
            except PermissionError:
                self.logger.warning(
                    f"Permission denied accessing: {directory}"
                )
            except Exception as e:
                self.logger.error(f"Error searching {directory}: {e}")

        search_directory(self.root_folder, 1)
        self.excel_files.sort()
        self.logger.info(
            f"Found {len(self.excel_files)} trajectory files"
        )
        return self.excel_files

    def load_parameters(self, filepath: str) -> Dict[str, any]:
        """
        Load parameters from the Parameters sheet of an Excel file.

        Args:
            filepath: Path to the Excel file.

        Returns:
            Dictionary with parameter names as keys and values as values.
        """
        try:
            df = pd.read_excel(
                filepath,
                sheet_name="Parameters",
                engine="openpyxl"
            )

            params = {}
            for _, row in df.iterrows():
                param_name = row["Parameter"]
                param_value = row["Value"]
                params[param_name] = param_value

            return params

        except Exception as e:
            self.logger.error(
                f"Failed to load parameters from {filepath}: {e}"
            )
            return {}

    def validate_parameters(self) -> Tuple[bool, List[Dict]]:
        """
        Validate that all files have consistent parameters.

        Compares parameters from all files against the first file,
        excluding Video Path, Mask Directory, and Exclude Object IDs.

        Returns:
            Tuple of (is_consistent, inconsistencies).
            - is_consistent: True if all parameters match.
            - inconsistencies: List of dicts with inconsistency details:
              {"file": path, "param": name, "expected": value, "actual": value}
        """
        if not self.excel_files:
            self.logger.error("No Excel files to validate")
            return False, []

        # Load reference parameters from first file
        self.parameters = self.load_parameters(self.excel_files[0])
        if not self.parameters:
            self.logger.error(
                f"Failed to load reference parameters from {self.excel_files[0]}"
            )
            return False, []

        inconsistencies = []

        # Compare each subsequent file against reference
        for filepath in self.excel_files[1:]:
            file_params = self.load_parameters(filepath)

            for param_name, expected_value in self.parameters.items():
                # Skip excluded parameters
                if param_name in self.excluded_params:
                    continue

                actual_value = file_params.get(param_name)

                # Handle NaN comparison
                if pd.isna(expected_value) and pd.isna(actual_value):
                    continue

                if actual_value != expected_value:
                    inconsistencies.append({
                        "file": filepath,
                        "param": param_name,
                        "expected": expected_value,
                        "actual": actual_value
                    })
                    self.logger.warning(
                        f"Parameter mismatch in {filepath}: "
                        f"{param_name} = {actual_value} (expected {expected_value})"
                    )

        is_consistent = len(inconsistencies) == 0
        if is_consistent:
            self.logger.info("All parameters are consistent")
        else:
            self.logger.warning(
                f"Found {len(inconsistencies)} parameter inconsistencies"
            )

        return is_consistent, inconsistencies

    def load_object_sheets(
        self,
        filepath: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all Object sheets from an Excel file.

        Args:
            filepath: Path to the Excel file.

        Returns:
            Dictionary mapping sheet names to DataFrames.
        """
        objects = {}

        try:
            xl = pd.ExcelFile(filepath, engine="openpyxl")

            for sheet_name in xl.sheet_names:
                if sheet_name.startswith("Object_"):
                    df = pd.read_excel(
                        filepath,
                        sheet_name=sheet_name,
                        engine="openpyxl"
                    )
                    objects[sheet_name] = df
                    self.logger.debug(
                        f"Loaded {sheet_name} from {filepath}: "
                        f"{len(df)} rows"
                    )

            return objects

        except Exception as e:
            self.logger.error(
                f"Failed to load objects from {filepath}: {e}"
            )
            return {}

    def get_reference_parameters(self) -> Dict[str, any]:
        """
        Get the reference parameters (from first file).

        Returns:
            Dictionary of parameters, excluding Video Path,
            Mask Directory, and Exclude Object IDs.
        """
        filtered_params = {}
        for key, value in self.parameters.items():
            if key not in self.excluded_params:
                filtered_params[key] = value
        return filtered_params
