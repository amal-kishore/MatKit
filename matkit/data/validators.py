"""Data validation utilities for MatKit."""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure

from matkit.core.exceptions import ValidationError
from matkit.core.logging import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Comprehensive data validation for materials datasets.
    
    Features:
    - Structure validation
    - Composition validation
    - Property range checking
    - Missing data handling
    - Outlier detection
    """
    
    def __init__(self, strict: bool = False):
        """
        Initialize validator.
        
        Parameters
        ----------
        strict : bool, default=False
            Whether to raise errors on validation failure
        """
        self.strict = strict
        self.validation_report = {}
    
    def validate_dataset(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        property_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate entire dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset to validate
        required_columns : list of str, optional
            Required column names
        property_ranges : dict, optional
            Valid ranges for properties {property: (min, max)}
            
        Returns
        -------
        pd.DataFrame
            Validated dataset (potentially with rows removed)
        dict
            Validation report
        """
        logger.info(f"Validating dataset with {len(df)} entries")
        
        original_size = len(df)
        report = {
            "original_size": original_size,
            "issues": [],
            "removed_rows": []
        }
        
        # Check required columns
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                msg = f"Missing required columns: {missing}"
                report["issues"].append(msg)
                if self.strict:
                    raise ValidationError(msg)
        
        # Validate structures if present
        if "structure" in df.columns:
            df, structure_report = self._validate_structures(df)
            report["structure_validation"] = structure_report
        
        # Validate compositions if present
        if "composition" in df.columns or "formula" in df.columns:
            df, comp_report = self._validate_compositions(df)
            report["composition_validation"] = comp_report
        
        # Check property ranges
        if property_ranges:
            df, range_report = self._validate_ranges(df, property_ranges)
            report["range_validation"] = range_report
        
        # Check for duplicates
        df, dup_report = self._check_duplicates(df)
        report["duplicate_check"] = dup_report
        
        # Final statistics
        report["final_size"] = len(df)
        report["removed_count"] = original_size - len(df)
        report["removal_rate"] = report["removed_count"] / original_size if original_size > 0 else 0
        
        self.validation_report = report
        
        logger.info(
            f"Validation complete: {len(df)}/{original_size} entries retained "
            f"({report['removal_rate']:.1%} removed)"
        )
        
        return df, report
    
    def _validate_structures(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate structure data."""
        report = {
            "invalid_structures": [],
            "unreasonable_structures": []
        }
        
        valid_mask = pd.Series(True, index=df.index)
        
        for idx, row in df.iterrows():
            structure = row.get("structure")
            
            if structure is None:
                report["invalid_structures"].append(idx)
                valid_mask[idx] = False
                continue
            
            if not isinstance(structure, Structure):
                try:
                    # Try to convert
                    if isinstance(structure, dict):
                        structure = Structure.from_dict(structure)
                        df.at[idx, "structure"] = structure
                    else:
                        report["invalid_structures"].append(idx)
                        valid_mask[idx] = False
                        continue
                except Exception:
                    report["invalid_structures"].append(idx)
                    valid_mask[idx] = False
                    continue
            
            # Check for reasonable structure
            if len(structure) == 0:
                report["unreasonable_structures"].append((idx, "empty structure"))
                valid_mask[idx] = False
            elif structure.volume < 1.0:  # Unreasonably small
                report["unreasonable_structures"].append((idx, f"volume={structure.volume:.2f}"))
                valid_mask[idx] = False
            elif structure.density < 0.1 or structure.density > 30:  # Unreasonable density
                report["unreasonable_structures"].append((idx, f"density={structure.density:.2f}"))
                valid_mask[idx] = False
            
            # Check for overlapping atoms
            try:
                if structure.distance_matrix.min() < 0.5:  # Atoms too close
                    report["unreasonable_structures"].append((idx, "overlapping atoms"))
                    valid_mask[idx] = False
            except Exception:
                pass
        
        # Filter dataframe
        if not self.strict:
            df = df[valid_mask].copy()
        
        report["removed_count"] = (~valid_mask).sum()
        
        return df, report
    
    def _validate_compositions(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate composition data."""
        report = {
            "invalid_compositions": [],
            "unreasonable_compositions": []
        }
        
        valid_mask = pd.Series(True, index=df.index)
        
        # Determine composition column
        comp_col = "composition" if "composition" in df.columns else "formula"
        
        for idx, row in df.iterrows():
            comp_data = row.get(comp_col)
            
            if comp_data is None:
                report["invalid_compositions"].append(idx)
                valid_mask[idx] = False
                continue
            
            try:
                if isinstance(comp_data, str):
                    comp = Composition(comp_data)
                elif isinstance(comp_data, Composition):
                    comp = comp_data
                else:
                    report["invalid_compositions"].append(idx)
                    valid_mask[idx] = False
                    continue
                
                # Check for reasonable composition
                if len(comp.elements) == 0:
                    report["unreasonable_compositions"].append((idx, "empty composition"))
                    valid_mask[idx] = False
                elif len(comp.elements) > 10:  # Unusually complex
                    report["unreasonable_compositions"].append(
                        (idx, f"n_elements={len(comp.elements)}")
                    )
                    # Don't remove, just flag
                
                # Check for valid oxidation states if available
                try:
                    comp.oxi_state_guesses()
                except Exception:
                    # Not all compositions have valid oxidation states
                    pass
                    
            except Exception as e:
                report["invalid_compositions"].append((idx, str(e)))
                valid_mask[idx] = False
        
        # Filter dataframe
        if not self.strict:
            df = df[valid_mask].copy()
        
        report["removed_count"] = (~valid_mask).sum()
        
        return df, report
    
    def _validate_ranges(
        self,
        df: pd.DataFrame,
        property_ranges: Dict[str, Tuple[float, float]]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate property ranges."""
        report = {
            "out_of_range": {},
            "missing_properties": {}
        }
        
        valid_mask = pd.Series(True, index=df.index)
        
        for prop, (min_val, max_val) in property_ranges.items():
            if prop not in df.columns:
                report["missing_properties"][prop] = len(df)
                continue
            
            # Check for out of range values
            prop_data = df[prop]
            
            # Handle missing values
            missing_mask = prop_data.isna()
            report["missing_properties"][prop] = missing_mask.sum()
            
            # Check ranges for non-missing values
            valid_data = prop_data[~missing_mask]
            out_of_range_mask = (valid_data < min_val) | (valid_data > max_val)
            
            if out_of_range_mask.any():
                out_indices = valid_data[out_of_range_mask].index.tolist()
                report["out_of_range"][prop] = {
                    "count": len(out_indices),
                    "indices": out_indices[:10],  # First 10
                    "values": valid_data[out_of_range_mask].head(10).tolist()
                }
                
                # Mark as invalid
                valid_mask[out_indices] = False
        
        # Filter dataframe
        if not self.strict:
            df = df[valid_mask].copy()
        
        report["removed_count"] = (~valid_mask).sum()
        
        return df, report
    
    def _check_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Check for duplicate entries."""
        report = {
            "duplicate_count": 0,
            "duplicate_columns": []
        }
        
        # Check for exact duplicates
        duplicates = df.duplicated()
        if duplicates.any():
            report["duplicate_count"] = duplicates.sum()
            df = df[~duplicates].copy()
        
        # Check for duplicates in key columns
        key_columns = []
        
        if "material_id" in df.columns:
            key_columns.append("material_id")
        if "formula" in df.columns:
            key_columns.append("formula")
        if "structure" in df.columns and len(df) < 10000:  # Only for small datasets
            # Create structure hash
            df["_structure_hash"] = df["structure"].apply(
                lambda s: hash(str(s)) if s is not None else None
            )
            key_columns.append("_structure_hash")
        
        for col in key_columns:
            if col in df.columns:
                dup_mask = df.duplicated(subset=[col], keep="first")
                if dup_mask.any():
                    report["duplicate_columns"].append({
                        "column": col,
                        "count": dup_mask.sum()
                    })
                    df = df[~dup_mask].copy()
        
        # Clean up temporary columns
        if "_structure_hash" in df.columns:
            df = df.drop("_structure_hash", axis=1)
        
        return df, report
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: float = 3.0
    ) -> Dict[str, List[int]]:
        """
        Detect outliers in numerical columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset
        columns : list of str, optional
            Columns to check. If None, checks all numeric columns
        method : str, default="iqr"
            Method for outlier detection ("iqr", "zscore", "isolation")
        threshold : float, default=3.0
            Threshold for outlier detection
            
        Returns
        -------
        dict
            Outlier indices for each column
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            data = df[col].dropna()
            
            if len(data) == 0:
                continue
            
            if method == "iqr":
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                outlier_mask = (data < lower) | (data > upper)
                
            elif method == "zscore":
                z_scores = np.abs((data - data.mean()) / data.std())
                outlier_mask = z_scores > threshold
                
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            if outlier_mask.any():
                outliers[col] = data[outlier_mask].index.tolist()
        
        return outliers