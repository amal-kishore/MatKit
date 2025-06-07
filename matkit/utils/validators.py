"""Validation utilities for MatKit."""

import pandas as pd
from pymatgen.core import Structure, Composition
from typing import Any, Optional

from matkit.core.exceptions import ValidationError


def validate_structure(structure: Any) -> Structure:
    """
    Validate and convert structure input.
    
    Parameters
    ----------
    structure : Any
        Structure input (Structure object, dict, or file path)
        
    Returns
    -------
    Structure
        Validated pymatgen Structure
        
    Raises
    ------
    ValidationError
        If structure is invalid
    """
    if isinstance(structure, Structure):
        return structure
    elif isinstance(structure, dict):
        try:
            return Structure.from_dict(structure)
        except Exception as e:
            raise ValidationError(f"Invalid structure dict: {e}")
    elif isinstance(structure, str):
        try:
            return Structure.from_file(structure)
        except Exception as e:
            raise ValidationError(f"Failed to load structure from file: {e}")
    else:
        raise ValidationError(f"Invalid structure type: {type(structure)}")


def validate_composition(composition: Any) -> Composition:
    """
    Validate and convert composition input.
    
    Parameters
    ----------
    composition : Any
        Composition input (Composition object or string)
        
    Returns
    -------
    Composition
        Validated pymatgen Composition
        
    Raises
    ------
    ValidationError
        If composition is invalid
    """
    if isinstance(composition, Composition):
        return composition
    elif isinstance(composition, str):
        try:
            return Composition(composition)
        except Exception as e:
            raise ValidationError(f"Invalid composition string: {e}")
    else:
        raise ValidationError(f"Invalid composition type: {type(composition)}")


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[list] = None,
    min_rows: int = 1
) -> pd.DataFrame:
    """
    Validate DataFrame input.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    required_columns : list, optional
        Required column names
    min_rows : int, default=1
        Minimum number of rows
        
    Returns
    -------
    pd.DataFrame
        Validated DataFrame
        
    Raises
    ------
    ValidationError
        If DataFrame is invalid
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"Expected DataFrame, got {type(df)}")
    
    if len(df) < min_rows:
        raise ValidationError(f"DataFrame has {len(df)} rows, need at least {min_rows}")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValidationError(f"Missing required columns: {missing}")
    
    return df