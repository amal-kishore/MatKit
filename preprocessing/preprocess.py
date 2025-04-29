# matkit/preprocessing/preprocess.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer

def preprocess_data(df: pd.DataFrame, numerical_cols: list, categorical_cols: list):
    """
    Preprocess the data using scaling and encoding.

    Parameters:
    - df (pd.DataFrame): DataFrame to preprocess.
    - numerical_cols (list): List of numerical column names.
    - categorical_cols (list): List of categorical column names.

    Returns:
    - Pipeline: Preprocessing pipeline
::contentReference[oaicite:13]{index=13}
 
