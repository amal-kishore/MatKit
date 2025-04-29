# matkit/featurization/additional.py

import pandas as pd

def add_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional engineered features.

    Parameters:
    - df (pd.DataFrame): DataFrame with existing features.

    Returns:
    - pd.DataFrame: DataFrame with additional features.
    """
    df = df.copy()
    df['electroneg_diff'] = df['avg_electronegativity'] * df['mean_bond_length']
    df['charge_density'] = df['avg_oxidation_state'] / df['volume_per_atom']
    df['bond_coord_ratio'] = df['mean_bond_length'] / df['mean_coordination']
    df['electron_density'] = df['avg_valence_electrons'] * df['density']
    df['metallicity'] = df['fraction_metals'] * df['avg_atomic_number']
    df['energy_factor'] = df['avg_electronegativity'] * df['avg_valence_electrons']
    df['volume_density_ratio'] = df['volume_per_atom'] / df['density']
    df['electronegativity_squared'] = df['avg_electronegativity'] ** 2
    df['coord_per_volume'] = df['mean_coordination'] / df['volume_per_atom']
    df['ionic_character'] = df['avg_oxidation_state'] * df['avg_electronegativity']
    return df
