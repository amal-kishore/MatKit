# matkit/featurization/structure.py

import numpy as np
import pandas as pd
from pymatgen.core import Structure

def featurize_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute structure-based features.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'structure' column.

    Returns:
    - pd.DataFrame: DataFrame with new structure features.
    """
    features = {
        'density': [],
        'volume_per_atom': [],
        'cation_to_anion_ratio': [],
        'mean_atomic_number': [],
        'mean_bond_length': [],
        'mean_coordination': []
    }

    for struct in df['structure']:
        try:
            density = struct.density
            volume = struct.volume
            num_sites = struct.num_sites
            volume_per_atom = volume / num_sites if num_sites else np.nan

            cations = sum(1 for site in struct if site.species_string != 'O')
            anions = num_sites - cations
            cation_to_anion_ratio = cations / anions if anions else np.nan

            atomic_numbers = [site.specie.Z for site in struct]
            mean_atomic_number = np.mean(atomic_numbers) if atomic_numbers else np.nan

            bond_lengths = []
            for i, site in enumerate(struct):
                neighbors = struct.get_neighbors(i, 5.0)
                bond_lengths.extend([neighbor[1] for neighbor in neighbors])
            mean_bond_length = np.mean(bond_lengths) if bond_lengths else np.nan

            coordination_numbers = [len(struct.get_neighbors(i, 5.0)) for i in range(num_sites)]
            mean_coordination = np.mean(coordination_numbers) if coordination_numbers else np.nan

            features['density'].append(density)
            features['volume_per_atom'].append(volume_per_atom)
            features['cation_to_anion_ratio'].append(cation_to_anion_ratio)
            features['mean_atomic_number'].append(mean_atomic_number)
            features['mean_bond_length'].append(mean_bond_length)
            features['mean_coordination'].append(mean_coordination)
        except:
            for key in features:
                features[key].append(np.nan)

    for key in features:
        df[key] = features[key]

    return df
