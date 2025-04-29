# matkit/featurization/composition.py

import pandas as pd
from pymatgen.core import Composition, Element

def featurize_composition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composition-based features.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'formula_pretty' column.

    Returns:
    - pd.DataFrame: DataFrame with new composition features.
    """
    features = {
        'avg_atomic_number': [],
        'avg_atomic_mass': [],
        'avg_electronegativity': [],
        'avg_oxidation_state': [],
        'avg_valence_electrons': [],
        'fraction_metals': []
    }

    for formula in df['formula_pretty']:
        comp = Composition(formula)
        elements = comp.elements
        fractions = [comp.get_atomic_fraction(el) for el in elements]

        atomic_numbers = [el.Z for el in elements]
        atomic_masses = [el.atomic_mass for el in elements]
        electronegativities = [el.X if el.X else 0 for el in elements]
        oxidation_states = [sum(el.common_oxidation_states)/len(el.common_oxidation_states) if el.common_oxidation_states else 0 for el in elements]
        valence_electrons = []
        is_metal_flags = [el.is_metal for el in elements]

        for el in elements:
            try:
                full_elec_struct = el.full_electronic_structure
                max_n = max([n for (n, l, occ) in full_elec_struct])
                valence_e = sum(occ for (n, l, occ) in full_elec_struct if n == max_n)
                valence_electrons.append(valence_e)
            except:
                valence_electrons.append(0)

        features['avg_atomic_number'].append(sum(a * f for a, f in zip(atomic_numbers, fractions)))
        features['avg_atomic_mass'].append(sum(a * f for a, f in zip(atomic_masses, fractions)))
        features['avg_electronegativity'].append(sum(a * f for a, f in zip(electronegativities, fractions)))
        features['avg_oxidation_state'].append(sum(a * f for a, f in zip(oxidation_states, fractions)))
        features['avg_valence_electrons'].append(sum(a * f for a, f in zip(valence_electrons, fractions)))
        features['fraction_metals'].append(sum(f for f, is_metal in zip(fractions, is_metal_flags) if is_metal))

    for key in features:
        df[key] = features[key]

    return df
