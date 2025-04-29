# matkit/data_fetching/fetch_data.py

import pandas as pd
from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def fetch_materials_data(api_key: str, formula: str = "ABC") -> pd.DataFrame:
    """
    Fetch materials data from the Materials Project.

    Parameters:
    - api_key (str): Your Materials Project API key.
    - formula (str): Chemical formula to search for.

    Returns:
    - pd.DataFrame: DataFrame containing materials data.
    """
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            formula=formula,
            is_stable=True,
            fields=["material_id", "formula_pretty", "structure", "band_gap", "is_gap_direct"]
        )

    data = []
    for doc in docs:
        try:
            structure = doc.structure
            sga = SpacegroupAnalyzer(structure)
            crystal_system = sga.get_crystal_system()

            data.append({
                "material_id": doc.material_id,
                "formula_pretty": doc.formula_pretty,
                "structure": structure,
                "band_gap": doc.band_gap,
                "crystal_system": crystal_system,
                "is_gap_direct": doc.is_gap_direct
            })
        except Exception as e:
            print(f"Error processing {doc.formula_pretty}: {e}")

    return pd.DataFrame(data)
