"""Chemistry utility functions."""

from typing import Dict, Any, Optional
import json
from pathlib import Path

from pymatgen.core import Composition, Element
import numpy as np

from matkit.core.logging import get_logger

logger = get_logger(__name__)


def get_element_data() -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive element property data.
    
    Returns
    -------
    dict
        Element properties indexed by symbol
    """
    # Build element data from pymatgen
    element_data = {}
    
    for z in range(1, 119):  # Elements 1-118
        try:
            elem = Element.from_Z(z)
            
            element_data[elem.symbol] = {
                "atomic_number": elem.Z,
                "atomic_mass": elem.atomic_mass,
                "atomic_radius": elem.atomic_radius or elem.atomic_radius_calculated,
                "electronegativity": elem.X,
                "ionization_energy": elem.ionization_energy,
                "electron_affinity": elem.electron_affinity,
                "melting_point": elem.melting_point,
                "boiling_point": elem.boiling_point,
                "density": elem.density_of_solid,
                "specific_heat": elem.specific_heat_capacity,
                "thermal_conductivity": elem.thermal_conductivity,
                "valence_electrons": elem.full_electronic_structure[-1][2] if elem.full_electronic_structure else 0,
                "unfilled_orbitals": _count_unfilled_orbitals(elem),
                "metallic_radius": elem.metallic_radius,
                "covalent_radius": elem.covalent_radius,
                "van_der_waals_radius": elem.van_der_waals_radius,
                "mendeleev_no": elem.mendeleev_no,
                "electrical_resistivity": elem.electrical_resistivity,
                "bulk_modulus": elem.bulk_modulus,
                "young_modulus": elem.youngs_modulus,
                "brinell_hardness": elem.brinell_hardness,
                "poissons_ratio": elem.poissons_ratio,
                "mineral_hardness": elem.mineral_hardness,
                "vickers_hardness": elem.vickers_hardness,
                "fusion_enthalpy": elem.fusion_enthalpy,
                "evaporation_heat": elem.evaporation_heat,
                "molar_volume": elem.molar_volume,
                "group": elem.group,
                "period": elem.row,
                "is_metal": elem.is_metal,
                "is_metalloid": elem.is_metalloid,
                "is_alkali": elem.is_alkali,
                "is_alkaline": elem.is_alkaline,
                "is_halogen": elem.is_halogen,
                "is_chalcogen": elem.is_chalcogen,
                "is_lanthanoid": elem.is_lanthanoid,
                "is_actinoid": elem.is_actinoid,
                "is_transition_metal": elem.is_transition_metal,
                "is_post_transition_metal": elem.is_post_transition_metal,
                "is_noble_gas": elem.is_noble_gas,
            }
            
            # Convert None to np.nan for numerical properties
            for key, value in element_data[elem.symbol].items():
                if value is None and key not in ["is_metal", "is_metalloid", "is_alkali", 
                                                  "is_alkaline", "is_halogen", "is_chalcogen",
                                                  "is_lanthanoid", "is_actinoid", "is_transition_metal",
                                                  "is_post_transition_metal", "is_noble_gas"]:
                    element_data[elem.symbol][key] = np.nan
                    
        except Exception as e:
            logger.warning(f"Failed to get data for element Z={z}: {e}")
    
    return element_data


def _count_unfilled_orbitals(element: Element) -> int:
    """Count number of unfilled orbitals for an element."""
    try:
        # Get electron configuration
        config = element.full_electronic_structure
        if not config:
            return 0
        
        # Maximum electrons per orbital type
        max_electrons = {"s": 2, "p": 6, "d": 10, "f": 14}
        
        unfilled = 0
        for n, l, e in config:
            orbital_type = ["s", "p", "d", "f"][l]
            if e < max_electrons[orbital_type]:
                unfilled += 1
                
        return unfilled
        
    except Exception:
        return 0


def validate_composition(comp_str: str) -> Optional[Composition]:
    """
    Validate and parse a composition string.
    
    Parameters
    ----------
    comp_str : str
        Composition string (e.g., "Fe2O3")
        
    Returns
    -------
    Composition or None
        Parsed composition or None if invalid
    """
    try:
        comp = Composition(comp_str)
        
        # Additional validation
        if len(comp.elements) == 0:
            logger.warning(f"Empty composition: {comp_str}")
            return None
            
        if any(el.Z > 118 for el in comp.elements):
            logger.warning(f"Invalid element in composition: {comp_str}")
            return None
            
        return comp
        
    except Exception as e:
        logger.warning(f"Failed to parse composition '{comp_str}': {e}")
        return None


def get_stoichiometry_pattern(composition: Composition) -> Dict[str, int]:
    """
    Extract stoichiometry pattern from composition.
    
    Parameters
    ----------
    composition : Composition
        Pymatgen Composition object
        
    Returns
    -------
    dict
        Stoichiometry pattern with element counts
    """
    # Get reduced composition
    reduced = composition.reduced_composition
    
    # Get integer formula
    int_formula = reduced.get_integer_formula_and_factor()[0]
    
    # Parse to get pattern
    pattern = {}
    for el, amt in Composition(int_formula).as_dict().items():
        pattern[el] = int(amt)
    
    return pattern


def check_stoichiometry(composition: Composition, pattern: str) -> bool:
    """
    Check if composition matches a stoichiometry pattern.
    
    Parameters
    ----------
    composition : Composition
        Composition to check
    pattern : str
        Pattern like "ABN2" where A,B are variable elements and N is fixed
        
    Returns
    -------
    bool
        Whether composition matches pattern
    """
    # For ABN2 pattern specifically
    if pattern == "ABN2":
        reduced_comp = composition.reduced_composition
        el_dict = reduced_comp.get_el_amt_dict()
        
        # Must have exactly 3 elements
        if len(el_dict) != 3:
            return False
        
        # Must have nitrogen with amount 2
        if "N" not in el_dict or abs(el_dict["N"] - 2) > 0.01:
            return False
        
        # Other two elements must each have amount 1
        other_elements = {el: amt for el, amt in el_dict.items() if el != "N"}
        if len(other_elements) != 2:
            return False
        
        for el, amt in other_elements.items():
            if abs(amt - 1) > 0.01:
                return False
        
        return True
    
    # For other patterns, use the general approach
    # Parse pattern to identify fixed elements
    fixed_elements = {}
    variable_positions = []
    
    i = 0
    while i < len(pattern):
        if pattern[i].isupper():
            # Start of element
            elem = pattern[i]
            i += 1
            
            # Check for lowercase (two-letter element)
            if i < len(pattern) and pattern[i].islower():
                elem += pattern[i]
                i += 1
            
            # Check for number
            count = ""
            while i < len(pattern) and pattern[i].isdigit():
                count += pattern[i]
                i += 1
            
            count = int(count) if count else 1
            
            # Check if this is a real element or placeholder
            try:
                Element(elem)
                fixed_elements[elem] = count
            except:
                # It's a placeholder like A, B
                variable_positions.append((elem, count))
        else:
            i += 1
    
    # Get composition stoichiometry
    comp_dict = composition.get_el_amt_dict()
    reduced_comp = composition.reduced_composition
    
    # Check fixed elements
    for elem, expected_count in fixed_elements.items():
        if elem not in comp_dict:
            return False
        
        actual_count = reduced_comp[elem]
        if abs(actual_count - expected_count) > 0.01:
            return False
    
    # Check total stoichiometry matches
    expected_total = sum(fixed_elements.values()) + sum(c for _, c in variable_positions)
    actual_total = sum(reduced_comp.get_el_amt_dict().values())
    
    if abs(actual_total - expected_total) > 0.01:
        return False
    
    # Check that remaining elements match variable positions
    remaining_elements = {el: amt for el, amt in reduced_comp.get_el_amt_dict().items() 
                         if el not in fixed_elements}
    
    if len(remaining_elements) != len(variable_positions):
        return False
    
    # Check counts match
    remaining_counts = sorted(remaining_elements.values(), reverse=True)
    expected_counts = sorted([c for _, c in variable_positions], reverse=True)
    
    for actual, expected in zip(remaining_counts, expected_counts):
        if abs(actual - expected) > 0.01:
            return False
    
    return True