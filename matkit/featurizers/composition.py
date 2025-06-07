"""Composition-based featurizers for MatKit."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from scipy import stats

from matkit.core.base import BaseFeaturizer, Citation
from matkit.core.exceptions import FeaturizationError
from matkit.utils.chemistry import get_element_data


class ElementalFeaturizer(BaseFeaturizer):
    """
    Generate features based on elemental properties.
    
    Features include statistics (mean, std, min, max, range) of various
    elemental properties across the composition.
    """
    
    def __init__(
        self,
        properties: Optional[List[str]] = None,
        stats: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize elemental featurizer.
        
        Parameters
        ----------
        properties : list of str, optional
            Element properties to use. If None, uses default set
        stats : list of str, optional
            Statistics to compute. If None, uses ["mean", "std", "min", "max", "range"]
        """
        super().__init__(**kwargs)
        
        self.properties = properties or [
            "atomic_number",
            "atomic_mass",
            "atomic_radius",
            "electronegativity",
            "ionization_energy",
            "electron_affinity",
            "melting_point",
            "boiling_point",
            "density",
            "specific_heat",
            "thermal_conductivity",
            "valence_electrons",
            "unfilled_orbitals",
            "metallic_radius",
            "covalent_radius",
            "van_der_waals_radius"
        ]
        
        self.stats = stats or ["mean", "std", "min", "max", "range"]
        self._element_data = get_element_data()
    
    def featurize(self, composition: Composition) -> np.ndarray:
        """Generate elemental features for a composition."""
        if isinstance(composition, str):
            composition = Composition(composition)
        
        features = []
        
        for prop in self.properties:
            # Get property values for all elements
            values = []
            for element, fraction in composition.get_el_amt_dict().items():
                el = Element(element)
                
                try:
                    # Get property value
                    if hasattr(el, prop):
                        val = getattr(el, prop)
                    else:
                        val = self._element_data.get(element, {}).get(prop, np.nan)
                    
                    if val is not None and not np.isnan(val):
                        values.extend([val] * int(fraction * 100))  # Weight by fraction
                
                except Exception:
                    continue
            
            # Compute statistics
            if values:
                values = np.array(values)
                
                for stat in self.stats:
                    if stat == "mean":
                        features.append(np.mean(values))
                    elif stat == "std":
                        features.append(np.std(values))
                    elif stat == "min":
                        features.append(np.min(values))
                    elif stat == "max":
                        features.append(np.max(values))
                    elif stat == "range":
                        features.append(np.max(values) - np.min(values))
                    elif stat == "mode":
                        mode_result = stats.mode(values)
                        features.append(mode_result.mode[0] if len(mode_result.mode) > 0 else np.nan)
                    else:
                        features.append(np.nan)
            else:
                features.extend([np.nan] * len(self.stats))
        
        return np.array(features)
    
    def feature_labels(self) -> List[str]:
        """Get feature labels."""
        labels = []
        for prop in self.properties:
            for stat in self.stats:
                labels.append(f"{prop}_{stat}")
        return labels
    
    def citations(self) -> List[Citation]:
        """Get citations."""
        return [
            Citation(
                authors=["Ward, L.", "Agrawal, A.", "Choudhary, A.", "Wolverton, C."],
                title="A general-purpose machine learning framework for predicting properties of inorganic materials",
                journal="npj Computational Materials",
                year=2016,
                doi="10.1038/npjcompumats.2016.28"
            )
        ]


class OxidationStateFeaturizer(BaseFeaturizer):
    """
    Features based on oxidation states of elements.
    
    Includes statistics of oxidation states and their differences.
    """
    
    def __init__(self, stats: Optional[List[str]] = None, **kwargs):
        """Initialize oxidation state featurizer."""
        super().__init__(**kwargs)
        self.stats = stats or ["mean", "std", "min", "max", "range"]
    
    def featurize(self, composition: Composition) -> np.ndarray:
        """Generate oxidation state features."""
        if isinstance(composition, str):
            composition = Composition(composition)
        
        features = []
        
        try:
            # Get oxidation states
            oxi_states = composition.oxi_state_guesses(max_sites=50)
            
            if oxi_states:
                # Use first guess
                oxi_comp = oxi_states[0]
                
                # Get oxidation state values
                oxi_values = []
                for el, amt in oxi_comp.items():
                    oxi_values.extend([el.oxi_state] * int(amt * 100))
                
                oxi_values = np.array(oxi_values)
                
                # Compute statistics
                for stat in self.stats:
                    if stat == "mean":
                        features.append(np.mean(oxi_values))
                    elif stat == "std":
                        features.append(np.std(oxi_values))
                    elif stat == "min":
                        features.append(np.min(oxi_values))
                    elif stat == "max":
                        features.append(np.max(oxi_values))
                    elif stat == "range":
                        features.append(np.max(oxi_values) - np.min(oxi_values))
                    else:
                        features.append(np.nan)
                
                # Additional features
                features.append(len(set(oxi_values)))  # Number of unique oxidation states
                features.append(np.sum(np.abs(oxi_values)))  # Sum of absolute oxidation states
                
            else:
                # No oxidation states found
                features = [np.nan] * (len(self.stats) + 2)
                
        except Exception:
            features = [np.nan] * (len(self.stats) + 2)
        
        return np.array(features)
    
    def feature_labels(self) -> List[str]:
        """Get feature labels."""
        labels = [f"oxidation_state_{stat}" for stat in self.stats]
        labels.extend(["n_unique_oxidation_states", "sum_abs_oxidation_states"])
        return labels
    
    def citations(self) -> List[Citation]:
        """Get citations."""
        return [
            Citation(
                authors=["Jain, A.", "Ong, S.P.", "Hautier, G.", "Chen, W.", "Richards, W.D.", "Dacek, S.", "Cholia, S.", "Gunter, D.", "Skinner, D.", "Ceder, G.", "Persson, K.A."],
                title="The Materials Project: A materials genome approach to accelerating materials innovation",
                journal="APL Materials",
                year=2013,
                doi="10.1063/1.4812323"
            )
        ]


class ChemicalComplexityFeaturizer(BaseFeaturizer):
    """
    Features describing chemical complexity of compositions.
    
    Includes entropy, number of elements, complexity metrics.
    """
    
    def featurize(self, composition: Composition) -> np.ndarray:
        """Generate chemical complexity features."""
        if isinstance(composition, str):
            composition = Composition(composition)
        
        features = []
        
        # Number of elements
        n_elements = len(composition.elements)
        features.append(n_elements)
        
        # Get atomic fractions
        fractions = list(composition.get_atomic_fraction_dict().values())
        
        # Shannon entropy
        entropy = -np.sum([f * np.log(f) for f in fractions if f > 0])
        features.append(entropy)
        
        # Normalized entropy
        max_entropy = np.log(n_elements) if n_elements > 1 else 0
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        features.append(norm_entropy)
        
        # Effective number of species (exp of entropy)
        eff_species = np.exp(entropy)
        features.append(eff_species)
        
        # Gini coefficient (inequality measure)
        sorted_fractions = sorted(fractions)
        n = len(sorted_fractions)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_fractions)) / (n * np.sum(sorted_fractions)) - (n + 1) / n
        features.append(gini)
        
        # Maximum and minimum fractions
        features.append(max(fractions))
        features.append(min(fractions))
        
        # Fraction of most abundant element
        features.append(max(fractions))
        
        # Number of minority elements (< 10%)
        n_minority = sum(1 for f in fractions if f < 0.1)
        features.append(n_minority)
        
        # Check if alloy-like (all fractions between 0.2 and 0.5)
        is_alloy_like = all(0.2 <= f <= 0.5 for f in fractions)
        features.append(float(is_alloy_like))
        
        return np.array(features)
    
    def feature_labels(self) -> List[str]:
        """Get feature labels."""
        return [
            "n_elements",
            "shannon_entropy",
            "normalized_entropy",
            "effective_n_species",
            "gini_coefficient",
            "max_atomic_fraction",
            "min_atomic_fraction",
            "majority_fraction",
            "n_minority_elements",
            "is_alloy_like"
        ]
    
    def citations(self) -> List[Citation]:
        """Get citations."""
        return [
            Citation(
                authors=["Yang, Y.", "Zhang, Y."],
                title="Prediction of high-entropy stabilized solid-solution in multi-component alloys",
                journal="Materials Chemistry and Physics",
                year=2012,
                doi="10.1016/j.matchemphys.2011.11.021"
            )
        ]


class ThermodynamicFeaturizer(BaseFeaturizer):
    """
    Thermodynamic features based on mixing theory.
    
    Includes mixing enthalpy, entropy, and related quantities.
    """
    
    def __init__(self, temperature: float = 298.15, **kwargs):
        """
        Initialize thermodynamic featurizer.
        
        Parameters
        ----------
        temperature : float, default=298.15
            Temperature in Kelvin for calculations
        """
        super().__init__(**kwargs)
        self.temperature = temperature
        self.R = 8.314  # Gas constant in J/(mol·K)
    
    def featurize(self, composition: Composition) -> np.ndarray:
        """Generate thermodynamic features."""
        if isinstance(composition, str):
            composition = Composition(composition)
        
        features = []
        
        # Get molar fractions
        fractions = list(composition.get_atomic_fraction_dict().values())
        n_elements = len(fractions)
        
        # Mixing entropy (configurational)
        mixing_entropy = -self.R * np.sum([f * np.log(f) for f in fractions if f > 0])
        features.append(mixing_entropy)
        
        # Maximum mixing entropy (for equiatomic)
        max_mixing_entropy = self.R * np.log(n_elements) if n_elements > 1 else 0
        features.append(max_mixing_entropy)
        
        # Normalized mixing entropy
        norm_mixing_entropy = mixing_entropy / max_mixing_entropy if max_mixing_entropy > 0 else 0
        features.append(norm_mixing_entropy)
        
        # Free energy contribution from mixing entropy
        free_energy_mix = -self.temperature * mixing_entropy
        features.append(free_energy_mix)
        
        # Mismatch parameters
        elements = list(composition.get_el_amt_dict().keys())
        
        if n_elements > 1:
            # Atomic size mismatch
            radii = []
            for el in elements:
                try:
                    radii.append(Element(el).atomic_radius or Element(el).atomic_radius_calculated)
                except:
                    radii.append(np.nan)
            
            if not any(np.isnan(radii)):
                mean_radius = np.mean(radii)
                size_mismatch = np.sqrt(np.sum([(r - mean_radius)**2 for r in radii]) / n_elements) / mean_radius
                features.append(size_mismatch)
            else:
                features.append(np.nan)
            
            # Electronegativity mismatch
            electroneg = []
            for el in elements:
                try:
                    electroneg.append(Element(el).X)
                except:
                    electroneg.append(np.nan)
            
            if not any(np.isnan(electroneg)):
                electroneg_mismatch = np.std(electroneg)
                features.append(electroneg_mismatch)
            else:
                features.append(np.nan)
        else:
            features.extend([0.0, 0.0])  # No mismatch for single element
        
        # Valence electron concentration (VEC)
        vec_values = []
        for el, frac in composition.get_el_amt_dict().items():
            try:
                n_valence = Element(el).group
                vec_values.append(n_valence * frac)
            except:
                pass
        
        if vec_values:
            vec = sum(vec_values) / sum(composition.get_el_amt_dict().values())
            features.append(vec)
        else:
            features.append(np.nan)
        
        return np.array(features)
    
    def feature_labels(self) -> List[str]:
        """Get feature labels."""
        return [
            "mixing_entropy",
            "max_mixing_entropy",
            "normalized_mixing_entropy",
            "free_energy_mixing",
            "atomic_size_mismatch",
            "electronegativity_mismatch",
            "valence_electron_concentration"
        ]
    
    def citations(self) -> List[Citation]:
        """Get citations."""
        return [
            Citation(
                authors=["Zhang, Y.", "Zuo, T.T.", "Tang, Z.", "Gao, M.C.", "Dahmen, K.A.", "Liaw, P.K.", "Lu, Z.P."],
                title="Microstructures and properties of high-entropy alloys",
                journal="Progress in Materials Science",
                year=2014,
                doi="10.1016/j.pmatsci.2013.10.001"
            )
        ]