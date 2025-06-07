"""Conversion featurizers for MatKit data type transformations."""

from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

import numpy as np
from pymatgen.core import Composition, Structure, IStructure
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core.periodic_table import Element

from matkit.core.base import BaseFeaturizer, Citation
from matkit.core.exceptions import FeaturizationError


class BaseConverter(BaseFeaturizer, ABC):
    """
    Abstract base class for data type conversion featurizers.
    
    Converters transform input data from one format to another
    (e.g., string to composition, structure to composition).
    """
    
    def __init__(self, **kwargs):
        """Initialize base converter."""
        super().__init__(**kwargs)
    
    @abstractmethod
    def convert(self, data: Any) -> Any:
        """
        Convert input data to target format.
        
        Parameters
        ----------
        data : Any
            Input data to convert
            
        Returns
        -------
        Any
            Converted data
        """
        pass
    
    def featurize(self, data: Any) -> np.ndarray:
        """
        Convert data and return as numpy array if applicable.
        
        For most converters, this just calls convert() and wraps
        in array if needed.
        """
        result = self.convert(data)
        if isinstance(result, (int, float)):
            return np.array([result])
        elif isinstance(result, (list, tuple)):
            return np.array(result)
        else:
            return result
    
    def feature_labels(self) -> List[str]:
        """Get feature labels for converted data."""
        return ["converted_data"]


class FormulaToComposition(BaseConverter):
    """
    Convert chemical formula strings to pymatgen Composition objects.
    
    Handles various formula formats including:
    - Simple formulas: "NaCl", "H2O" 
    - Complex formulas: "Ca(OH)2", "Al2(SO4)3"
    - Fractional compositions: "Li0.5CoO2"
    """
    
    def __init__(self, normalize: bool = True, **kwargs):
        """
        Initialize formula to composition converter.
        
        Parameters
        ----------
        normalize : bool, default=True
            Whether to normalize composition to smallest integers
        """
        super().__init__(**kwargs)
        self.normalize = normalize
    
    def convert(self, formula: str) -> Composition:
        """
        Convert formula string to Composition.
        
        Parameters
        ----------
        formula : str
            Chemical formula string
            
        Returns
        -------
        Composition
            Pymatgen Composition object
        """
        try:
            comp = Composition(formula)
            if self.normalize:
                comp = comp.reduced_composition
            return comp
        
        except Exception as e:
            raise FeaturizationError(f"Failed to parse formula '{formula}': {str(e)}")
    
    def feature_labels(self) -> List[str]:
        """Get feature labels."""
        return ["composition"]
    
    def citations(self) -> List[Citation]:
        """Get citations."""
        return [
            Citation(
                authors=["Ong, S.P.", "Richards, W.D.", "Jain, A.", "Hautier, G.", "Kocher, M.", "Cholia, S.", "Gunter, D.", "Chevrier, V.L.", "Persson, K.A.", "Ceder, G."],
                title="Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis",
                journal="Computational Materials Science",
                year=2013,
                doi="10.1016/j.commatsci.2012.10.028"
            )
        ]


class StructureToComposition(BaseConverter):
    """
    Extract composition from pymatgen Structure objects.
    
    Converts Structure objects to their corresponding Composition,
    optionally normalizing and handling symmetry-equivalent sites.
    """
    
    def __init__(self, 
                 normalize: bool = True,
                 remove_duplicates: bool = True,
                 **kwargs):
        """
        Initialize structure to composition converter.
        
        Parameters
        ----------
        normalize : bool, default=True
            Whether to normalize composition to smallest integers
        remove_duplicates : bool, default=True
            Whether to remove symmetrically equivalent sites before extraction
        """
        super().__init__(**kwargs)
        self.normalize = normalize
        self.remove_duplicates = remove_duplicates
    
    def convert(self, structure: Structure) -> Composition:
        """
        Extract composition from structure.
        
        Parameters
        ----------
        structure : Structure
            Pymatgen Structure object
            
        Returns
        -------
        Composition
            Extracted composition
        """
        try:
            if self.remove_duplicates:
                # Get primitive structure to remove duplicates
                prim_struct = structure.get_primitive_structure()
                comp = prim_struct.composition
            else:
                comp = structure.composition
            
            if self.normalize:
                comp = comp.reduced_composition
                
            return comp
            
        except Exception as e:
            raise FeaturizationError(f"Failed to extract composition from structure: {str(e)}")
    
    def feature_labels(self) -> List[str]:
        """Get feature labels."""
        return ["composition"]
    
    def citations(self) -> List[Citation]:
        """Get citations."""
        return [
            Citation(
                authors=["Ong, S.P.", "Richards, W.D.", "Jain, A.", "Hautier, G.", "Kocher, M.", "Cholia, S.", "Gunter, D.", "Chevrier, V.L.", "Persson, K.A.", "Ceder, G."],
                title="Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis",
                journal="Computational Materials Science",
                year=2013,
                doi="10.1016/j.commatsci.2012.10.028"
            )
        ]


class StructureToImmutableStructure(BaseConverter):
    """
    Convert mutable Structure to immutable IStructure.
    
    IStructure objects are hashable and can be used as dictionary keys
    or in sets, useful for deduplication and caching.
    """
    
    def convert(self, structure: Structure) -> IStructure:
        """
        Convert Structure to IStructure.
        
        Parameters
        ----------
        structure : Structure
            Mutable pymatgen Structure
            
        Returns
        -------
        IStructure
            Immutable pymatgen IStructure
        """
        try:
            return IStructure.from_sites(structure.sites, 
                                       charge=structure.charge)
        except Exception as e:
            raise FeaturizationError(f"Failed to convert to IStructure: {str(e)}")
    
    def feature_labels(self) -> List[str]:
        """Get feature labels."""
        return ["immutable_structure"]
    
    def citations(self) -> List[Citation]:
        """Get citations."""
        return [
            Citation(
                authors=["Ong, S.P.", "Richards, W.D.", "Jain, A.", "Hautier, G.", "Kocher, M.", "Cholia, S.", "Gunter, D.", "Chevrier, V.L.", "Persson, K.A.", "Ceder, G."],
                title="Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis",
                journal="Computational Materials Science",
                year=2013,
                doi="10.1016/j.commatsci.2012.10.028"
            )
        ]


class StructureWithOxidationStates(BaseConverter):
    """
    Add oxidation states to pymatgen Structure objects.
    
    Uses bond valence analysis or other methods to assign
    oxidation states to all sites in the structure.
    """
    
    def __init__(self, 
                 method: str = "bv_analyzer",
                 max_sites: int = 200,
                 **kwargs):
        """
        Initialize structure oxidation state converter.
        
        Parameters
        ----------
        method : str, default="bv_analyzer"
            Method for assigning oxidation states
            Options: "bv_analyzer", "composition_guess"
        max_sites : int, default=200
            Maximum number of sites to process
        """
        super().__init__(**kwargs)
        self.method = method
        self.max_sites = max_sites
        self.bv_analyzer = BVAnalyzer()
    
    def convert(self, structure: Structure) -> Structure:
        """
        Add oxidation states to structure.
        
        Parameters
        ----------
        structure : Structure
            Input structure without oxidation states
            
        Returns
        -------
        Structure
            Structure with oxidation states assigned
        """
        try:
            if len(structure) > self.max_sites:
                raise FeaturizationError(f"Structure too large ({len(structure)} > {self.max_sites} sites)")
            
            if self.method == "bv_analyzer":
                # Use bond valence analyzer
                oxi_struct = self.bv_analyzer.get_oxi_state_decorated_structure(structure)
                
            elif self.method == "composition_guess":
                # Use composition-based guessing
                comp = structure.composition
                oxi_guesses = comp.oxi_state_guesses(max_sites=self.max_sites)
                
                if not oxi_guesses:
                    raise FeaturizationError("No valid oxidation state assignment found")
                
                # Use first guess
                oxi_comp = oxi_guesses[0]
                
                # Create new structure with oxidation states
                new_sites = []
                for site in structure:
                    element = site.specie
                    for oxi_el in oxi_comp.elements:
                        if oxi_el.symbol == element.symbol:
                            new_site = site.copy()
                            new_site.species = {oxi_el: 1.0}
                            new_sites.append(new_site)
                            break
                
                oxi_struct = Structure.from_sites(new_sites, charge=structure.charge)
            
            else:
                raise FeaturizationError(f"Unknown oxidation state method: {self.method}")
            
            return oxi_struct
            
        except Exception as e:
            raise FeaturizationError(f"Failed to add oxidation states: {str(e)}")
    
    def feature_labels(self) -> List[str]:
        """Get feature labels."""
        return ["oxidized_structure"]
    
    def citations(self) -> List[Citation]:
        """Get citations."""
        return [
            Citation(
                authors=["Adams, S."],
                title="Practical considerations in determining bond valence parameters",
                journal="Acta Crystallographica Section B",
                year=2001,
                doi="10.1107/S0108768101003068"
            )
        ]


class CompositionWithOxidationStates(BaseConverter):
    """
    Add oxidation states to pymatgen Composition objects.
    
    Uses various algorithms to guess most likely oxidation
    states for elements in the composition.
    """
    
    def __init__(self, 
                 max_sites: int = 200,
                 use_first_guess: bool = True,
                 **kwargs):
        """
        Initialize composition oxidation state converter.
        
        Parameters
        ----------
        max_sites : int, default=200
            Maximum number of sites for oxidation state calculation
        use_first_guess : bool, default=True
            Whether to use first oxidation state guess or try multiple
        """
        super().__init__(**kwargs)
        self.max_sites = max_sites
        self.use_first_guess = use_first_guess
    
    def convert(self, composition: Composition) -> Composition:
        """
        Add oxidation states to composition.
        
        Parameters
        ----------
        composition : Composition
            Input composition without oxidation states
            
        Returns
        -------
        Composition
            Composition with oxidation states assigned
        """
        try:
            # Get oxidation state guesses - returns tuple of dicts
            oxi_guesses = composition.oxi_state_guesses(max_sites=self.max_sites)
            
            if not oxi_guesses:
                raise FeaturizationError("No valid oxidation state assignment found")
            
            # Get first guess dict and convert back to composition
            oxi_dict = oxi_guesses[0] if self.use_first_guess else oxi_guesses[0]
            
            # Create composition with oxidation states from original amounts
            # but elements with oxidation states
            from pymatgen.core.periodic_table import Species
            
            # Get original element amounts
            el_amounts = composition.get_el_amt_dict()
            
            # Create new composition with oxidation state species
            new_composition_dict = {}
            for element, amount in el_amounts.items():
                if element in oxi_dict:
                    # Create species with oxidation state
                    species = Species(element, oxi_dict[element])
                    new_composition_dict[species] = amount
                else:
                    # Keep original element
                    new_composition_dict[element] = amount
            
            return Composition(new_composition_dict)
                
        except Exception as e:
            raise FeaturizationError(f"Failed to add oxidation states to composition: {str(e)}")
    
    def feature_labels(self) -> List[str]:
        """Get feature labels."""
        return ["oxidized_composition"]
    
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


class PymatgenFunctionWrapper(BaseConverter):
    """
    Generic wrapper to apply any pymatgen function to objects.
    
    Allows applying arbitrary pymatgen functions as featurizers,
    useful for custom transformations and analysis.
    """
    
    def __init__(self, 
                 function,
                 function_args: Optional[tuple] = None,
                 function_kwargs: Optional[Dict] = None,
                 **kwargs):
        """
        Initialize function wrapper.
        
        Parameters
        ----------
        function : callable
            Function to apply to input data
        function_args : tuple, optional
            Additional positional arguments for function
        function_kwargs : dict, optional
            Additional keyword arguments for function
        """
        super().__init__(**kwargs)
        self.function = function
        self.function_args = function_args or ()
        self.function_kwargs = function_kwargs or {}
    
    def convert(self, data: Any) -> Any:
        """
        Apply function to input data.
        
        Parameters
        ----------
        data : Any
            Input data for function
            
        Returns
        -------
        Any
            Function output
        """
        try:
            return self.function(data, *self.function_args, **self.function_kwargs)
        except Exception as e:
            raise FeaturizationError(f"Function application failed: {str(e)}")
    
    def feature_labels(self) -> List[str]:
        """Get feature labels."""
        func_name = getattr(self.function, '__name__', 'function')
        return [f"{func_name}_output"]
    
    def citations(self) -> List[Citation]:
        """Get citations."""
        return [
            Citation(
                authors=["Ong, S.P.", "Richards, W.D.", "Jain, A.", "Hautier, G.", "Kocher, M.", "Cholia, S.", "Gunter, D.", "Chevrier, V.L.", "Persson, K.A.", "Ceder, G."],
                title="Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis",
                journal="Computational Materials Science",
                year=2013,
                doi="10.1016/j.commatsci.2012.10.028"
            )
        ]