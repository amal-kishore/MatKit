"""Featurization module for MatKit."""

from matkit.featurizers.composition import (
    ElementalFeaturizer,
    OxidationStateFeaturizer,
    ChemicalComplexityFeaturizer,
    ThermodynamicFeaturizer
)

from matkit.featurizers.conversions import (
    BaseConverter,
    FormulaToComposition,
    StructureToComposition,
    StructureToImmutableStructure,
    StructureWithOxidationStates,
    CompositionWithOxidationStates,
    PymatgenFunctionWrapper
)

# Additional featurizers to be implemented
# from matkit.featurizers.structure import (...)
# from matkit.featurizers.electronic import (...)
# from matkit.featurizers.graph import (...)
# from matkit.featurizers.ensemble import (...)

__all__ = [
    # Composition featurizers
    "ElementalFeaturizer",
    "OxidationStateFeaturizer",
    "ChemicalComplexityFeaturizer",
    "ThermodynamicFeaturizer",
    
    # Conversion utilities
    "BaseConverter",
    "FormulaToComposition",
    "StructureToComposition", 
    "StructureToImmutableStructure",
    "StructureWithOxidationStates",
    "CompositionWithOxidationStates",
    "PymatgenFunctionWrapper"
]