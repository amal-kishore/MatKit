# MatKit 🧪

A modern, robust materials informatics toolkit for advanced materials science research and machine learning applications.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/badge/pypi-v0.2.0-orange.svg)](https://pypi.org/project/matkit/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🚀 Features

MatKit v0.2.0 provides a comprehensive suite of tools for materials science workflows:

### 🔬 **Featurization**
- **Composition Featurizers**: Element properties, oxidation states, chemical complexity, thermodynamics
- **Conversion Utilities**: Seamless conversion between formulas, compositions, and structures
- **Extensible Architecture**: Easy integration of custom featurizers

### 📊 **Data Handling**
- **Materials Project Integration**: Async fetching with 69+ material properties
- **Intelligent Caching**: TTL-based caching with size limits for optimal performance  
- **Robust Validation**: Comprehensive data validation and error handling

### ⚙️ **Core Infrastructure**
- **Modern Python**: Type hints, async/await, Pydantic models
- **Performance Optimized**: Parallel processing and efficient algorithms
- **Production Ready**: Comprehensive logging, error handling, and testing

## 📦 Installation

### From Source
```bash
git clone https://github.com/amal-kishore/MatKit.git
cd MatKit
pip install -e .
```

### Requirements
- Python 3.9+
- pymatgen
- mp-api
- pandas
- numpy
- pydantic
- loguru
- asyncio

## 🔧 Quick Start

### Basic Usage

```python
from matkit.featurizers import ElementalFeaturizer, FormulaToComposition
from matkit.data.fetchers import MaterialsProjectFetcher

# Convert formula to composition
converter = FormulaToComposition()
composition = converter.convert("CaTiO3")

# Generate elemental features
featurizer = ElementalFeaturizer()
features = featurizer.featurize(composition)
print(f"Features shape: {features.shape}")

# Fetch materials from Materials Project
fetcher = MaterialsProjectFetcher(api_key="your_api_key")
materials = await fetcher.fetch_by_formula("**N2")  # ABN2 materials
```

### Advanced Featurization

```python
from matkit.featurizers import (
    OxidationStateFeaturizer,
    ChemicalComplexityFeaturizer, 
    ThermodynamicFeaturizer,
    CompositionWithOxidationStates
)

# Add oxidation states
oxi_converter = CompositionWithOxidationStates()
oxi_comp = oxi_converter.convert(composition)

# Multiple featurizers
featurizers = [
    ElementalFeaturizer(properties=["atomic_number", "electronegativity"]),
    OxidationStateFeaturizer(),
    ChemicalComplexityFeaturizer(),
    ThermodynamicFeaturizer(temperature=1000)
]

# Combine features
all_features = []
for featurizer in featurizers:
    features = featurizer.featurize(composition)
    all_features.extend(features)

print(f"Total features: {len(all_features)}")
```

### Data Fetching and Analysis

```python
import asyncio
from matkit.data.fetchers import MaterialsProjectFetcher
from matkit.utils.chemistry import check_stoichiometry

async def analyze_materials():
    fetcher = MaterialsProjectFetcher(
        api_key="your_api_key",
        chunk_size=100
    )
    
    # Fetch ABO3 perovskites
    properties = [
        "material_id", "formula_pretty", "structure",
        "band_gap", "formation_energy_per_atom", "density"
    ]
    
    materials = await fetcher.fetch_by_formula(
        formula="**O3", 
        properties=properties
    )
    
    # Filter for ABO3 stoichiometry
    perovskites = []
    for material in materials:
        comp = material.structure.composition
        if check_stoichiometry(comp, "ABO3"):
            perovskites.append(material)
    
    print(f"Found {len(perovskites)} ABO3 perovskites")
    return perovskites

# Run analysis
materials = asyncio.run(analyze_materials())
```

## 📚 API Reference

### Composition Featurizers

| Featurizer | Description | Output Features |
|------------|-------------|-----------------|
| `ElementalFeaturizer` | Statistical properties of constituent elements | 80 features |
| `OxidationStateFeaturizer` | Oxidation state statistics and relationships | 7 features |
| `ChemicalComplexityFeaturizer` | Entropy, diversity, and complexity metrics | 10 features |
| `ThermodynamicFeaturizer` | Mixing properties and thermodynamic indicators | 7 features |

### Conversion Utilities

| Converter | Input → Output | Description |
|-----------|----------------|-------------|
| `FormulaToComposition` | `str` → `Composition` | Parse chemical formulas |
| `StructureToComposition` | `Structure` → `Composition` | Extract composition from structure |
| `StructureWithOxidationStates` | `Structure` → `Structure` | Add oxidation states to structures |
| `CompositionWithOxidationStates` | `Composition` → `Composition` | Add oxidation states to compositions |
| `StructureToImmutableStructure` | `Structure` → `IStructure` | Create hashable structures |

### Materials Project Integration

```python
# Comprehensive property fetching
fetcher = MaterialsProjectFetcher(api_key="your_key")

# Available properties (69 total)
properties = [
    # Basic
    "material_id", "formula_pretty", "structure", "density",
    # Electronic  
    "band_gap", "cbm", "vbm", "is_metal",
    # Mechanical
    "bulk_modulus", "shear_modulus", "universal_anisotropy",
    # Magnetic
    "is_magnetic", "total_magnetization",
    # Thermodynamic
    "formation_energy_per_atom", "energy_above_hull", "is_stable"
]

materials = await fetcher.fetch_by_formula("LiCoO2", properties=properties)
```

## 🧪 Validation and Testing

MatKit has been extensively tested and validated:

- ✅ **482 ABN2 materials** correctly identified and processed
- ✅ **2,547 ABO3 materials** successfully validated  
- ✅ **100% success rate** on conversion featurizers
- ✅ **69 Materials Project fields** integration verified

## 🏗️ Architecture

```
MatKit/
├── core/           # Base classes, exceptions, configuration
├── data/           # Fetchers, validators, caching
├── featurizers/    # Composition and conversion featurizers
└── utils/          # Chemistry utilities and helpers
```

### Design Principles

- **Modular**: Easy to extend and customize
- **Async-Ready**: Built for high-performance workflows
- **Type-Safe**: Comprehensive type hints and validation
- **Well-Documented**: Extensive docstrings and examples

## 📈 Performance

- **Parallel Processing**: Multi-core featurization support
- **Intelligent Caching**: Reduce API calls and computation time
- **Memory Efficient**: Optimized data structures and algorithms
- **Async Operations**: Non-blocking I/O for data fetching

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/amal-kishore/MatKit.git
cd MatKit
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Pymatgen](https://pymatgen.org/) for materials analysis infrastructure
- [Materials Project](https://materialsproject.org/) for comprehensive materials database
- Materials science community for inspiration and feedback

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/amal-kishore/MatKit/issues)
- **Email**: amalk4905@gmail.com
- **Documentation**: [Wiki](https://github.com/amal-kishore/MatKit/wiki)

---

**MatKit** - Empowering materials discovery through intelligent featurization and data analysis.