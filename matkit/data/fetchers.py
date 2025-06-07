"""Data fetchers for various materials databases."""

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure, Composition
from tenacity import retry, stop_after_attempt, wait_exponential

from matkit.core.config import get_config
from matkit.core.exceptions import APIError, DataError, ConfigurationError
from matkit.core.logging import get_logger
from matkit.data.cache import DataCache

logger = get_logger(__name__)


class BaseFetcher(ABC):
    """Base class for all data fetchers."""
    
    def __init__(self, cache: Optional[DataCache] = None):
        """Initialize fetcher with optional cache."""
        self.config = get_config()
        self.cache = cache or DataCache()
    
    @abstractmethod
    async def fetch_async(self, **kwargs) -> pd.DataFrame:
        """Async fetch method to be implemented by subclasses."""
        pass
    
    def fetch(self, **kwargs) -> pd.DataFrame:
        """Synchronous wrapper for async fetch."""
        return asyncio.run(self.fetch_async(**kwargs))
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    async def _make_request(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        try:
            response = await client.get(
                url,
                params=params,
                headers=headers,
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise APIError(
                f"HTTP error: {e.response.status_code}",
                status_code=e.response.status_code,
                response_body=e.response.text
            )
        except Exception as e:
            raise APIError(f"Request failed: {str(e)}")


class MaterialsProjectFetcher(BaseFetcher):
    """
    Fetcher for Materials Project database.
    
    Provides access to computed materials properties including structure,
    electronic properties, elastic properties, and more.
    """
    
    def __init__(self, api_key: Optional[str] = None, cache: Optional[DataCache] = None):
        """
        Initialize Materials Project fetcher.
        
        Parameters
        ----------
        api_key : str, optional
            Materials Project API key. If None, uses environment variable
        cache : DataCache, optional
            Cache instance for storing results
        """
        super().__init__(cache)
        self.api_key = api_key or self.config.mp_api_key
        if not self.api_key:
            raise ConfigurationError(
                "Materials Project API key not provided. "
                "Set MP_API_KEY environment variable or pass api_key parameter."
            )
    
    async def fetch_async(
        self,
        criteria: Optional[Dict[str, Any]] = None,
        properties: Optional[List[str]] = None,
        chunk_size: int = 1000,
        max_results: Optional[int] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch materials data from Materials Project.
        
        Parameters
        ----------
        criteria : dict, optional
            Query criteria (e.g., {"elements": {"$all": ["Li", "Fe", "O"]}})
        properties : list, optional
            Properties to retrieve. If None, fetches default set
        chunk_size : int, default=1000
            Number of materials to fetch per request
        max_results : int, optional
            Maximum number of results to return
        use_cache : bool, default=True
            Whether to use cached results
            
        Returns
        -------
        pd.DataFrame
            Materials data with requested properties
        """
        # Check cache
        cache_key = f"mp_{criteria}_{properties}_{max_results}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info("Using cached Materials Project data")
                return cached
        
        # Default properties
        if properties is None:
            properties = [
                "material_id",
                "formula_pretty",
                "structure",
                "energy_per_atom",
                "formation_energy_per_atom",
                "band_gap",
                "total_magnetization",
                "volume",
                "density",
                "symmetry"
            ]
        
        logger.info(f"Fetching data from Materials Project with criteria: {criteria}")
        
        try:
            with MPRester(self.api_key) as mpr:
                # Use summary search for efficiency
                # Note: Some criteria might need to be converted for the current API
                search_criteria = {}
                target_nelements = None
                
                if criteria:
                    # Handle elements parameter
                    if "elements" in criteria:
                        elements_criteria = criteria["elements"]
                        if isinstance(elements_criteria, dict) and "$all" in elements_criteria:
                            # Convert {"$all": ["N"]} to elements=["N"]
                            search_criteria["elements"] = elements_criteria["$all"]
                        else:
                            search_criteria["elements"] = elements_criteria
                    
                    # Handle nelements parameter (filter after fetching)
                    if "nelements" in criteria:
                        target_nelements = criteria["nelements"]
                    
                    # Copy other criteria
                    for key, value in criteria.items():
                        if key not in ["elements", "nelements"]:
                            search_criteria[key] = value
                
                # Calculate num_chunks properly
                num_chunks = None
                if max_results:
                    num_chunks = max(1, max_results // chunk_size)
                
                results = mpr.summary.search(
                    **search_criteria,
                    fields=properties,
                    chunk_size=chunk_size,
                    num_chunks=num_chunks
                )
                
                # Convert to DataFrame
                data_list = []
                count = 0
                
                for doc in results:
                    if max_results and count >= max_results:
                        break
                    
                    # Extract data
                    row = {}
                    structure = None
                    
                    for prop in properties:
                        value = getattr(doc, prop, None)
                        
                        # Special handling for structures
                        if prop == "structure" and value:
                            structure = value
                            row["structure"] = value
                            row["formula"] = value.composition.reduced_formula
                            row["n_atoms"] = len(value)
                        elif prop == "symmetry" and value:
                            row["space_group"] = value.symbol
                            row["crystal_system"] = value.crystal_system
                        else:
                            row[prop] = value
                    
                    # Filter by number of elements if specified
                    if target_nelements is not None:
                        if structure:
                            n_elements = len(structure.composition.elements)
                            if n_elements != target_nelements:
                                continue
                        else:
                            # Skip if we can't determine number of elements
                            continue
                    
                    data_list.append(row)
                    count += 1
                
                df = pd.DataFrame(data_list)
                logger.info(f"Retrieved {len(df)} materials from Materials Project")
                
                # Cache results
                if use_cache:
                    self.cache.set(cache_key, df)
                
                return df
                
        except Exception as e:
            raise APIError(f"Failed to fetch from Materials Project: {str(e)}")


class CIFFetcher(BaseFetcher):
    """Fetcher for Crystallography Information Files (CIF)."""
    
    async def fetch_async(
        self,
        file_paths: Optional[List[Union[str, Path]]] = None,
        directory: Optional[Union[str, Path]] = None,
        recursive: bool = True
    ) -> pd.DataFrame:
        """
        Load structures from CIF files.
        
        Parameters
        ----------
        file_paths : list of str or Path, optional
            Specific CIF files to load
        directory : str or Path, optional
            Directory containing CIF files
        recursive : bool, default=True
            Whether to search recursively in directory
            
        Returns
        -------
        pd.DataFrame
            Structures loaded from CIF files
        """
        if not file_paths and not directory:
            raise ValueError("Either file_paths or directory must be provided")
        
        # Collect all CIF files
        cif_files = []
        if file_paths:
            cif_files.extend([Path(f) for f in file_paths])
        
        if directory:
            dir_path = Path(directory)
            if recursive:
                cif_files.extend(dir_path.rglob("*.cif"))
            else:
                cif_files.extend(dir_path.glob("*.cif"))
        
        logger.info(f"Loading {len(cif_files)} CIF files")
        
        # Load structures
        data_list = []
        for cif_file in cif_files:
            try:
                structure = Structure.from_file(str(cif_file))
                data_list.append({
                    "file_name": cif_file.name,
                    "file_path": str(cif_file),
                    "structure": structure,
                    "formula": structure.composition.reduced_formula,
                    "n_atoms": len(structure),
                    "volume": structure.volume,
                    "density": structure.density
                })
            except Exception as e:
                logger.warning(f"Failed to load {cif_file}: {e}")
                if not self.config.skip_errors:
                    raise DataError(f"Failed to parse CIF file {cif_file}: {str(e)}")
        
        df = pd.DataFrame(data_list)
        logger.info(f"Successfully loaded {len(df)} structures from CIF files")
        
        return df


class CODFetcher(BaseFetcher):
    """
    Fetcher for Crystallography Open Database (COD).
    
    Provides access to experimental crystal structures.
    """
    
    BASE_URL = "https://www.crystallography.net/cod/search.html"
    
    async def fetch_async(
        self,
        formula: Optional[str] = None,
        spacegroup: Optional[str] = None,
        elements: Optional[List[str]] = None,
        max_results: int = 100
    ) -> pd.DataFrame:
        """
        Search and fetch structures from COD.
        
        Parameters
        ----------
        formula : str, optional
            Chemical formula to search
        spacegroup : str, optional
            Space group symbol
        elements : list of str, optional
            Elements that must be present
        max_results : int, default=100
            Maximum number of results
            
        Returns
        -------
        pd.DataFrame
            Crystal structures from COD
        """
        # Build query parameters
        params = {
            "format": "json",
            "max": max_results
        }
        
        if formula:
            params["formula"] = formula
        if spacegroup:
            params["sg"] = spacegroup
        if elements:
            params["elements"] = ",".join(elements)
        
        logger.info(f"Searching COD with parameters: {params}")
        
        async with httpx.AsyncClient() as client:
            response_data = await self._make_request(client, self.BASE_URL, params)
        
        # Process results
        data_list = []
        for entry in response_data.get("results", []):
            try:
                # Download CIF
                cif_url = f"https://www.crystallography.net/cod/{entry['cod_id']}.cif"
                async with httpx.AsyncClient() as client:
                    cif_response = await client.get(cif_url)
                    cif_response.raise_for_status()
                
                # Parse structure
                structure = Structure.from_str(cif_response.text, fmt="cif")
                
                data_list.append({
                    "cod_id": entry["cod_id"],
                    "formula": entry.get("formula", ""),
                    "spacegroup": entry.get("sg", ""),
                    "structure": structure,
                    "a": entry.get("a"),
                    "b": entry.get("b"),
                    "c": entry.get("c"),
                    "alpha": entry.get("alpha"),
                    "beta": entry.get("beta"),
                    "gamma": entry.get("gamma"),
                    "volume": structure.volume,
                    "density": structure.density,
                    "authors": entry.get("authors", ""),
                    "title": entry.get("title", ""),
                    "journal": entry.get("journal", ""),
                    "year": entry.get("year")
                })
                
            except Exception as e:
                logger.warning(f"Failed to process COD entry {entry.get('cod_id')}: {e}")
        
        df = pd.DataFrame(data_list)
        logger.info(f"Retrieved {len(df)} structures from COD")
        
        return df


class MaterialsDataFetcher:
    """
    Unified interface for fetching materials data from multiple sources.
    
    This class provides a high-level API for accessing various materials
    databases with automatic source detection and result merging.
    """
    
    def __init__(
        self,
        mp_api_key: Optional[str] = None,
        cache: Optional[DataCache] = None
    ):
        """Initialize unified fetcher."""
        self.cache = cache or DataCache()
        self.mp_fetcher = MaterialsProjectFetcher(mp_api_key, self.cache)
        self.cif_fetcher = CIFFetcher(self.cache)
        self.cod_fetcher = CODFetcher(self.cache)
    
    async def fetch_async(
        self,
        source: str = "mp",
        merge_duplicates: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch materials data from specified source.
        
        Parameters
        ----------
        source : str, default="mp"
            Data source ("mp", "cod", "cif", or "all")
        merge_duplicates : bool, default=True
            Whether to merge duplicate structures
        **kwargs
            Source-specific parameters
            
        Returns
        -------
        pd.DataFrame
            Materials data
        """
        logger.info(f"Fetching materials data from source: {source}")
        
        if source == "mp":
            df = await self.mp_fetcher.fetch_async(**kwargs)
        elif source == "cod":
            df = await self.cod_fetcher.fetch_async(**kwargs)
        elif source == "cif":
            df = await self.cif_fetcher.fetch_async(**kwargs)
        elif source == "all":
            # Fetch from all sources and merge
            tasks = []
            if "mp_kwargs" in kwargs:
                tasks.append(self.mp_fetcher.fetch_async(**kwargs.get("mp_kwargs", {})))
            if "cod_kwargs" in kwargs:
                tasks.append(self.cod_fetcher.fetch_async(**kwargs.get("cod_kwargs", {})))
            if "cif_kwargs" in kwargs:
                tasks.append(self.cif_fetcher.fetch_async(**kwargs.get("cif_kwargs", {})))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            dfs = []
            for result in results:
                if isinstance(result, pd.DataFrame):
                    dfs.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Failed to fetch from source: {result}")
            
            if not dfs:
                raise DataError("Failed to fetch data from any source")
            
            df = pd.concat(dfs, ignore_index=True)
            
            # Mark source
            df["source"] = source
        else:
            raise ValueError(f"Unknown source: {source}")
        
        # Merge duplicates if requested
        if merge_duplicates and "structure" in df.columns:
            df = self._merge_duplicates(df)
        
        return df
    
    def fetch(self, **kwargs) -> pd.DataFrame:
        """Synchronous wrapper for fetch_async."""
        return asyncio.run(self.fetch_async(**kwargs))
    
    def _merge_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge duplicate structures based on composition and space group."""
        logger.info("Merging duplicate structures")
        
        # Create unique key
        df["_key"] = df.apply(
            lambda x: f"{x.get('formula', '')}__{x.get('space_group', '')}",
            axis=1
        )
        
        # Group and take first of each group
        unique_df = df.groupby("_key").first().reset_index(drop=True)
        
        logger.info(f"Reduced from {len(df)} to {len(unique_df)} unique structures")
        
        return unique_df