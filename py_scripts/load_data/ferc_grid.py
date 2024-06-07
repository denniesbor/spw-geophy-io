from pathlib import Path
import geopandas as gpd

# Get the absolute path to the directory containing this script
current_dir = Path(__file__).resolve().parent

# Construct the path to the data directory
data_dir = current_dir.parent / 'data'

class LoadFERCGridRegions:
    """
    A class to process the FERC Electric Power Grid Regions dataset.

    Parameters
    ----------
    ferc_path : str, optional
        The path to the FERC grid regions dataset, by default 'data/ferc_regions/FERC_grid_regions.geojson'
    region : str, optional
        The region to load, by default None

    Attributes
    ----------
    ferc_path : str
        The path to the FERC grid regions dataset.
    us_crs : str
        The coordinate reference system (CRS) for the United States.
    region : str or None
        The region to load.
    grid_regions : gpd.GeoDataFrame
        The loaded and processed FERC grid regions dataset.

    Methods
    -------
    load_ferc_grid_regions()
        Load the FERC grid regions dataset.

    """

    def __init__(self, ferc_path: str = data_dir / "ferc_regions"/ 'FERC_grid_regions.geojson', region=None):
        self.ferc_path = ferc_path
        
        print(self.ferc_path)
        self.us_crs = 'EPSG:4326'
        self.region = region
        self.grid_regions = self.load_ferc_grid_regions()
        
    def load_ferc_grid_regions(self):
        """
        Load the FERC grid regions dataset.

        Returns
        -------
        gpd.GeoDataFrame
            The loaded and processed FERC grid regions dataset.

        """
        # load the FERC grid regions dataset
        grid_regions = gpd.read_file(self.ferc_path)
        # set the CRS to EPSG:4326
        grid_regions = grid_regions.to_crs(self.us_crs)
        
        if self.region is not None:
            grid_regions = grid_regions[grid_regions.REGION == self.region]
        return grid_regions