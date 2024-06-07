from pathlib import Path
import geopandas as gpd
import bezpy
from .ferc_grid import LoadFERCGridRegions

# Get the absolute path to the directory containing this script
current_dir = Path(__file__).resolve().parent

# Construct the path to the data directory
data_dir = current_dir.parent / 'data'


class LoadTLData(LoadFERCGridRegions):
    """
    A class for loading and processing transmission lines data.

    Parameters
    ----------
    tl_path : str, optional
        The path to the transmission lines dataset, by default 'data/US_transmission_lines.geojson'

    Attributes
    ----------
    tl_path : str
        The path to the transmission lines dataset.
    us_crs : str
        The coordinate reference system (CRS) for the transmission lines dataset.
    transmission_lines : gpd.GeoDataFrame
        The loaded and processed transmission lines dataset.

    Methods
    -------
    spatial_join_data()
        Perform a spatial join between the transmission lines and the FERC grid regions.
    load_transmission_lines()
        Load the transmission lines dataset.

    """

    def __init__(self, tl_path: str = data_dir / 'US_transmission_lines.geojson', ferc_path = data_dir / "ferc_regions"/ 'FERC_grid_regions.geojson', region=None):
        super().__init__(region=region, ferc_path=ferc_path)
        self.tl_path = tl_path
        self.us_crs = 'EPSG:4326'
        self.transmission_lines = self.load_transmission_lines()
        self.spatial_join_data()
        
    def spatial_join_data(self):
        """
        Perform a spatial join between the transmission lines and the FERC grid regions.
        """
        self.transmission_lines = self.transmission_lines.sjoin(self.grid_regions, how='inner', predicate='intersects')
        
    def load_transmission_lines(self):
        """
        Load the transmission lines dataset.

        Returns
        -------
        gpd.GeoDataFrame
            The loaded and processed transmission lines dataset.

        """
        # load the transmission lines dataset
        transmission_lines = gpd.read_file(self.tl_path)
        
        # set the CRS to EPSG:4326
        return transmission_lines.to_crs(self.us_crs)
        
