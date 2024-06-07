
'''
This module loads and prepares the US substations, transmission lines, US EarthScope ground conductivity data, and Intermagnet geomagnetic observatories data.
The datasets are downloaded from:
- US substations: https://hifld-geoplatform.opendata.arcgis.com/datasets/electric-substations
    The raw data used has been partially cleaned and saved as 'data/US_substations.csv'
- US transmission lines: https://hifld-geoplatform.hub.arcgis.com/datasets/geoplatform::transmission-lines/about
    The raw data used has been partially cleaned and saved as 'data/US_transmission_lines.geojson'
- FERC Electric Power Grid Regions: https://hifld-geoplatform.opendata.arcgis.com/datasets/electric-power-grid-regions
    The boundary files are derived from digitizing the FERC grid regions and saved as 'data/FERC_grid_regions.geojson'
- US EarthScope ground conductivity data: https://ds.iris.edu/ds/
- Intermagnet geomagnetic observatories data: https://imag-data.bgs.ac.uk/GIN_V1/GINForms2

_author_: 'Dennies Bor'
requires:
    - pandas
    - geopandas
    - shapely
    - numpy
    - matplotlib
    - xarray
    - bezpy
'''

from .magnetic_data import MagneticData
from .emtf_data import LoadEMTFData
from .tl_data import LoadTLData

class DataManager:
    """
    A class to manage loading and processing of data.

    Parameters:
    ferc_path (str): The path to the FERC data.
    region (str, optional): The region for which the data is loaded. Defaults to None.
    """

    def __init__(self, ferc_path: str, region=None):
        self.ferc_path = ferc_path
        self.region = region
        self.mag_data = None
        self.emtf_data = None
        self.tl_data = None

    def load_magnetic_data(self, mag_data_folder, year_of_data="2024"):
        """
        Load magnetic data from a folder.

        Parameters:
        mag_data_folder (str): The path to the folder containing the magnetic data.
        year_of_data (str): The year of data to load.
        """
        
        print(f"Loading magnetic data from {mag_data_folder} for {year_of_data} and path to FERC data: {self.ferc_path} and region: {self.region}")   
        self.mag_data = MagneticData(mag_data_folder=mag_data_folder, year_of_data=year_of_data, ferc_path=self.ferc_path, region=self.region)

    def load_emtf_data(self, emtf_path: str):
        """
        Load EMTF data from a specified folder.

        Parameters:
        emtf_path (str): The path to the folder containing the EMTF data.
        """
        self.emtf_data = LoadEMTFData(emtf_path, self.ferc_path, self.region)

    def load_tl_data(self, tl_path: str):
        """
        Load ground conductivity transfer functions from a user defined folder.

        Parameters:
        tl_path (str): The path to the folder containing the TL data.
        """
        self.tl_data = LoadTLData(tl_path, self.ferc_path, self.region)