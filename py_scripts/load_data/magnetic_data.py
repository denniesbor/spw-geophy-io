from pathlib import Path
import os
from datetime import datetime
import warnings
import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from multiprocessing import Pool
from .ferc_grid import LoadFERCGridRegions

warnings.filterwarnings("ignore")
# Get the absolute path to the directory containing this script
current_dir = Path(__file__).resolve().parent

# Construct the path to the data directory
data_dir = current_dir.parent / 'data'
mag_data_folder = data_dir / 'magnetic_data'


class MagneticData(LoadFERCGridRegions):
    """
    A class to process data from the Intermagnet geomagnetic observatories in its raw format.

    Parameters:
    mag_data_folder (str): The path to the folder containing the magnetic data.
    year_of_data (str): The year of the data to process.
    region (str): The ferc grid region to filter the data. Default is None (entire CONUS).
    ferc_path (str): The path to the FERC grid regions file. Default is data_dir / "ferc_regions"/ 'FERC_grid_regions.geojson'.

    Attributes:
    magnetic_folder (str): The path to the magnetic data folder.
    region (str): The region to filter the data.
    usgs_obs (list): List of USGS observatories.
    nrcan_obs (list): List of NRCan observatories.
    obsv_xarrays (dict): Dictionary containing the processed observatory data as xarray datasets.
    obsv_sites_gdf (GeoDataFrame): GeoDataFrame containing the observatory sites with spatial information.

    Methods:
    spatial_join_data(): Perform a spatial join between the observatory sites and the FERC grid regions.
    convert_hdz_to_xyz(df): Convert the magnetic data from HDZ format to XYZ format.
    process_magnetic_files(file_path): Process a magnetic file and return a pandas DataFrame.
    process_directory(dir): Process a directory containing magnetic files of different observatories.
    """

    def __init__(self, mag_data_folder=mag_data_folder, year_of_data="2024", region=None, ferc_path=data_dir / "ferc_regions"/ 'FERC_grid_regions.geojson'):
        super().__init__(region=region, ferc_path=ferc_path)
        self.magnetic_folder = os.path.join(mag_data_folder, year_of_data)
        self.region = region
        self.usgs_obs = ['bou', 'brw', 'bsl', 'cmo', 'ded', 'dlr', 'frd', 'frn', 'gua', 'hon', 'new',
                         'shu', 'sit', 'sjg', 'tuc']
        self.nrcan_obs = ['blc', 'cbb', 'fcc', 'mea', 'ott', 'res', 'stj', 'vic', 'ykc']
        
        observatory_dirs = [dir for dir in os.listdir(self.magnetic_folder) if dir.lower() in self.usgs_obs + self.nrcan_obs]
        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(self.process_directory, observatory_dirs)
        
        self.obsv_xarrays = {dir: dataset for dir, dataset in results if dataset is not None}
        self.start_time = datetime.utcfromtimestamp(self.obsv_xarrays["BOU"].Timestamp[0].data.astype('O')/1e9)
        self.end_time  = datetime.utcfromtimestamp(self.obsv_xarrays["BOU"].Timestamp[-1].data.astype('O')/1e9)
        OBS_data = {
            'obsv_name': [name for name in self.obsv_xarrays],
            'geometry': [Point(self.obsv_xarrays[name].attrs['Longitude'], self.obsv_xarrays[name].attrs['Latitude']) for name in self.obsv_xarrays]
        }

        self.obsv_sites_gdf = gpd.GeoDataFrame(OBS_data, crs=self.us_crs)
        # self.spatial_join_data()

    def spatial_join_data(self):
        """
        Perform a spatial join between the observatory sites and the FERC grid regions.
        """
        self.obsv_sites_gdf = self.obsv_sites_gdf.sjoin(self.grid_regions, how='inner', predicate='intersects')


    # Function to convert HDZ to XYZ
    @staticmethod
    def convert_hdz_to_xyz(df):
        """
        Convert the magnetic data from HDZ format to XYZ format.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the HDZ data.

        Returns:
        pd.DataFrame: The DataFrame with the XYZ data.
        """
        df['H'] = df['H'] / 60  # Convert hour angle from minutes to degrees
        df['H_rad'] = np.radians(df['H'])  # Convert degrees to radians
        df['D'] = df['D'] / 60  # Convert declination from minutes to degrees
        df['D_rad'] = np.radians(df['D'])  # Convert degrees to radians
        df['X'] = df['H'] * np.cos(df['D_rad'])  # Calculate X component
        df['Y'] = df['H'] * np.sin(df['D_rad'])  # Calculate Y component
        df['Z'] = df['Z']  # Z remains unchanged
        return df[['Timestamp', 'X', 'Y', 'Z']]

    def process_magnetic_files(self, file_path):
        """
        Process a magnetic file and return a pandas DataFrame.

        Parameters:
        file_path (str): The path to the magnetic file.

        Returns:
        pd.DataFrame: The processed DataFrame.
        """
        metadata_lines = []
        headers = []
        with open(file_path, 'r') as file:
            files = file.readlines()
            start_data = 0
            start_comments = 0
            for i, line in enumerate(files):
                if line.startswith(" #") and start_comments == 0:
                    metadata_lines = files[:i]
                if line.startswith("DATE"):
                    headers = line.split('|')[0].split()
                    headers = headers[:3] + [component[-1] for component in headers[3:]]
                    start_data = i + 1
                    break

        metadata_entries = [line.split('|')[0].strip() for line in metadata_lines]
        delimiter_length = len(metadata_entries[0]) - len(metadata_entries[0].split()[1])

        metadata_dict = {}
        for entry in metadata_entries:
            key = entry[:delimiter_length].strip()
            value = entry[delimiter_length:].strip()
            metadata_dict[key] = value

        data = pd.read_csv(
            file_path,
            skiprows=start_data,
            delim_whitespace=True,
            names=headers,
            parse_dates={'Timestamp': ['DATE', 'TIME']}
        )

        placeholders = [99999, 88888]

        if {'H', 'D', 'Z'}.issubset(data.columns):
            for column in ['H', 'D', 'Z']:
                median_value = data[column].replace(placeholders, np.nan).median()
                data[column] = data[column].replace(placeholders, median_value)
            result = self.convert_hdz_to_xyz(data)
        elif {'X', 'Y', 'Z'}.issubset(data.columns):
            result = data[['Timestamp', 'X', 'Y', 'Z']]
            for column in ['X', 'Y', 'Z']:
                median_value = result[column].replace(placeholders, np.nan).median()
                result[column] = result[column].replace(placeholders, median_value)
        else:
            print("Data columns are missing for either HDZ or XYZ format.")
            return None

        result.set_index('Timestamp', inplace=True)
        ds = xr.Dataset.from_dataframe(result)
        ds.attrs['Latitude'] = float(metadata_dict['Geodetic Latitude'])
        ds.attrs['Longitude'] = float(metadata_dict['Geodetic Longitude']) - 360
        ds.attrs['Name'] = metadata_dict['IAGA Code']
        ds.attrs.update(metadata_dict)
        del ds.attrs['Geodetic Latitude']
        del ds.attrs['Geodetic Longitude']
        del ds.attrs['IAGA Code']
        return ds
    
    def process_directory(self, dir):
        """
        Process a directory containing magnetic files of different observatories.

        Parameters:
        dir (str): The name of the directory.

        Returns:
        tuple: A tuple containing the directory name and the concatenated dataset.
        """
        print(f"Processing {dir}")
        datasets = []
        for filename in sorted(os.listdir(os.path.join(self.magnetic_folder, dir))):
            if filename.endswith('.min'):
                file_path = os.path.join(self.magnetic_folder, dir, filename)
                print(file_path)
                ds = self.process_magnetic_files(file_path)
                if ds is not None:
                    datasets.append(ds)
        if datasets:
            combined_dataset = xr.concat(datasets, dim='Timestamp')
            return (dir, combined_dataset)
        else:
            print(f"Missing datasets to combine in {dir}")
            return (dir, None)

