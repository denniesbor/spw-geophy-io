from pathlib import Path
import os
import geopandas as gpd
import numpy as np
import bezpy
from shapely.geometry import Point
from concurrent.futures import ThreadPoolExecutor, as_completed
from .ferc_grid import LoadFERCGridRegions

# Get the absolute path to the directory containing this script
current_dir = Path(__file__).resolve().parent

# Construct the path to the data directory
data_dir = current_dir.parent / 'data'

class LoadEMTFData(LoadFERCGridRegions):
    """
    A class to load ground conductivity transfer functions from the EarthScope Magnetotelluric Facility (EMTF) dataset.
    
    Parameters
    ----------
    emtf_path : str, optional
        The path to the EMTF data directory. Default is `data_dir / 'EMTF'`.
    ferc_path : str, optional
        The path to the FERC grid regions GeoJSON file. Default is `data_dir / "ferc_regions"/ 'FERC_grid_regions.geojson'`.
    """
    def __init__(self, emtf_path: str = data_dir / 'EMTF', ferc_path=data_dir / "ferc_regions"/ 'FERC_grid_regions.geojson', region=None):
        super().__init__(region=region, ferc_path=ferc_path)
        self.emtf_path = emtf_path
        self.us_crs = 'EPSG:4326'
        self.MT_sites = self.process_sites()
        self.mt_sites_gdf = self.load_emtf_data()
        self.filtered_site_xys = np.array([])   
        self.spatial_join_data()
        
    def load_emtf_data(self):
        """
        Load EMTF data into a GeoDataFrame.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing site IDs and geometries.
        """
        return gpd.GeoDataFrame({
            'site_id': [site.name for site in self.MT_sites],
            'geometry': [Point(site.longitude, site.latitude) for site in self.MT_sites]
        }, crs=self.us_crs)
    
    def spatial_join_data(self):
        """
        Perform a spatial join between the MT sites and the FERC grid regions.
        """
        self.mt_sites_gdf = self.mt_sites_gdf.sjoin(self.grid_regions, how='inner', predicate='intersects')
        
        # Extract the indices of the sites that are within the grid regions
        within_indices = self.mt_sites_gdf.index.to_list()

        # Filter MT_sites and site_xys based on the indices from within_indices
        self.filtered_MT_sites = [self.MT_sites[i] for i in within_indices]
        self.filtered_site_xys = np.array([(site.latitude, site.longitude) for site in self.filtered_MT_sites])
        
    def process_xml_file(self, full_path):
        """
        Process an XML file containing a ground conductivity transfer function
        using the bezpy library. Returns None if the rating is less than 3.

        Parameters
        ----------
        full_path : str
            The path to the XML file.

        Returns
        -------
        bezpy.mt.MT
            The processed MT object.
        """
        try:
            site_name = os.path.basename(full_path).split('.')[1]
            site = bezpy.mt.read_xml(full_path)

            if site.rating < 3:
                print("Site rating is low")
                return None
            return site
        except Exception as e:
            print(f"Error processing {full_path}: {e}")
            return None

    def process_sites(self):
        """
        Process all XML files in the specified directory to extract MT site information.

        Returns
        -------
        list
            List of MT site objects.
        """
        print("Processing MT sites")
        MT_sites = []
        completed = 0

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for root, dirs, files in os.walk(self.emtf_path):
                for file in files:
                    if file.endswith('.xml'):
                        full_path = os.path.join(root, file)
                        futures.append(executor.submit(self.process_xml_file, full_path))

            for future in as_completed(futures):
                result = future.result()
                if result:
                    completed += 1
                    MT_sites.append(result)

                    if completed % 100 == 0:
                        print("Completed", completed)
        return MT_sites