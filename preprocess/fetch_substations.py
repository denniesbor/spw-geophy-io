# ----------------------------------------------------------------------------------
# Script to download the power substation data from the OpenStreetMap database.
# Utilize the osmnx library to download the data.
# Please install the osmnx library before running this script.
# Author: Dennies Bor - GMU
# ----------------------------------------------------------------------------------
import os
import pickle
import requests
import logging
from pathlib import Path
import warnings
from io import StringIO
from concurrent.futures import ThreadPoolExecutor

import osmnx as ox
import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    filename="fetch_substations.log",
)

# Define the data path
data_path = Path("__file__").resolve().parents[1] / "data"

# Create the data directory if it doesn't exist
os.makedirs(data_path, exist_ok=True)

# Define the URLs
states_url = "https://gist.githubusercontent.com/dantonnoriega/bf1acd2290e15b91e6710b6fd3be0a53/raw/11d15233327c8080c9646c7e1f23052659db251d/us-state-ansi-fips.csv"
counties_url = "https://raw.githubusercontent.com/kjhealy/us-county/master/data/census/fips-by-state.csv"

# Save paths
states_data_path = data_path / "us-state-ansi-fips.csv"
counties_data_path = data_path / "fips-by-state.csv"

download_states_counties = True


def download_data(url, save_path):
    """This function downloads the data from the given URL
    and saves it to the specified path.
    params:
    ---------
    url: str
      URL to download the data from
    save_path: str
      Path to save the downloaded data
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        logging.info(f"{save_path.name} downloaded successfully.")
    else:
        logging.error(
            f"Failed to download {save_path.name}. Status code: {response.status_code}"
        )


if download_states_counties:
    # Download and save the states data
    download_data(states_url, states_data_path)

    # Download and save the counties data
    download_data(counties_url, counties_data_path)


# Routines
def read_text_file(file_path):
    """This custom function reads large text/csv
    files, especially those with encoding errors due
    to compression
    params:
    ---------
    file: str
      data path of the input file
    returns:
    ---------
    dataframe: pd.dataframe
      pandas dataframe
    """

    with open(file_path, encoding="utf-8", errors="ignore") as file:
        content = file.read()

    content_io = StringIO(content)
    return pd.read_csv(content_io, low_memory=False)


# Load the states and counties data
us_states_df = read_text_file(states_data_path)
counties_df = read_text_file(counties_data_path)


# Get the power substations of all the states within the US
def get_pwr_stations(state: str):
    """
    Fetches power substations data for a given state.
    Parameters:
        state (str): The state for which to fetch the data.
    Returns:
        substations (GeoDataFrame): A GeoDataFrame containing the power substations data.
    """

    # Define the area of interest and the tag filters
    area_of_interest = f"{state}, USA"
    tags = {"power": "substation"}

    # Fetch the data
    substations = ox.geometries_from_place(area_of_interest, tags)
    logging.info(f"Power substations data for {state} fetched successfully.")

    return substations


def main():

    states = us_states_df.stname.to_list()

    logging.info("Fetching power substations data for all states...")

    # Using ThreadPoolExecutor to parallelize the requests
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(get_pwr_stations, states))

    # Save the results
    with open(data_path / "gdf_list.pkl", "wb") as file:
        pickle.dump(results, file)

    logging.info("Power substations data saved to gdf_list.pkl")

    # Process and combine the results
    cols = ["name", "operator", "voltage", "nodes", "power", "substation", "geometry"]
    substations_loc = [result[cols] for result in results if result is not None]
    subst_gpd = pd.concat(substations_loc)
    gdf = gpd.GeoDataFrame(subst_gpd, geometry="geometry")

    # Save the combined GeoDataFrame
    gdf.to_file(data_path / "combined_substations.geojson", driver="GeoJSON")
    logging.info("Combined substations data saved to combined_substations.geojson")


if __name__ == "__main__":
    main()
