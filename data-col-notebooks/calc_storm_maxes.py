# --------------------------------------------------------------------------
# This script calculates the maximum magnetic fields, electric fields, and voltages at MT sites during geomagnetic storms.
# It uses the SECS (Spherical Elementary Current Systems) model to fit observatory data and calculate the fields.
# The script processes EMTF data, geomagnetic data, and transmission lines data to calculate the fields.
# Some of the code is adapted from Greg Lucas's 2018 Hazard Paper.
# Author: Dennies Bor
# --------------------------------------------------------------------------
# %%
import os
import time
import sys
import psutil
import logging
import pickle
from pathlib import Path
import multiprocessing
import powerlaw
import xarray as xr
import pandas as gpd
import bezpy
from pysecs import SECS
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# Set up logging, a reusable function
def setup_logging():
    """
    Set up logging to both file and console.
    """
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler("Getting storm maxes.log")
    file_handler.setLevel(logging.INFO)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


setup_logging()


# Routines and modules that are used to extract max Es, Bs, and Vs during a storm
# -----------------------------------------------------------------------
# Looad and Prepare EMTF data
# -----------------------------------------------------------------------
def process_xml_file(full_path):
    """
    Read and parse EMTF XML files.

    Parameters
    ----------
    full_path : str
        The full path to the XML file.

    Returns
    -------
    bezpy.mt.Site or None
        The processed site object if successful, None otherwise.

    Notes
    -----
    This function reads an EMTF XML file, checks its rating, and returns
    the processed site object if the rating is 3 or higher.
    """
    try:
        site_name = os.path.basename(full_path).split(".")[1]
        site = bezpy.mt.read_xml(full_path)

        if site.rating < 3:
            logging.info(f"Skipped: {full_path} (Outside region or low rating)")
            return None

        logging.info(f"Processed: {full_path}")
        return site
    except Exception as e:
        logging.error(f"Error processing {full_path}: {e}")
        return None


def process_sites(directory):
    """
    Process all XML files in the given directory and its subdirectories.

    Parameters
    ----------
    directory : str
        The root directory to search for XML files.

    Returns
    -------
    list
        A list of processed MT site objects.

    Notes
    -----
    This function uses a ThreadPoolExecutor to process XML files concurrently.
    It prints progress every 100 completed files.
    """
    MT_sites = []
    completed = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".xml"):
                    full_path = os.path.join(root, file)
                    futures.append(executor.submit(process_xml_file, full_path))

        for future in as_completed(futures):
            result = future.result()
            if result:
                completed += 1
                MT_sites.append(result)

                if completed % 100 == 0:
                    logging.info(f"Completed {completed} files")

    logging.info(f"Completed processing {completed} files")

    return MT_sites


def load_mt_sites(mt_sites_pickle, emtf_path):
    """
    Load MT sites from pickle file or process EMTF files.
    """
    if os.path.exists(mt_sites_pickle):
        with open(mt_sites_pickle, "rb") as pkl:
            return pickle.load(pkl)
    else:
        MT_sites = process_sites(emtf_path)
        with open(mt_sites_pickle, "wb") as pkl:
            pickle.dump(MT_sites, pkl)
        return MT_sites


# --------------------------------------------------------------------------
# Load and Prepare geomagnetic data
# --------------------------------------------------------------------------
def process_geomag_data(data_path):

    # Load geomagnetic data
    logging.info(f"Loading geomagnetic data from {data_path}")
    mag_data = xr.open_dataset(data_path)
    logging.info("Geomagnetic data loaded")

    return mag_data


# --------------------------------------------------------------------------
# Load and Prepare transmission lines data
# ----------------------------------------------------------------
def read_transmission_lines(translines_shp, trans_lines_pickle, site_xys):
    """
    Read transmission lines shapefile and pickle the data.

    Parameters
    ----------
    translines_shp : str
        The path to the transmission lines
    translines_pickle : str
        The path to the processed and pickled transmission lines

    Returns
    -------
    geopandas.GeoDataFrame
        The processed transmission lines data
    """

    # Check if transmission lines is pickled
    if os.path.exists(trans_lines_pickle):
        with open(trans_lines_pickle, "rb") as pkl:
            df = pickle.load(pkl)

        return df

    else:

        # Use Delaunay triangulation to set weights
        t1 = time.time()
        # US Transmission lines
        trans_lines_gdf = gpd.read_file(translines_shp)

        # Rename the ID column
        trans_lines_gdf.rename({"ID": "line_id"}, inplace=True, axis=1)

        # Apply crs
        trans_lines_gdf = trans_lines_gdf.to_crs(epsg=4326)

        # Translate MultiLineString to LineString geometries, taking only the first LineString
        trans_lines_gdf.loc[
            trans_lines_gdf["geometry"].apply(lambda x: x.geom_type)
            == "MultiLineString",
            "geometry",
        ] = trans_lines_gdf.loc[
            trans_lines_gdf["geometry"].apply(lambda x: x.geom_type)
            == "MultiLineString",
            "geometry",
        ].apply(
            lambda x: list(x.geoms)[0]
        )

        # Let's focus on high voltage transmissions - > 200 kV
        trans_lines_gdf = trans_lines_gdf[(trans_lines_gdf["VOLTAGE"] >= 200)]

        # Use bezpy to get the tranmsission line lenght
        trans_lines_gdf["obj"] = trans_lines_gdf.apply(
            bezpy.tl.TransmissionLine, axis=1
        )
        trans_lines_gdf["length"] = trans_lines_gdf.obj.apply(lambda x: x.length)

        trans_lines_gdf.obj.apply(lambda x: x.set_delaunay_weights(site_xys))
        print("Done filling interpolation weights: {0} s".format(time.time() - t1))

        # Remove lines with bad integration
        E_test = np.ones((1, len(site_xys), 2))
        arr_delaunay = np.zeros(shape=(1, len(trans_lines_gdf)))
        for i, tLine in enumerate(trans_lines_gdf.obj):
            arr_delaunay[:, i] = tLine.calc_voltages(E_test, how="delaunay")

        # Filter the trans_lines_gdf
        trans_lines_gdf_not_na = trans_lines_gdf[~np.isnan(arr_delaunay[0, :])]

        # Pickle the trans_lines_gdf
        with open(trans_lines_pickle, "wb") as pkl:
            pickle.dump(trans_lines_gdf_not_na, pkl)

        return trans_lines_gdf_not_na


# %%
# --------------------------------------------------------------------------
# Prepare the paths and load the datasets
# --------------------------------------------------------------------------
data_dir = Path("__file__").resolve().parent / "data"  # working dir
emtf_path = data_dir / "EMTF"  # Path to EMTF data
mt_sites_pickle = emtf_path / "mt_pickle.pkl"  # Path to pickled MT sites
mag_data_path = data_dir / "processed_geomag_data.nc"  # Path to geomagnetic data
translines_path = (
    data_dir / "Electric__Power_Transmission_Lines"
)  # Path to transmission lines data
trans_lines_pickle = (
    translines_path / "trans_lines_pickle.pkl"
)  # Path to pickled transmission lines
translines_shp = (
    translines_path / "Electric__Power_Transmission_Lines.shp"
)  # Path to transmission lines shapefile

# Read storm periods data
storm_data_loc = data_dir / "kp_ap_indices" / "storm_periods.csv"
storm_df = pd.read_csv(storm_data_loc)
storm_df["Start"] = pd.to_datetime(storm_df["Start"])
storm_df["End"] = pd.to_datetime(storm_df["End"])

# Load datasets
magnetic_data = process_geomag_data(mag_data_path)
MT_sites = load_mt_sites(mt_sites_pickle, emtf_path)

# # Let's slice MT_sites for tersting (with 10 sites only)
# MT_sites = MT_sites[:10]
site_xys = [(site.latitude, site.longitude) for site in MT_sites]

# Load transmission lines
df = read_transmission_lines(translines_shp, trans_lines_pickle, site_xys)
n_trans_lines = df.shape[0]

# Prepare data for SECS
obs_dict = {
    site.lower(): magnetic_data.sel(site=site) for site in magnetic_data.site.values
}

# %%
# --------------------------------------------------------------------------
# Fit the observatory data to ionospheric model -secs
# Compute the electric field and magnetic field at the MT sites
# Adapted from pysecs by Greg Lucas
# --------------------------------------------------------------------------
from pysecs import SECS
import datetime
from shapely.geometry import LineString

R_earth = 6371e3


def calculate_SECS(B, obs_xy, pred_xy):
    """Calculate SECS output magnetic field

    B shape: (ntimes, nobs, 3 (xyz))

    obs_xy shape: (nobs, 2 (lat, lon))

    pred_xy shape: (npred, 2 (lat, lon))"""
    if obs_xy.shape[0] != B.shape[1]:
        raise ValueError("Number of observation points doesn't match B input")

    obs_lat_lon_r = np.zeros((len(obs_xy), 3))
    obs_lat_lon_r[:, 0] = obs_xy[:, 0]
    obs_lat_lon_r[:, 1] = obs_xy[:, 1]
    obs_lat_lon_r[:, 2] = R_earth

    # B = np.dstack((B, np.full(B.shape[:2], 1)))

    B_std = np.ones(B.shape)
    B_std[..., 2] = np.inf

    # specify the SECS grid
    lat, lon, r = np.meshgrid(
        np.linspace(15, 85, 36),
        np.linspace(-175, -25, 76),
        R_earth + 110000,
        indexing="ij",
    )
    secs_lat_lon_r = np.hstack(
        (lat.reshape(-1, 1), lon.reshape(-1, 1), r.reshape(-1, 1))
    )

    secs = SECS(sec_df_loc=secs_lat_lon_r)

    secs.fit(obs_loc=obs_lat_lon_r, obs_B=B, obs_std=B_std, epsilon=0.05)

    # Create prediction points
    pred_lat_lon_r = np.zeros((len(pred_xy), 3))
    pred_lat_lon_r[:, 0] = pred_xy[:, 0]
    pred_lat_lon_r[:, 1] = pred_xy[:, 1]
    pred_lat_lon_r[:, 2] = R_earth

    B_pred = secs.predict_B(pred_lat_lon_r)

    return B_pred


# --------------------------------------------------------------------------
# Calculate the maxes for all storms
# --------------------------------------------------------------------------
def calculate_maxes(start_time, end_time, calcV=False):
    """
    Calculate maximum magnetic fields, electric fields, and voltages for given time range and observation data.

    Parameters
    ----------
    start_time : datetime
        The start time of the calculation period.
    end_time : datetime
        The end time of the calculation period.
    calcV : bool, optional
        Whether to calculate voltages, by default True.
    obs_dict : dict, optional
        Dictionary of observation datasets, by default None.

    Returns
    -------
    tuple
        A tuple containing:
        - site_maxB : ndarray
            Maximum magnetic field magnitude for each site.
        - site_maxE : ndarray
            Maximum electric field magnitude for each site.
        - line_maxV : ndarray
            Maximum voltage magnitude for each transmission line (if calcV is True).

    Notes
    -----
    This function processes observation data, calculates SECS (Spherical Elementary Current Systems),
    and determines maximum values for magnetic fields, electric fields, and optionally, voltages at the MT sites.
    It uses interpolation and filtering techniques on the input data.
    """
    t0 = time.time()

    obs_xy = []
    B_obs = []
    site_xys = np.array([(site.latitude, site.longitude) for site in MT_sites])

    for name, dataset in obs_dict.items():
        data = dataset.loc[{"time": slice(start_time, end_time)}].interpolate_na("Time")
        if len(data["time"]) == 0:
            continue

        data = np.array(data.loc[{"time": slice(start_time, end_time)}].to_array().T)
        if np.any(np.isnan(data)):
            continue

        obs_xy.append((dataset.latitude, dataset.longitude))
        B_obs.append(data)

    obs_xy = np.squeeze(np.array(obs_xy))
    B_obs = np.squeeze(np.array(B_obs)).transpose(2, 0, 1)

    B_pred = calculate_SECS(B_obs, obs_xy, site_xys)
    logging.info(f"Done calculating magnetic fields: {time.time() - t0}")

    site_maxB = np.max(np.sqrt(B_pred[:, :, 0] ** 2 + B_pred[:, :, 1] ** 2), axis=0)

    E_pred = np.zeros((len(B_obs), len(site_xys), 2))
    for i, site in enumerate(MT_sites):
        Ex, Ey = site.convolve_fft(B_pred[:, i, 0], B_pred[:, i, 1], dt=60)
        E_pred[:, i, 0] = Ex
        E_pred[:, i, 1] = Ey

    logging.info(f"Done calculating electric fields: {time.time() - t0}")
    site_maxE = np.max(np.sqrt(E_pred[:, :, 0] ** 2 + E_pred[:, :, 1] ** 2), axis=0)

    if calcV:
        logging.info("Calculating voltages...")
        arr_delaunay = np.zeros(shape=(E_pred.shape[0], n_trans_lines))
        for i, tLine in enumerate(df.obj):
            arr_delaunay[:, i] = tLine.calc_voltages(E_pred, how="delaunay")
        line_maxV = np.nanmax(np.abs(arr_delaunay), axis=0)
        logging.info(f"Done calculating voltages: {time.time() - t0}")
    else:
        logging.info("Skipping voltage calculation")
        line_maxV = np.zeros(n_trans_lines)

    return (site_maxB, site_maxE, line_maxV)


# --------------------------------------------------------------------------
# Calculating storm maxes
# --------------------------------------------------------------------------
def process_storm(args):
    """
    Process a single storm event.

    Parameters
    ----------
    args : tuple
        Contains (i, row, calcV, MT_sites, obs_dict)

    Returns
    -------
    tuple
        Contains (i, maxB, maxE, maxV)
    """
    i, row, calcV = args
    storm_times = (row["Start"], row["End"])
    logging.info(f"Working on storm: {i + 1}")
    maxB, maxE, maxV = calculate_maxes(storm_times[0], storm_times[1], calcV)
    return i, maxB, maxE, maxV


# %%
def main():

    # Log done loading data, and print unique obs in obs_dict
    logging.info(f"Done loading data, Obs in obs_dict: {obs_dict.keys()}")

    CALCULATE_VALUES = True
    if CALCULATE_VALUES:
        t0 = time.time()
        logging.info(f"Starting to calculate storm maxes...")

    # if CALCULATE_VALUES:
    #     n_storms = len(storm_df)
    #     n_sites = len(site_xys)

    #     maxE_arr = np.zeros((n_sites, n_storms))
    #     maxB_arr = np.zeros((n_sites, n_storms))
    #     maxV_arr = np.zeros((n_trans_lines, n_storms))

    #     calcV = True

    #     args = [
    #         (i, row, calcV, MT_sites, df, obs_dict) for i, row in storm_df.iterrows()
    #     ]

    #     # Use multiprocessing with 4 workers
    #     for arg in tqdm(args):
    #         i, maxB, maxE, maxV = process_storm(arg)

    #         maxB_arr[:, i] = maxB
    #         maxE_arr[:, i] = maxE
    #         maxV_arr[:, i] = maxV

    #     logging.info(f"Done calculating storm maxes: {time.time() - t0}")

    #     # Save results
    #     np.save(data_dir / "maxB_arr.npy", maxB_arr)
    #     np.save(data_dir / "maxE_arr.npy", maxE_arr)
    #     np.save(data_dir / "maxV_arr.npy", maxV_arr)

    #     logging.info(f"Saved results to {data_dir}")

    if CALCULATE_VALUES:
        n_storms = len(storm_df)
        n_sites = len(site_xys)

        maxE_arr = np.zeros((n_sites, n_storms))
        maxB_arr = np.zeros((n_sites, n_storms))
        maxV_arr = np.zeros((n_trans_lines, n_storms))

        calcV = True
        args = [(i, row, calcV) for i, row in storm_df.iterrows()]

        # Use multiprocessing with a pool of 8 workers / half the number of cores
        # Asjust according to your system
        with multiprocessing.Pool(12) as pool:
            results = pool.map(process_storm, args)

        for result in tqdm(results):
            i, maxB, maxE, maxV = result
            maxB_arr[:, i] = maxB
            maxE_arr[:, i] = maxE
            maxV_arr[:, i] = maxV

        logging.info(f"Done calculating storm maxes: {time.time() - t0}")

        # Save results
        np.save(data_dir / "maxB_arr_testing.npy", maxB_arr)
        np.save(data_dir / "maxE_arr_testing.npy", maxE_arr)
        np.save(data_dir / "maxV_arr_testing.npy", maxV_arr)

        logging.info(f"Saved results to {data_dir}")


if __name__ == "__main__":
    main()

# %%
