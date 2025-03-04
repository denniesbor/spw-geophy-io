"""
Script to process time-series geomagnetic data sourced from various locations. 

Authors:
- Dennies Bor
- Ed Oughton

Date:
- February 2025
"""
import os
import logging
from pathlib import Path
from multiprocessing import Pool
import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats, signal
import bezpy.mag
from itertools import chain
import collections

DATA_LOC = Path("__file__").resolve().parent.parent / "spw-geophy-io" / "data"

def process_magnetic_files(file_path, is_processed=False):
    """
    Process magnetic field data from raw or processed files.

    """
    if is_processed:
        data = pd.read_csv(file_path, index_col=0)
        data.index = pd.to_datetime(data.index)
        Latitude, Longitude = data["Latitude"].iloc[0], data["Longitude"].iloc[0]
        iaga_code = Path(file_path).parent.name
    else:
        data, headers = bezpy.mag.read_iaga(file_path, return_header=True)
        data.index.name = "Timestamp"
        for component in ["X", "Y", "Z"]:
            data[component] = data[component].interpolate(method="nearest").fillna(method="bfill").fillna(method="ffill")
            data[component] = signal.detrend(data[component])
        Latitude, Longitude = float(headers["geodetic latitude"]), float(headers["geodetic longitude"]) - 360
        iaga_code = headers["iaga code"]
        data["Latitude"], data["Longitude"] = Latitude, Longitude
    
    ds = xr.Dataset.from_dataframe(data)
    ds.attrs.update({"Latitude": Latitude, "Longitude": Longitude, "Name": iaga_code})

    return ds, data


def process_directory(dir_path):
    """
    Process all magnetic field files in a given directory.
    
    """
    logging.info(f"Processing {dir_path}")
    def process_file(filename):
        if filename.endswith((".min", ".csv")):
            file_path = os.path.join(dir_path, filename)
            try:
                ds, result = process_magnetic_files(file_path, is_processed=filename.endswith(".csv"))
                result["site_id"] = os.path.basename(dir_path)
                return ds, result
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {str(e)}")
        return None, None
    
    results = list(filter(lambda x: x[0] is not None, map(process_file, sorted(os.listdir(dir_path)))))
    datasets, file_results = zip(*results) if results else ([], [])
    if datasets:
        return os.path.basename(dir_path), xr.concat(datasets, dim="Timestamp"), list(file_results)
    logging.warning(f"No valid datasets found in {dir_path}")

    return os.path.basename(dir_path), None, None


def process_all_directories(geomag_folder, usgs_obs, nrcan_obs):
    """
    Process all directories containing geomagnetic observation data.

    """
    all_dirs = [os.path.join(root, d) for root, dirs, _ in os.walk(geomag_folder) for d in dirs if d.upper() in usgs_obs + nrcan_obs]
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_directory, all_dirs)
    obsv_xarrays = {dir: dataset for dir, dataset, _ in results if dataset is not None and not np.isnan(dataset.X.min().values)}
    
    return results, obsv_xarrays


def combine_results(results, obsv_xarrays):
    """
    Combine processed results into a single dataset.
    
    """
    return list(chain.from_iterable(result_list for _, _, result_list in 
        results if result_list and result_list[0].site_id.iloc[0] in obsv_xarrays))


def get_mode(series):
    """
    Get the mode of a series.
    
    """
    mode_result = stats.mode(series)

    return mode_result.mode if mode_result.mode.size > 0 else np.nan


def prepare_dataset(combined_df):
    """
    Prepare dataset for analysis by structuring and sorting data.
    
    """
    combined_df = combined_df.sort_values(by=["site_id", "Timestamp"])
    unique_lat_lon = combined_df.groupby("site_id").agg({"Longitude": get_mode, "Latitude": get_mode}).reset_index()
    time_steps = sorted(set(pd.to_datetime(combined_df.index.unique()).to_list()))
    site_ids = unique_lat_lon["site_id"].unique()
    dB = np.stack([combined_df.pivot_table(index=combined_df.index, columns="site_id", values=col, aggfunc="first").reindex(time_steps).values for col in ["X", "Y", "Z"]], axis=-1)
    ds = xr.Dataset(coords={
        "longitude": ("site", unique_lat_lon["Longitude"].values),
        "latitude": ("site", unique_lat_lon["Latitude"].values),
        "site": unique_lat_lon["site_id"].values,
        "component": ["X", "Y", "Z"],
        "time": time_steps,
    }, data_vars={"B": (("time", "site", "component"), dB)})
    
    return ds


def run_function(geomag_folder, usgs_obs, nrcan_obs):
    """
    Main function to process all geomagnetic data and save results.
    
    """
    path = geomag_folder / "combined_geomag_df.csv"
    if not os.path.exists(path):
        results, obsv_xarrays = process_all_directories(geomag_folder, usgs_obs, nrcan_obs)
        logging.info(f"Processed {len(obsv_xarrays)} valid observatories")
        combined_df = pd.concat(combine_results(results, obsv_xarrays))
        combined_df.to_csv(path)
    else:
        combined_df = pd.read_csv(path)
    ds = prepare_dataset(combined_df)
    ds.to_netcdf(geomag_folder / "processed_geomag_data.nc", format="NETCDF4", engine="netcdf4")

    return


if __name__ == "__main__":

    geomag_folder = DATA_LOC / "geomag_data"
    usgs_obs = [obs.upper() for obs in ["bou", "brw", "bsl", "cmo", "ded", "frd", "frn", "gua", "hon", "new", "shu", "sit", "sjg", "tuc"]]
    nrcan_obs = ["ALE", "BLC", "BRD", "CBB", "FCC", "IQA", "MEA", "OTT", "RES", "STJ", "VIC", "YKC"]
    run_function(geomag_folder, usgs_obs, nrcan_obs)
