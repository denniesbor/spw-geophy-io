# %%
import os
import logging
from pathlib import Path
from multiprocessing import Pool
from functools import partial, reduce
from itertools import chain
from pathlib import Path
import collections
import xarray as xr
import numpy as np
import pandas as pd
import os
import logging
from multiprocessing import Pool
import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import bezpy.mag
from scipy import signal
import bezpy

# %%

data_dir = Path("__file__").resolve().parent.parent / "data"
geomag_folder = data_dir / "geomag_data"

usgs_obs = [
    obs.upper()
    for obs in [
        "bou",
        "brw",
        "bsl",
        "cmo",
        "ded",
        "frd",
        "frn",
        "gua",
        "hon",
        "new",
        "shu",
        "sit",
        "sjg",
        "tuc",
    ]
]
nrcan_obs = [
    "ALE",
    "BLC",
    "BRD",
    "CBB",
    "FCC",
    "IQA",
    "MEA",
    "OTT",
    "RES",
    "STJ",
    "VIC",
    "YKC",
]

# All obs
all_obs = usgs_obs + nrcan_obs


# %%
def process_magnetic_files(file_path, is_processed=False):

    if is_processed:
        data = pd.read_csv(file_path, index_col=0)
        data.index = pd.to_datetime(data.index)

        Latitude = data["Latitude"].iloc[0]
        Longitude = data["Longitude"].iloc[0]

        # Get from the directory name (parent of the file)
        parent_dir = Path(file_path).parent

        iaga_code = parent_dir.name

        ds = xr.Dataset.from_dataframe(data)

        ds.attrs["Latitude"] = Latitude
        ds.attrs["Longitude"] = Longitude
        ds.attrs["Name"] = iaga_code

        return ds, data

    data, headers = bezpy.mag.read_iaga(file_path, return_header=True)
    # Rename index of the data to Timestamp
    data.index.name = "Timestamp"

    # Fill NaNs at the start and end, then interpolate remaining NaNs
    for component in ["X", "Y", "Z"]:

        data[component] = (
            data[component]
            .interpolate(method="nearest")
            .fillna(method="bfill")
            .fillna(method="ffill")
        )

        # Detrend data linearly
        data[component] = signal.detrend(data[component])

    # Convert the DataFrame to an xarray Dataset
    ds = xr.Dataset.from_dataframe(data)
    Latitude = float(headers["geodetic latitude"])
    Longitude = float(headers["geodetic longitude"]) - 360

    # Assign latitude and longitude as dataset attributes under the desired names
    ds.attrs["Latitude"] = Latitude
    ds.attrs["Longitude"] = Longitude

    data["Latitude"] = Latitude
    data["Longitude"] = Longitude

    # Clean up specific metadata fields
    ds.attrs["Name"] = headers["iaga code"]

    # Add metadata as global attributes to the dataset
    ds.attrs.update(headers)

    # Remove original keys / Eliminate duplication
    del ds.attrs["geodetic latitude"]
    del ds.attrs["geodetic longitude"]
    del ds.attrs["iaga code"]

    return ds, data


# %%


def process_directory(dir_path):
    logging.info(f"Processing {dir_path}")

    def process_file(filename):
        if filename.endswith(".min") or filename.endswith(".csv"):
            file_path = os.path.join(dir_path, filename)
            try:
                if filename.endswith(".min"):
                    ds, result = process_magnetic_files(file_path, is_processed=False)
                else:
                    ds, result = process_magnetic_files(file_path, is_processed=True)
                result["site_id"] = os.path.basename(dir_path)
                return ds, result
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {str(e)}")
                return None, None
        return None, None

    try:
        results = list(
            filter(
                lambda x: x[0] is not None,
                map(process_file, sorted(os.listdir(dir_path))),
            )
        )
        datasets, file_results = zip(*results) if results else ([], [])

        if datasets:
            combined_dataset = xr.concat(datasets, dim="Timestamp")
            return (os.path.basename(dir_path), combined_dataset, list(file_results))
        else:
            logging.warning(f"No valid datasets found in {dir_path}")
            return (os.path.basename(dir_path), None, None)
    except Exception as e:
        logging.error(f"Error processing directory {dir_path}: {str(e)}")
        return (os.path.basename(dir_path), None, None)


def process_all_directories(geomag_folder, usgs_obs, nrcan_obs):
    all_observatory_dirs = [
        os.path.join(root, d)
        for root, dirs, _ in os.walk(geomag_folder)
        for d in dirs
        if d.upper() in usgs_obs + nrcan_obs
    ]

    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_directory, all_observatory_dirs)

    obsv_xarrays = {
        dir: dataset
        for dir, dataset, _ in results
        if dataset is not None and not np.isnan(dataset.X.min().values)
    }

    return results, obsv_xarrays


def combine_results(results, obsv_xarrays):
    return list(
        chain.from_iterable(
            result_list
            for _, _, result_list in results
            if result_list and result_list[0].site_id.iloc[0] in obsv_xarrays
        )
    )


# Get mode
def get_mode(series):
    mode_result = stats.mode(series)

    return mode_result.mode if mode_result.mode.size > 0 else np.nan


def prepare_dataset(combined_df):
    combined_df = combined_df.sort_values(by=["site_id", "Timestamp"])

    # Get unique longitudes, latitudes, time steps, and site_ids
    unique_lat_lon = (
        combined_df.groupby("site_id")
        .agg({"Longitude": get_mode, "Latitude": get_mode})
        .reset_index()
    )

    time_steps = combined_df.index.unique()
    time_steps = sorted(list(set(pd.to_datetime(time_steps).to_list())))

    site_ids = unique_lat_lon["site_id"].unique()

    ntimes, nsites = len(time_steps), len(site_ids)

    pivot = (
        lambda col: combined_df.pivot_table(
            index=combined_df.index, columns="site_id", values=col, aggfunc="first"
        )
        .reindex(time_steps)
        .values
    )

    dB = np.stack([pivot("X"), pivot("Y"), pivot("Z")], axis=-1)

    ds = xr.Dataset(
        coords={
            "longitude": ("site", unique_lat_lon["Longitude"].values),
            "latitude": ("site", unique_lat_lon["Latitude"].values),
            "site": unique_lat_lon["site_id"].values,
            "component": ["X", "Y", "Z"],
            "time": pd.to_datetime(time_steps).to_list(),
        },
        data_vars={"B": (("time", "site", "component"), dB)},
    )

    # Start at 1985-01-01
    start = pd.to_datetime("1985-01-01")

    indices = np.where(ds.time == start)[0]

    print(f"The start {start} occurs at the following indices:")
    print(indices)

    # If you want to see the full start values at these indices:
    print("\nFull start values at these indices:")
    print(ds.time[indices].values)

    # Add these lines to check for uniqueness
    print("\nChecking uniqueness of time steps:")
    print(f"Number of time steps: {len(time_steps)}")
    print(f"Number of unique time steps: {len(set(time_steps))}")

    if len(time_steps) != len(set(time_steps)):
        print("Warning: Time steps are not unique!")
        duplicates = [
            item for item, count in collections.Counter(time_steps).items() if count > 1
        ]
        print(f"Duplicate time steps: {duplicates}")

    return ds


def main(geomag_folder, usgs_obs, nrcan_obs):
    results, obsv_xarrays = process_all_directories(geomag_folder, usgs_obs, nrcan_obs)
    logging.info(f"Processed {len(obsv_xarrays)} valid observatories")
    all_results = combine_results(results, obsv_xarrays)
    combined_df = pd.concat(all_results)

    # Save combinbed df
    combined_df.to_csv("combined_df.csv")

    logging.info(f"Combined {len(combined_df)} observatories")

    ds = prepare_dataset(combined_df)
    print(ds)
    ds.to_netcdf(data_dir / "processed_geomag_data.nc")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(geomag_folder, usgs_obs, nrcan_obs)
