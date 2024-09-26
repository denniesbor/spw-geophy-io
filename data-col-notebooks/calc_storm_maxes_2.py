# ---------------------------------------------------------------
# This script calculates the 100-year, 250-year, 500-year, and 1000-year return values for the geomagnetic field
# based on the storm data from 1985 to 2024. This script fits a power law and lognormal distribution to the maxes of B, E, and V.
# Some of the code is adapted from the original code by Greg 2018 Hazard paper.
# Author: Dennies Bor
# ---------------------------------------------------------------

# %%
import sys
import os
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
import logging

from tqdm import tqdm
import h5py
import pandas as pd
import geopandas as gpd
import numpy as np
import scipy.special
import powerlaw


# Re use loggin from calc storm maxes
# Set up logging, a reusable function
def setup_logging():
    """
    Set up logging to both file and console.
    """
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler("Get 100-year hazards.log")
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


# Logging
setup_logging()

# ---------------------------------------------------------------
# Set up the data directories
# ---------------------------------------------------------------
data_dir = Path(__file__).resolve().parent / "data"
storm_data_loc = data_dir / "kp_ap_indices" / "storm_periods.csv"
emtf_data_path = data_dir / "EMTF" / "emtf_data.pkl"
transmission_line_path = (
    data_dir / "Electric__Power_Transmission_Lines" / "trans_lines_pickle.pkl"
)

# ---------------------------------------------------------------
# Read the MT sites pickle file
# ---------------------------------------------------------------
if os.path.exists(emtf_data_path):
    with open(emtf_data_path, "rb") as pkl:
        MT_sites = pickle.load(pkl)
    logger.info("MT sites loaded")
else:
    logger.error("Run the calc_storm_maxes.py file first to prepare the data")
    raise FileNotFoundError(
        "Run the calc_storm_maxes.py file first to prepare the data"
    )

# Prepare MT Sites information
mt_sites_names = [site.name for site in MT_sites]
mt_sites_coords = np.array([(site.latitude, site.longitude) for site in MT_sites])

# ---------------------------------------------------------------
# Read the transmission lines
# ---------------------------------------------------------------
if os.path.exists(transmission_line_path):
    with open(transmission_line_path, "rb") as pkl:
        trans_lines_gdf = pickle.load(pkl)
    logger.info("Transmission lines loaded")
else:
    logger.error("Run the calc_storm_maxes.py file first to prepare the data")
    raise FileNotFoundError(
        "Run the calc_storm_maxes.py file first to prepare the data"
    )

line_ids = np.int32(trans_lines_gdf.line_id.to_numpy())

# ---------------------------------------------------------------
# Read the storm times data
# ---------------------------------------------------------------
storm_df = pd.read_csv(storm_data_loc)
storm_df["Start"] = pd.to_datetime(storm_df["Start"])
storm_df["End"] = pd.to_datetime(storm_df["End"])

# ---------------------------------------------------------------
# Load the maxes as numpy arrays
# ---------------------------------------------------------------
maxB_arr = np.load(data_dir / "maxB_arr.npy")
maxE_arr = np.load(data_dir / "maxE_arr.npy")
maxV_arr = np.load(data_dir / "maxV_arr.npy")

# ---------------------------------------------------------------
# Define key storm events
# ---------------------------------------------------------------
n_years = 39  # Number of years from 1985 to 2024

# Define event date ranges
halloween_start = datetime(2003, 10, 29, 0)
halloween_end = datetime(2003, 11, 3, 0)

gannon_start = datetime(2024, 5, 11, 0)  # Mother's Day storm in 2024
gannon_end = gannon_start + timedelta(days=1)  # 1 day duration

st_patricks_start = datetime(2015, 3, 17, 0)  # St. Patrick's Day storm 2015
st_patricks_end = st_patricks_start + timedelta(days=1)  # 1 day duration


# ---------------------------------------------------------------
# Helper function to check if a storm overlaps with an event
# ---------------------------------------------------------------
def storm_overlaps(storm_start, storm_end, event_start, event_end=None):
    """Check if a storm overlaps with a given event date range."""
    if event_end is None:
        event_end = event_start + timedelta(days=1)  # Default to 1 day if no end date
    return (storm_start <= event_end) and (storm_end >= event_start)


# ---------------------------------------------------------------
# Find storm indices for Halloween, Gannon, and St. Patrick's Day events
# ---------------------------------------------------------------
def get_event_indices(storm_df, event_start, event_end=None):
    return storm_df[
        storm_df.apply(
            lambda row: storm_overlaps(
                row["Start"], row["End"], event_start, event_end
            ),
            axis=1,
        )
    ].index


halloween_idx = get_event_indices(storm_df, halloween_start, halloween_end)
gannon_idx = get_event_indices(storm_df, gannon_start, gannon_end)
st_patricks_idx = get_event_indices(storm_df, st_patricks_start, st_patricks_end)


# ---------------------------------------------------------------
# Extract the maxes for the Halloween, Gannon, and St. Patrick's Day storms
# ---------------------------------------------------------------
def extract_max_values(maxB_arr, maxE_arr, maxV_arr, indices):
    """Extract maxB, maxE, and maxV for specific storm indices."""
    return (
        maxB_arr[:, indices],
        maxE_arr[:, indices],
        maxV_arr[:, indices],
    )


maxB_halloween, maxE_halloween, maxV_halloween = extract_max_values(
    maxB_arr, maxE_arr, maxV_arr, halloween_idx
)
maxB_gannon, maxE_gannon, maxV_gannon = extract_max_values(
    maxB_arr, maxE_arr, maxV_arr, gannon_idx
)
maxB_st_patricks, maxE_st_patricks, maxV_st_patricks = extract_max_values(
    maxB_arr, maxE_arr, maxV_arr, st_patricks_idx
)

# %%
# ---------------------------------------------------------------
# Fit a powerlaw and lognormal distribution to the maxes of B, E, and V
# These routines are adapted from the original code by Greg Lucas Hazard paper
# ---------------------------------------------------------------
erf = scipy.special.erf
erfinv = scipy.special.erfinv


def lognormal_ppf(y, mu, sigma, xmin):
    Q = erf((np.log(xmin) - mu) / (np.sqrt(2) * sigma))
    Q = Q * y - y + 1.0
    Q = erfinv(Q)
    return np.exp(mu + np.sqrt(2) * sigma * Q)


def fit_data(data):
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = powerlaw.Fit(data, xmin=np.min(data), verbose=False)

    fitting_func = fit.lognormal
    if np.any(np.isnan(fitting_func.cdf())):
        warnings.warn("No lognormal fit (changing to positive)")
        fitting_func = fit.lognormal_positive

    return fitting_func


def calc_return_period(fitting_func, return_period, nyears=n_years):
    ndata = len(fitting_func.parent_Fit.data)
    if ndata == 0:
        return np.nan

    x_return = 1 - (1 / return_period) * nyears / ndata
    y_return = lognormal_ppf(
        1 - x_return, fitting_func.mu, fitting_func.sigma, xmin=fitting_func.xmin
    )
    # Get rid of inf's
    if ~np.isfinite(y_return):
        return np.nan

    return y_return


# ---------------------------------------------------------------
# Routine to fit data and calculate return values
# ---------------------------------------------------------------
def safe_fit_and_calc(curr_data, return_periods, n_years):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fitting_func = fit_data(curr_data)

            if len(w) > 0:
                logger.warning(
                    f"Warnings during fitting: {[str(warn.message) for warn in w]}"
                )

            results = {}
            for period in return_periods:
                y_return = calc_return_period(fitting_func, period, nyears=n_years)
                if np.isnan(y_return) or np.isinf(y_return):
                    logger.warning(
                        f"Invalid return value for period {period}: {y_return}"
                    )
                    return None
                results[period] = y_return
            return results
    except Exception as e:
        logger.error(f"Error in fitting or calculation: {str(e)}")
        return None


def process_site(site_data, n_samples, n_years, return_periods, quantity):
    results = {period: np.full(n_samples, np.nan) for period in return_periods}
    valid_samples = 0
    attempts = 0
    max_attempts = 100

    while valid_samples < n_samples and attempts < max_attempts:
        if quantity == "B" or quantity == "V":
            curr_data = np.sort(np.random.choice(site_data, n_years, replace=True))
        else:
            curr_data = (
                np.sort(np.random.choice(site_data, n_years, replace=True)) / 1000.0
            )

        if np.all(np.isnan(curr_data)):
            break

        curr_data = np.nan_to_num(curr_data)

        sample_results = safe_fit_and_calc(curr_data, return_periods, n_years)

        if sample_results is not None:
            for period in return_periods:
                results[period][valid_samples] = sample_results[period]
            valid_samples += 1

        attempts += 1

    if valid_samples < n_samples:
        logger.warning(
            f"Only {valid_samples}/{n_samples} valid samples obtained after {attempts} attempts"
        )

    return results


def bootstrap_analysis(
    maxB_arr, n_samples=100, n_years=39, return_periods=[100, 500, 1000], quantity="B"
):
    n_sites = maxB_arr.shape[0]

    # Partial function to fix all arguments except site_data
    process_site_partial = partial(
        process_site,
        n_samples=n_samples,
        n_years=n_years,
        return_periods=return_periods,
        quantity=quantity,
    )

    # Use multiprocessing to process sites in parallel
    with Pool(processes=cpu_count()) as pool:
        results_list = list(
            tqdm(
                pool.imap(process_site_partial, maxB_arr),
                total=n_sites,
                desc="Processing sites",
            )
        )

    # Combine results from all sites
    results = {
        period: np.array([site_result[period] for site_result in results_list])
        for period in return_periods
    }

    return results


def calculate_confidence_intervals(results):
    confidence_intervals = {}
    for period, data in results.items():
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            lower = np.nanpercentile(data, 5, axis=1)
            median = np.nanpercentile(data, 50, axis=1)
            upper = np.nanpercentile(data, 95, axis=1)
            confidence_intervals[period] = (lower, median, upper)
        else:
            logger.warning(f"No valid data for period {period}")
            confidence_intervals[period] = (np.nan, np.nan, np.nan)
    return confidence_intervals


def confidence_limits(results, return_periods):

    # Calculate confidence intervals
    confidence_intervals = calculate_confidence_intervals(results)
    for period in return_periods:
        lower, median, upper = confidence_intervals[period]
        valid_sites = np.sum(~np.isnan(median))
        logging.info(f"{period}-year event (valid sites: {valid_sites}/{len(median)}):")
        logging.info(f"  Median: {np.nanmean(median):.2f}")
        logging.info(f"  95% CI: ({np.nanmean(lower):.2f} - {np.nanmean(upper):.2f})")

    # If you want site-specific results
    for i in range(maxB_arr.shape[0]):
        logging.info(f"Site {i+1}:")
        for period in return_periods:
            lower, median, upper = confidence_intervals[period]
            if not np.isnan(median[i]):
                logging.info(
                    f"  {period}-year event: {median[i]:.2f} ({lower[i]:.2f} - {upper[i]:.2f})"
                )
            else:
                logging.info(f"  {period}-year event: No valid data")


# ---------------------------------------------------------------
# Run the analysis
# ---------------------------------------------------------------
n_samples = 100
return_periods = [100, 250, 500, 1000]
data_dir = Path("__file__").resolve().parent / "data"
resuls_B_path = data_dir / "results_B.pkl"
resuls_E_path = data_dir / "results_E.pkl"
resuls_V_path = data_dir / "results_V.pkl"


# ---------------------------------------------------------------
# Load or calculate results for Magnetic, Electric, and Induced Voltage fields
# ---------------------------------------------------------------
def load_or_calculate_results(file_path, max_arr, quantity):
    if os.path.exists(file_path):
        logging.info(f"{quantity} maxes available")
        with open(file_path, "rb") as pkl:
            return pickle.load(pkl)
    else:
        results = bootstrap_analysis(
            max_arr, n_samples, n_years, return_periods, quantity=quantity
        )
        with open(file_path, "wb") as pkl:
            pickle.dump(results, pkl)
        return results


results_B = load_or_calculate_results(resuls_B_path, maxB_arr, "B")
results_E = load_or_calculate_results(resuls_E_path, maxE_arr, "E")
results_V = load_or_calculate_results(resuls_V_path, maxV_arr, "V")


# ---------------------------------------------------------------
# Calculate confidence intervals
# ---------------------------------------------------------------
confidence_limits(results_B, return_periods)
confidence_limits(results_E, return_periods)
confidence_limits(results_V, return_periods)


# ---------------------------------------------------------------
# Calculate statistical predictions for 50/100/500/1000-year events
# ---------------------------------------------------------------
def extract_confidence_medians(results):
    ci = calculate_confidence_intervals(results)
    return {year: ci[year][1] for year in [100, 250, 500, 1000]}


results_B_ci = extract_confidence_medians(results_B)
results_E_ci = extract_confidence_medians(results_E)
results_V_ci = extract_confidence_medians(results_V)


# ---------------------------------------------------------------
# Flatten arrays for storm events: Halloween, Gannon, and St. Patrick's Day
# ---------------------------------------------------------------
def flatten_max_values(max_arr, indices):
    return max_arr[:, indices].flatten()


maxB_halloween = flatten_max_values(maxB_arr, halloween_idx)
maxE_halloween = flatten_max_values(maxE_arr, halloween_idx)
maxV_halloween = flatten_max_values(maxV_arr, halloween_idx)

maxB_gannon = flatten_max_values(maxB_arr, gannon_idx)
maxE_gannon = flatten_max_values(maxE_arr, gannon_idx)
maxV_gannon = flatten_max_values(maxV_arr, gannon_idx)

maxB_st_patricks = flatten_max_values(maxB_arr, st_patricks_idx)
maxE_st_patricks = flatten_max_values(maxE_arr, st_patricks_idx)
maxV_st_patricks = flatten_max_values(maxV_arr, st_patricks_idx)


# ---------------------------------------------------------------
# Save data into an HDF5 file
# ---------------------------------------------------------------
with h5py.File(data_dir / "geomagnetic_data.h5", "w") as f:
    # Create main groups
    sites = f.create_group("sites")
    events = f.create_group("events")
    predictions = f.create_group("predictions")

    # Store MT site data
    mt_sites = sites.create_group("mt_sites")
    mt_sites.create_dataset("names", data=np.array(mt_sites_names, dtype="S"))
    mt_sites.create_dataset("coordinates", data=mt_sites_coords)

    # Store transmission line data
    transmission_lines = sites.create_group("transmission_lines")
    transmission_lines.create_dataset("line_ids", data=line_ids)

    # Store real storm data
    for storm in ["halloween", "gannon", "st_patricks"]:
        storm_group = events.create_group(storm)
        storm_group.create_dataset("E", data=locals()[f"maxE_{storm}"])
        storm_group.create_dataset("B", data=locals()[f"maxB_{storm}"])
        storm_group.create_dataset("V", data=locals()[f"maxV_{storm}"])

    # Store statistical predictions
    for field in ["E", "B", "V"]:
        field_group = predictions.create_group(field)
        for year in [250, 100, 500, 1000]:
            data = locals()[f"results_{field}_ci"][year]
            field_group.create_dataset(f"{year}_year", data=data)

    # Add metadata
    f.attrs["description"] = (
        "Geomagnetic data including MT sites, transmission lines, real storm events, and statistical predictions"
    )
    f.attrs["date_created"] = np.bytes_(datetime.now().isoformat())

logging.info("Data saved to geomagnetic_data.h5")
