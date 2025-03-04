"""
Script to calculate return period values for the geomagnetic 
field (e.g., 1-in-100-year). A lognormal distribution is 
fitted to the maxes of B, E, and V. 

Credit to Lucas at el. 2018.

Make sure you have first run the preprocessing scripts. 

Authors:
- Dennies Bor
- Ed Oughton

Date:
- February 2025
"""
import sys
import os
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

from tqdm import tqdm
import h5py
import pandas as pd
import geopandas as gpd
import numpy as np
import scipy.special
import powerlaw

DATA_LOC = Path(__file__).resolve().parent / '..' / "data"


def storm_overlaps(storm_start, storm_end, event_start, event_end=None):
    """Check if a storm overlaps with a given event date range."""
    if event_end is None:
        event_end = event_start + timedelta(days=1)  # Default to 1 day if no end date
    return (storm_start <= event_end) and (storm_end >= event_start)


def get_event_indices(storm_df, event_start, event_end=None):
    """Find storm indices for Halloween, Gannon, and St. Patrick's Day events"""
    return storm_df[
        storm_df.apply(
            lambda row: storm_overlaps(
                row["Start"], row["End"], event_start, event_end
            ),
            axis=1,
        )
    ].index


def extract_max_values(maxB_arr, maxE_arr, maxV_arr, indices):
    """Extract maxB, maxE, and maxV for specific storm indices."""
    return (
        maxB_arr[:, indices],
        maxE_arr[:, indices],
        maxV_arr[:, indices],
    )


def lognormal_ppf(y, mu, sigma, xmin):
    """
    Fit a powerlaw and lognormal distribution to the maxes of B, E, and V.
    Adapted from the original Lucas et al. 2018 paper.
    """
    erf = scipy.special.erf
    erfinv = scipy.special.erfinv
    Q = erf((np.log(xmin) - mu) / (np.sqrt(2) * sigma))
    Q = Q * y - y + 1.0
    Q = erfinv(Q)
    return np.exp(mu + np.sqrt(2) * sigma * Q)


def fit_data(data):
    """

    """
    # Extract the sign of the data (-1 for negative, 1 for positive)
    sign_vector = np.sign(np.mean(np.nan_to_num(data, nan=0)))

    # Fit lognormal to absolute values of the data
    abs_data = np.abs(data)
    
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = powerlaw.Fit(abs_data, xmin=np.min(abs_data), verbose=False)

    fitting_func = fit.lognormal
    if np.any(np.isnan(fitting_func.cdf())):
        warnings.warn("No lognormal fit (changing to positive)")
        fitting_func = fit.lognormal_positive
    
    # Store the sign vector for back-transformation
    fitting_func.sign_vector = sign_vector

    return fitting_func


def calc_return_period(fitting_func, return_period, nyears):
    """Calculate return period"""
    ndata = len(fitting_func.parent_Fit.data)
    if ndata == 0:
        return np.nan

    x_return = 1 - (1 / return_period) * nyears / ndata
    y_return = lognormal_ppf(1 - x_return, fitting_func.mu, fitting_func.sigma, xmin=fitting_func.xmin)

    # Reapply the original sign
    if hasattr(fitting_func, 'sign_vector'):
        sign_vector = fitting_func.sign_vector
        y_return *= sign_vector
    
    # Get rid of inf's
    if not np.all(np.isfinite(y_return)):
        return 0.0
    
    # Make nans zero
    if np.isnan(y_return):
        y_return = 0.0

    return y_return


def safe_fit_and_calc(curr_data, return_periods, n_years):
    """Routine to fit data and calculate return values"""
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fitting_func = fit_data(curr_data)

            results = {}
            for period in return_periods:
                y_return = calc_return_period(fitting_func, period, nyears=n_years)
                if np.any(np.isnan(y_return)) or np.any(np.isinf(y_return)):
                    return None
                results[period] = y_return
            return results
    except Exception as e:
        return None


def process_site(site_data, n_samples, n_years, return_periods, quantity):
    """Process a single site without multiprocessing."""
    results = {period: np.full(n_samples, np.nan) for period in return_periods}
    max_attempts = 100

    # Generate all samples at once
    sampled_data = np.random.choice(site_data, (max_attempts, n_years), replace=True)

    # Normalize data if necessary
    if quantity not in ["B", "V"]:
        sampled_data /= 1000.0

    sampled_data = np.sort(np.nan_to_num(sampled_data), axis=1)

    valid_samples = 0
    for attempt in range(max_attempts):
        if valid_samples >= n_samples:
            break

        sample_results = safe_fit_and_calc(sampled_data[attempt], return_periods, n_years)

        if sample_results:
            for period in return_periods:
                results[period][valid_samples] = sample_results[period]
            valid_samples += 1

    return results


def bootstrap_analysis(maxB_arr, n_samples=100, n_years=39, return_periods=None, quantity="B"):
    """Runs process_site in parallel while avoiding nested multiprocessing issues."""
    
    n_sites = maxB_arr.shape[0]
    
    process_site_partial = partial(
        process_site, n_samples=n_samples, n_years=n_years, return_periods=return_periods, quantity=quantity
    )
    from multiprocessing import get_context
    # Use "spawn" context to avoid issues on Windows
    with get_context("spawn").Pool(processes=cpu_count()) as pool:
        results_list = list(
            tqdm(pool.imap(process_site_partial, maxB_arr), total=n_sites, desc="Processing sites")
        )

    # Convert results to a structured dictionary
    results = {period: np.array([site_result[period] for site_result in results_list])
        for period in return_periods}

    return results


def confidence_limits(results, return_periods):
    confidence_intervals = calculate_confidence_intervals(results)

    for period in return_periods:
        period = int(period)
        lower, median, upper = confidence_intervals[period]

        # Compute mask once
        valid_mask = ~np.isnan(median)
        valid_sites = np.sum(valid_mask)

        if valid_sites > 0:
            median_mean = np.nanmean(median)
            lower_mean = np.nanmean(lower)
            upper_mean = np.nanmean(upper)

    # Site-specific results (vectorized)
    num_sites = median.shape[0]  # Assuming `median` is 1D
    for i in range(num_sites):
        for period in return_periods:
            lower, median, upper = confidence_intervals[period]
            if not np.isnan(median[i]):
                print(f"---{period}-year event: {median[i]:.2f} ({lower[i]:.2f} - {upper[i]:.2f})")
            else:
                print(f"---{period}-year event: No valid data")


def calculate_confidence_intervals(results):
    confidence_intervals = {}
    for period, data in results.items():
        if data.size == 0:
            print(f"No valid data for period {period}")
            confidence_intervals[period] = (np.nan, np.nan, np.nan)
            continue

        lower = np.nanpercentile(data, 5, axis=1)
        median = np.nanpercentile(data, 50, axis=1)
        upper = np.nanpercentile(data, 95, axis=1)

        confidence_intervals[period] = (lower, median, upper)

    return confidence_intervals


def load_or_calculate_results(file_path, max_arr, quantity):
    """Load or calculate results for Magnetic, Electric, and Induced Voltage fields"""
    if os.path.exists(file_path):
        with open(file_path, "rb") as pkl:
            return pickle.load(pkl)  # Works best with NumPy arrays
    else:
        results = bootstrap_analysis(max_arr, n_samples, n_years, return_periods, quantity=quantity)
        with open(file_path, "wb") as pkl:
            pickle.dump(results, pkl, protocol=pickle.HIGHEST_PROTOCOL)
            return results


def extract_confidence_medians(results, return_periods):
    """Calculate statistical predictions for examined return periods"""
    ci = calculate_confidence_intervals(results)
    return {year: ci[year][1] for year in return_periods}


def flatten_max_values(max_arr, indices):
    """Flatten arrays for storm events: Halloween, Gannon, and St. Patrick's Day"""
    return max_arr[:, indices].flatten()


if __name__ == "__main__":

    start_time = time.time()
    include_gannon = True

    storm_data_loc = DATA_LOC / "kp_ap_indices" / "storm_periods.csv"
    emtf_data_path = DATA_LOC / "EMTF" / "mt_pickle.pkl"
    filename = "trans_lines_pickle.pkl"
    transmission_line_path = DATA_LOC / "Electric__Power_Transmission_Lines" / filename

    if os.path.exists(emtf_data_path):
        with open(emtf_data_path, "rb") as pkl:
            MT_sites = pickle.load(pkl)
    else:
        raise FileNotFoundError(f"Can't find file: {emtf_data_path}")

    mt_sites_names = [site.name for site in MT_sites]
    mt_sites_coords = np.array([(site.latitude, site.longitude) for site in MT_sites])

    if os.path.exists(transmission_line_path):
        with open(transmission_line_path, "rb") as pkl:
            trans_lines_gdf = pickle.load(pkl)
        line_ids = np.int32(trans_lines_gdf.line_id.to_numpy())
    else:
        raise FileNotFoundError(f"Can't find file: {transmission_line_path}")

    storm_df = pd.read_csv(storm_data_loc)
    storm_df["Start"] = pd.to_datetime(storm_df["Start"])
    storm_df["End"] = pd.to_datetime(storm_df["End"])
    # # filter from 1985 to 2015
    storm_df = storm_df[(storm_df["Start"] >= "1985-01-01") & (storm_df["End"] <= "2016-01-01")]

    # Load the maxes as numpy arrays
    maxB_arr = np.load(DATA_LOC / "maxB_arr_testing_2.npy")
    maxE_arr = np.load(DATA_LOC / "maxE_arr_testing_2.npy")
    maxV_arr = np.load(DATA_LOC / "maxV_arr_testing_2.npy")

    # Define key storm events
    n_years = 39  # Number of years from 1985 to 2024

    # Define event date ranges
    halloween_start = datetime(2003, 10, 29, 0)
    halloween_end = datetime(2003, 11, 3, 0)

    gannon_start = datetime(2024, 5, 11, 0)  # Mother's Day storm in 2024
    gannon_end = gannon_start + timedelta(days=1)  # 1 day duration

    st_patricks_start = datetime(2015, 3, 17, 0)  # St. Patrick's Day storm 2015
    st_patricks_end = st_patricks_start + timedelta(days=1)  # 1 day duration

    hydro_quebec_start = datetime(1989, 3, 13, 0)  # Hydro-Quebec storm 1989   
    hydro_quebec_end = hydro_quebec_start + timedelta(days=1)  # 1 day duration

    halloween_idx = get_event_indices(storm_df, halloween_start, halloween_end)
    gannon_idx = get_event_indices(storm_df, gannon_start, gannon_end)
    st_patricks_idx = get_event_indices(storm_df, st_patricks_start, st_patricks_end)

    maxB_halloween, maxE_halloween, maxV_halloween = extract_max_values(maxB_arr, maxE_arr, maxV_arr, halloween_idx)
    maxB_gannon, maxE_gannon, maxV_gannon = extract_max_values(maxB_arr, maxE_arr, maxV_arr, gannon_idx)
    maxB_st_patricks, maxE_st_patricks, maxV_st_patricks = extract_max_values(maxB_arr, maxE_arr, maxV_arr, st_patricks_idx)
    maxB_hydro_quebec, maxE_hydro_quebec, maxV_hydro_quebec = extract_max_values(maxB_arr, maxE_arr, maxV_arr, get_event_indices(storm_df, hydro_quebec_start, hydro_quebec_end))

    # Run the analysis
    n_samples = 100
    return_periods = np.arange(25, 1001, 25)
    dir_path = DATA_LOC / "statistical_analysis"
    dir_path.mkdir(parents=True, exist_ok=True)

    resuls_B_path = dir_path / "results_B_return_periods.pkl"
    resuls_E_path = dir_path / "results_E_return_periods.pkl"
    resuls_V_path = dir_path / "results_V_return_periods.pkl"

    results_B = load_or_calculate_results(resuls_B_path, maxB_arr, "B")
    results_E = load_or_calculate_results(resuls_E_path, maxE_arr, "E")
    results_V = load_or_calculate_results(resuls_V_path, maxV_arr, "V")

    confidence_limits(results_B, return_periods)
    confidence_limits(results_E, return_periods)
    confidence_limits(results_V, return_periods)

    results_B_ci = extract_confidence_medians(results_B,  return_periods)
    results_E_ci = extract_confidence_medians(results_E, return_periods)
    results_V_ci = extract_confidence_medians(results_V, return_periods)

    maxB_halloween = flatten_max_values(maxB_arr, halloween_idx)
    maxE_halloween = flatten_max_values(maxE_arr, halloween_idx)
    maxV_halloween = flatten_max_values(maxV_arr, halloween_idx)

    maxB_gannon = flatten_max_values(maxB_arr, gannon_idx)
    maxE_gannon = flatten_max_values(maxE_arr, gannon_idx)
    maxV_gannon = flatten_max_values(maxV_arr, gannon_idx)

    maxB_st_patricks = flatten_max_values(maxB_arr, st_patricks_idx)
    maxE_st_patricks = flatten_max_values(maxE_arr, st_patricks_idx)
    maxV_st_patricks = flatten_max_values(maxV_arr, st_patricks_idx)

    # Save data into an HDF5 file
    path_out = DATA_LOC / "statistical_analysis" / "geomagnetic_data_return_periods.h5"
    with h5py.File(path_out, "w") as f:
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
            for year in return_periods:
                data = locals()[f"results_{field}_ci"][year]
                field_group.create_dataset(f"{year}_year", data=data)

        # Add metadata
        f.attrs["description"] = ("Geomagnetic data including MT sites " +
            "transmission lines, real storm events, and statistical predictions")
        f.attrs["date_created"] = np.bytes_(datetime.now().isoformat())

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")