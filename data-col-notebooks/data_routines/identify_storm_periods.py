# %%
import os
import logging
import pandas as pd
import requests
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Set up working directories
data_loc = Path(__file__).resolve().parent.parent / "data"
kp_dst_path = data_loc / "kp_ap_indices"
kp_dst_path.mkdir(parents=True, exist_ok=True)

print("Working directory:", data_loc)


# %%
def download_file(url, file_path):
    """Download a file from a given URL and save it to the specified path."""
    response = requests.get(url)
    response.raise_for_status()
    with file_path.open("wb") as f:
        f.write(response.content)
    print(f"File downloaded and saved to {file_path}")


def parse_combined_kp_ap_file(file_path):
    columns = [
        "Year",
        "Month",
        "Day",
        "Hour",
        "FracHour",
        "DecimalDate1",
        "DecimalDate2",
        "Kp",
        "Ap",
        "Flag",
    ]
    kp_df = pd.read_csv(file_path, sep="\s+", header=None, names=columns)
    kp_df["Kp_0to9"] = (kp_df["Kp"] * 3).round().astype(int)
    kp_df["Datetime"] = pd.to_datetime(kp_df[["Year", "Month", "Day", "Hour"]])
    return kp_df


def analyze_kp_ap_data(df):
    print("Data range:", df["Datetime"].min(), "to", df["Datetime"].max())
    print("\nBasic statistics:")
    print(df[["Kp", "Kp_0to9", "Ap"]].describe())
    print("\nDates with highest Ap index:")
    print(df.nlargest(10, "Ap")[["Datetime", "Ap", "Kp"]])
    print("\nDistribution of Kp values (0-9 scale):")
    print(df["Kp_0to9"].value_counts().sort_index())


def parse_dst_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    data = []
    for line in lines:
        if "|" in line:
            continue
        parts = line.split()
        if len(parts) == 4:
            datetime = parts[0] + " " + parts[1]
            doy, dst = parts[2], parts[3]
            data.append([datetime, doy, dst])

    return pd.DataFrame(data, columns=["Datetime", "DOY", "DST"])


def process_dst_data(df):
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["DOY"] = df["DOY"].astype(int)
    df["DST"] = df["DST"].astype(float)
    df = df[df["DST"] != 99999.99]
    df.set_index("Datetime", inplace=True)
    return df


# Identify storm periods
def identify_storms(
    dst_df, df_kp, dst_threshold=-150, kp_threshold=8, time_delta_days=1
):
    """
    Identify storm periods from 1980 to 2024 using Kyoto DST and Postdam GFZ Dst values.

    Parameters:
    - dst_df (DataFrame): DataFrame containing Kyoto DST values.
    - df_kp (DataFrame): DataFrame containing Postdam GFZ Dst values.
    - dst_threshold (float, optional): Threshold value for Kyoto DST. Defaults to -150 nt.
    - kp_threshold (float, optional): Threshold value for Postdam GFZ Dst. Defaults to 8.
    - time_delta_days (int, optional): Number of days to consider for storm duration. Defaults to 1.

    Returns:
    - storm_df (DataFrame): DataFrame containing storm information, including start, end, and duration of each storm period.
    """
    # Function code here
    ...
    # Ensure datetime index
    dst_df = dst_df.set_index(pd.to_datetime(dst_df.index))
    df_kp = df_kp.set_index(pd.to_datetime(df_kp.index))

    # Identify potential storm periods
    dst_storms = dst_df[dst_df["DST"] <= dst_threshold].index
    kp_storms = df_kp[df_kp["Kp"] >= kp_threshold].index

    # Combine storm timestamps
    all_storm_times = sorted(set(dst_storms) | set(kp_storms))

    # Initialize storm tracking
    storm_periods = []
    current_storm_start = None
    current_storm_end = None

    for time in all_storm_times:
        if current_storm_start is None:
            current_storm_start = time
            current_storm_end = time
        elif (time - current_storm_end) <= timedelta(days=time_delta_days):
            current_storm_end = time
        else:
            storm_periods.append((current_storm_start, current_storm_end))
            current_storm_start = time
            current_storm_end = time

    if current_storm_start is not None:
        storm_periods.append((current_storm_start, current_storm_end))

    # Merge overlapping or close storm periods
    merged_storm_periods = []
    for start, end in storm_periods:
        if not merged_storm_periods or start - merged_storm_periods[-1][1] > timedelta(
            days=time_delta_days * 2
        ):
            merged_storm_periods.append([start, end])
        else:
            merged_storm_periods[-1][1] = max(merged_storm_periods[-1][1], end)

    # Extend storm periods by time_delta
    extended_storm_periods = []
    for start, end in merged_storm_periods:
        extended_start = start - timedelta(days=time_delta_days)
        extended_end = end + timedelta(days=time_delta_days)
        extended_storm_periods.append((extended_start, extended_end))

    # Create a DataFrame with storm information
    storm_df = pd.DataFrame(extended_storm_periods, columns=["Start", "End"])
    storm_df["Duration"] = storm_df["End"] - storm_df["Start"]

    return storm_df


# Visualize storm periods
def visualize_storm_periods(dst_df, kp_df):
    """The function to visualize storm periods using DST and Kp indices."""

    # Create the figure and GridSpec
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.1)

    # DST plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dst_df.index, dst_df["DST"], color="#1f77b4", label="DST")
    ax1.set_ylabel("DST Index (nT)")
    ax1.set_title(
        "Geomagnetic Activity: DST and Kp Indices", fontsize=16, fontweight="bold"
    )
    ax1.legend(loc="upper right")

    # Kp plot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(kp_df.index, kp_df["Kp"], color="#ff7f0e", label="Kp")
    ax2.set_ylabel("Kp Index")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right")

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    # Add gridlines
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Adjust y-axis for Kp plot to show 0-9 range
    ax2.set_ylim(0, 9)
    ax2.set_yticks(range(0, 10))

    # Add horizontal lines for storm thresholds
    ax1.axhline(y=-50, color="r", linestyle="--", alpha=0.7)
    ax1.text(
        dst_df.index.max(),
        -50,
        "Moderate Storm",
        va="bottom",
        ha="right",
        color="r",
        alpha=0.7,
    )

    ax2.axhline(y=5, color="r", linestyle="--", alpha=0.7)
    ax2.text(
        kp_df.index.max(), 5, "G1 Storm", va="bottom", ha="right", color="r", alpha=0.7
    )

    # Adjust layout and save
    plt.tight_layout()
    fig.savefig(kp_dst_path / "storm_periods.png", bbox_inches="tight", dpi=150)


def main():
    # Download and process KP data
    kp_url = "https://kp.gfz-potsdam.de/kpdata?startdate=1980-01-01&enddate=2024-09-16&format=kp2#kpdatadownload-143"
    kp_file_path = kp_dst_path / "kpdata.txt"
    download_file(kp_url, kp_file_path)

    kp_df = parse_combined_kp_ap_file(kp_file_path)
    analyze_kp_ap_data(kp_df)
    print("\nKP Analysis complete. Check the current directory for output plots.")

    # The DST data has been downloaded and manually saved to a file named 'dst_data.txt'.
    # The data is downloaded from Kyoto World Data Center for Geomagnetism.
    # Couldn't script the download as the server restricts to 25 years of data.
    # Had to partially download the files and merge them manually as .txt
    dst_file_path = kp_dst_path / "dst_data.txt"
    dst_df = parse_dst_file(dst_file_path)
    dst_df = process_dst_data(dst_df)

    # Set Datetime as index for KP data
    kp_df.set_index("Datetime", inplace=True)

    print("\nData processing complete.")
    print("KP data shape:", kp_df.shape)
    print("DST data shape:", dst_df.shape)

    # You can add further analysis or plotting here
    storm_df = identify_storms(dst_df, kp_df)

    # Export this data for use in later stages
    # Save storm whose datetime is 1985 >
    storm_df = storm_df[storm_df["Start"] > pd.to_datetime("1985-01-01")]
    storm_df.to_csv((kp_dst_path / "storm_periods.csv"), index=False)

    # Visualize the storm periods
    visualize_storm_periods(dst_df, kp_df)


if __name__ == "__main__":
    main()
