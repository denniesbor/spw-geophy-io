# Data Download and Preprocessing

I have provided routines and scripts to download data from USGS and NRCAN FTP servers.

## USGS Geomagnetic Data

USGS data, accessible from [USGS Geomag](https://geomag.usgs.gov/ws/docs), is straightforward to obtain, especially for data before 1989.

## NRCAN Geomagnetic Data

NRCAN data is shared in mseed format through the FDSNWS network. Information on the channels, location, and network can be accessed at [NRCAN Geomag](https://geomag.nrcan.gc.ca/data-donnee/sd-en.php).

We use the `obspy` package to read the mseed files and export them into CSV files for preprocessing. Some requests return 404 or 204 errors for NRCAN data, even though the data is available. Therefore, you may need to run the scripts repeatedly to obtain all datasets. You can set up a cron task to run the fetch script periodically until you have acquired all the necessary data.

## Data from 1991 to Present

For data from 1991 to the present, we primarily use data accessible from Intermagnet. Use the script `download_geomag_data.py` to download geomagnetic data for NRCAN and US observatories.

## Electromagnetic Transfer Functions (EMTF)

We have used data from the entire contiguous United States from the EarthScope USArray and the latest USMT arrays. The XML files can be downloaded via the browser interface from [IRIS EMTF Data](https://ds.iris.edu/spud/emtf).

## Transmission Lines

We have used open data available for US transmission lines, accessible from [Transmission Lines Dataset](https://hifld-geoplatform.hub.arcgis.com/datasets/geoplatform::transmission-lines/about). We focused on extra-high voltage transmission lines, filtering out those with voltage ratings below 200 kV. 

## Substations Data

The substations data is queried from OpenStreetMap. The script to fetch this data is provided separately.

## Geomagnetic Indices
The Dst data is manually downloaded from Kyoto WDC from 1985 to 2024. This data is manually downloaded and saved into a text file for loading while running the `identify_storm_periods.py`. Moreover, the Kp data is automatically queried by the script from [GFZ Potsdam](https://kp.gfz-potsdam.de/kpdata?startdate=1980-01-01).

## Automated Shell scripts
Set up a cron job to repeatedly download the files by running `download_nrcan_old.sh`. I have manually looped and set a timer in the script. If it is computationally intensive, you can disable the loop in the bash scripts. Once the data is loaded, run the `process_geomag_data.py` and `process_tl_sub.py` for statistical analysis in the next phase.
