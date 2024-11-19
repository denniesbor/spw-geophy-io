# Data Download and Preprocessing

I have provided routines and scripts to download data from USGS and NRCAN FTP servers.

## USGS Geomagnetic Data

USGS data, accessible from [USGS Geomag](insert_link_here), is straightforward to obtain, especially for data before 1989.

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
