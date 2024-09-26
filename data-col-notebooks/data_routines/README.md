# Download and preprocess data
I have provided routines and scripts to download data from USGS and NRCAN FTP servers. 
USGS, accesible from here [usgs geomag]() is straighforward especially for data before 1989. For NRCAN, the data is shared in mseed format through FDSNWS network. Information on the channels, location and network can be accessed here, [!nrcan geomag](https://geomag.nrcan.gc.ca/data-donnee/sd-en.php)
Technically, we use `obspy` package to read the mseeed files and export into csv files for preprocessing. Some requests return 404, or 204 error for NRCAN data even through data is available. So you have to repeatedly run the scripts to get all the datasets. You can set a cron task that can repeatedly run the fetch script till you are satisfied you have all the scripts.

To download data from 1991 to date, we readily7 use data accessible from Intermagnet. USe the script `download_geomag_data.py` to download the data for the NRCAN and US observatories geomagneitc data. 

### Electromagnerc Transfer Functions (EMTF)
We have used data of the entire contigous used from the EarthScope USArray and the latest USMT arrays. The xml files can be downloaded from here which can be downloaded via the browser interface from here [!https://ds.iris.edu/spud/emtf]()

## Transmission lines
We have used the open data available of the US transmission lines accessible from here [(https://hifld-geoplatform.hub.arcgis.com/datasets/geoplatform::transmission-lines/about)]. We focussed on EXtra high voltage transmission lines, and so we filter out those lines whose voltage rating is below 200 kV. The substations data is also queried from openstreet map, and the script to fetch the data is 

