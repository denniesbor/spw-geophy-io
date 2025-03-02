#!/bin/bash

# Function to run download script
download_data() {
    script_name=$1

    echo "Running $script_name"

    # Run the python script
    python3 "$script_name"
}

# Number of repetitions for each script
iterations=10

# Loop for download_nrcan_old.py
for i in $(seq 1 $iterations); do
    echo "Iteration $i for download_nrcan_old.py"
    download_data "download_nrcan_old.py"
done

# Loop for download_usgs_old.py
for i in $(seq 1 $iterations); do
    echo "Iteration $i for download_usgs_old.py"
    download_data "download_usgs_old.py"
done
