// Fetch data from JSON file
async function fetchData() {
  let url =
    "https://gist.githubusercontent.com/denniesbor/81dfc2b05d3c7ee0f02dfc20ec15dce8/raw/a9c6c3b0f9fdc604d4884ded950b1d5035c90e22/tm_ss_df_gt_300v.json";
  try {
    const response = await fetch(url);
    let data = await response.json();
    console.log("Data fetched successfully:"); // Debugging step

    return data; // Return the fetched data
  } catch (error) {
    console.error("Error fetching data:", error);
  }
}

// Fetch GeoJSON for transmission lines
async function fetchTransmissionLines() {
  let url =
    "https://gist.githubusercontent.com/denniesbor/f55327cd9a7ba2c7da2725c5b03b17f0/raw/ece03a294a758201597da9c80a50759726425b09/tm_lines_within_ferc.geojson";
  try {
    const response = await fetch(url);
    const geoJsonData = await response.json();
    let transmissionLinesLayer = new google.maps.Data();
    transmissionLinesLayer.addGeoJson(geoJsonData);

    console.log("Transmission lines fetched successfully:"); // Debugging step

    let allTransmissionLines = geoJsonData.features; // Store all transmission lines

    return { allTransmissionLines, transmissionLinesLayer };
  } catch (error) {
    console.error("Error fetching transmission lines:", error);
  }
}
export { fetchData, fetchTransmissionLines };
