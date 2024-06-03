import { selectedSubstation } from "./Utils";

export function saveMarkersAsGeoJSON(markers) {

  let threePhaseTransformerCount = 0;
  let singlePhaseTransformerCount = 0;
  let primaryPowerLineCount = 0;
  let secondaryPowerLineCount = 0;
  let totalTransformerCount = 0;

  // Count the types of markers
  markers.forEach((marker) => {
    switch (marker.getLabel()) {
      case "three_ph_transformer":
        threePhaseTransformerCount++;
        totalTransformerCount++;
        break;
      case "single_ph_transformer":
        singlePhaseTransformerCount++;
        totalTransformerCount++;
        break;
      case "primary_power_line":
        primaryPowerLineCount++;
        break;
      case "sec_power_line":
        secondaryPowerLineCount++;
        break;
    }
  });

  const geojson = {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        geometry: {
          type: "MultiPoint",
          coordinates: markers.map((marker) => [
            marker.getPosition().lng(),
            marker.getPosition().lat(),
          ]),
        },
        properties: {
          SS_ID: selectedSubstation.SS_ID,
          SS_NAME: selectedSubstation.SS_NAME,
          SS_OPERATOR: selectedSubstation.SS_OPERATOR,
          SS_TYPE: selectedSubstation.SS_TYPE,
          SS_VOLTAGE: selectedSubstation.SS_VOLTAGE,
          connected_tl_id: selectedSubstation.connected_tl_id,
          LINE_VOLTS: selectedSubstation.LINE_VOLTS,
          REGION: selectedSubstation.REGION,
          REGION_ID: selectedSubstation.REGION_ID,
          threePhaseTransformerCount: threePhaseTransformerCount,
          singlePhaseTransformerCount: singlePhaseTransformerCount,
          primaryPowerLineCount: primaryPowerLineCount,
          secondaryPowerLineCount: secondaryPowerLineCount,
          totalTransformerCount: totalTransformerCount,
          markers: markers.map((marker) => ({
            type: marker.getLabel(),
            color: marker.icon.fillColor,
          })),
        },
      },
    ],
  };

  // Convert GeoJSON object to string
  const geojsonString = JSON.stringify(geojson, null, 2);

  // Create a Blob from the GeoJSON string
  const blob = new Blob([geojsonString], { type: "application/geo+json" });

  // Create a link element to trigger the download
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${selectedSubstation.REGION}_${
    selectedSubstation.SS_ID
  }_${selectedSubstation.lat.toFixed(4)}_${selectedSubstation.lon.toFixed(
    4
  )}.geojson`; // Set the file name

  // Append the link to the body
  document.body.appendChild(link);

  // Trigger the download
  link.click();

  // Clean up by removing the link
  document.body.removeChild(link);
}
