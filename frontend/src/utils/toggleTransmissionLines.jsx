import { getColor } from "./utils";

// Function to display transmission lines
export function displayTransmissionLines(
  connected_tl_id,
  transmissionLinesLayer,
  allTransmissionLines,
  map
) {
  if (transmissionLinesLayer) {
    transmissionLinesLayer.setMap(null);
  }
  transmissionLinesLayer = new google.maps.Data();
  const filteredLines = allTransmissionLines.features.filter((line) =>
    connected_tl_id.includes(line.properties.line_id)
  );
  transmissionLinesLayer.addGeoJson({
    type: "FeatureCollection",
    features: filteredLines,
  });
  // Set the style of the transmission lines based on their voltage
  transmissionLinesLayer.setStyle((feature) => {
    const voltage = feature.getProperty("VOLTAGE");
    const color = getColor(voltage);
    return {
      strokeColor: color,
      strokeWeight: 2,
    };
  });
  transmissionLinesLayer.setMap(map);
  return transmissionLinesLayer;
}

// Function to toggle transmission lines
export function toggleTransmissionLines(
  connected_tl_id,
  currentTransmissionLinesLayer,
  allTransmissionLines,
  map,
  isChecked,
  setTransmissionLinesLayer
) {
  if (isChecked) {
    currentTransmissionLinesLayer = displayTransmissionLines(
      connected_tl_id,
      currentTransmissionLinesLayer,
      allTransmissionLines,
      map
    );
  } else {
    if (currentTransmissionLinesLayer) {
      currentTransmissionLinesLayer.setMap(null);
      currentTransmissionLinesLayer = null;
    }
  }
  setTransmissionLinesLayer(currentTransmissionLinesLayer);
}
