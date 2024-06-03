export let selectedSubstation;
let currentMarkers = [];

// Function to update the info box based on selected substation
function updateInfoBox(substation) {
  document.getElementById("ssName").innerText = substation.SS_NAME;
  document.getElementById("ssOperator").innerText = substation.SS_OPERATOR;
  document.getElementById("ssVoltages").innerText = substation.SS_VOLTAGE;
  document.getElementById("lineVoltages").innerText =
    substation.LINE_VOLTS.join(", ");
}

// Function to update the map center based on selected substation
export function updateMapCenter(
  map,
  panorama,
  transmissionLinesLayer,
  allTransmissionLines
) {
  const ssDropdown = document.getElementById("ssDropdown");
  selectedSubstation = JSON.parse(ssDropdown.value);

  const location = {
    lat: parseFloat(selectedSubstation.lat),
    lng: parseFloat(selectedSubstation.lon),
  };

  updateInfoBox(selectedSubstation); // Update the info box

  // Clear existing markers
  currentMarkers.forEach((marker) => marker.setMap(null));
  currentMarkers = [];

  // Add a red marker at the center location
  const marker = new google.maps.marker.AdvancedMarkerElement({
    position: location,
    map: map,
  });

  currentMarkers.push(marker); // Store the marker

  // Set map center to the selected substation
  map.setCenter(location);
}

// Function to check if Street View exists and toggle it
export function checkStreetView(location, panorama) {
  const sv = new google.maps.StreetViewService();

  sv.getPanorama({ location: location, radius: 50 }, function (data, status) {
    if (status === "OK") {
      panorama.setPano(data.location.pano);
      panorama.setPov({
        heading: 270, // Adjust heading as needed
        pitch: 0,
      });
      panorama.setVisible(true);
    } else {
      panorama.setVisible(false);
      console.log("Street View data not found for this location.");
    }
  });
}

// Function to display transmission lines connected to the selected substation
function displayTransmissionLines(
  connected_tl_id,
  transmissionLinesLayer,
  allTransmissionLines,
  map
) {
  if (transmissionLinesLayer) {
    transmissionLinesLayer.setMap(null);
  }
  transmissionLinesLayer = new google.maps.Data();
  const filteredLines = allTransmissionLines.filter((line) =>
    connected_tl_id.includes(line.properties.line_id)
  );
  transmissionLinesLayer.addGeoJson({
    type: "FeatureCollection",
    features: filteredLines,
  });
  transmissionLinesLayer.setMap(map);
}
