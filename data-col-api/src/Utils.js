import { voltageColorMapping } from "../main"
import { getColor } from "../main"
export let selectedSubstation;
let currentMarkers = [];

// Function to update the info box based on selected substation
function updateInfoBox(substation) {
  console.log(voltageColorMapping);

  document.getElementById("ssName").innerText = substation.SS_NAME;
  document.getElementById("ssOperator").innerText = substation.SS_OPERATOR;
  document.getElementById("ssVoltages").innerText = substation.SS_VOLTAGE;

  // Get the container element for line voltages
  const container = document.getElementById("lineVoltages");

  // Clear existing content
  container.innerHTML = "";

  // Create and append colored box and text for each voltage
  substation.LINE_VOLTS.forEach(voltage => {
    const colorBox = document.createElement("span");
    colorBox.style.display = "inline-block";
    colorBox.style.width = "10px";
    colorBox.style.height = "10px";
    colorBox.style.backgroundColor = getColor(voltage);
    colorBox.style.marginRight = "5px";

    const voltageText = document.createElement("span");
    voltageText.style.color = "black";
    voltageText.innerText = voltage;

    container.appendChild(colorBox);
    container.appendChild(voltageText);
    container.appendChild(document.createTextNode(", "));
  });

  // Remove the trailing comma and space
  if (container.lastChild) {
    container.removeChild(container.lastChild);
  }
}

// Function to update the map center based on selected substation
export function updateMapCenter(
  map
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
    }
  });
}