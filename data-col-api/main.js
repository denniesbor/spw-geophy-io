import "./style.css";
import { getMap } from "./src/Map";
import { getMarkers } from "./src/Markers";
import {
  updateMapCenter,
  checkStreetView,
  selectedSubstation,
} from "./src/Utils";
import { RotateTilt } from "./src/RotateTilt";

let map, panorama, data, transmissionLinesLayer;
let allTransmissionLines = [];
let mapDivRect;

const toggleSubstationLabels = document.getElementById(
  "toggleSubstationLabels"
);

// Initialize map
async function initMap() {
  map = getMap();

  // Rotate or tilt map
  RotateTilt(map);

  mapDivRect = document.getElementById("map").getBoundingClientRect();
  panorama = map.getStreetView();

  // Fetch and process data
  fetchData();
  fetchTransmissionLines();

  // Event listeners
  document
    .getElementById("regionDropdown")
    .addEventListener("change", (event) => {
      populateSubstations(event),
        getMarkers(map);
    });
  document.getElementById("ssDropdown").addEventListener("change", () => {
    updateMapCenter(map);
    // Get markers  from Markers.js
    getMarkers(map);
  });
  toggleSubstationLabels.addEventListener("change", () => {
    getMarkers(map);
  });

  document
    .getElementById("toggleLines")
    .addEventListener("change", () =>
      toggleTransmissionLines(
        selectedSubstation.connected_tl_id,
        transmissionLinesLayer,
        allTransmissionLines,
        map
      )
    );

  // // document
  // //   .getElementById("captureButton")
  // //   .addEventListener("click", captureMap);

  map.addListener("click", function (event) {
    const location = {
      lat: event.latLng.lat(),
      lng: event.latLng.lng(),
    };
    let selectedMarker = document.querySelector(".marker.selected");
    if (!selectedMarker) {
      checkStreetView(location, panorama);
    }
  });
}

// Fetch data from JSON file
async function fetchData() {
  let url =
    "https://gist.githubusercontent.com/denniesbor/81dfc2b05d3c7ee0f02dfc20ec15dce8/raw/a9c6c3b0f9fdc604d4884ded950b1d5035c90e22/tm_ss_df_gt_300v.json";
  try {
    const response = await fetch(url);
    data = await response.json();
    if (Array.isArray(data)) {
      populateRegions();
    } else {
      console.error("Data is not an array:", data);
    }
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
    transmissionLinesLayer = new google.maps.Data();
    transmissionLinesLayer.addGeoJson(geoJsonData);

    allTransmissionLines = geoJsonData.features; // Store all transmission lines
  } catch (error) {
    console.error("Error fetching transmission lines:", error);
  }
}

// Function to fetch unique regions and populate the region dropdown
function populateRegions() {
  const regionDropdown = document.getElementById("regionDropdown");
  const regions = [...new Set(data.map((item) => item.REGION))];
  console.log(regions)
  regions.forEach((region) => {
    const option = document.createElement("option");
    option.value = region;
    option.text = region;
    regionDropdown.add(option);
  });

  // Set default region to 'PJM' if it exists, otherwise set to the first available region
  const defaultRegion = regions.includes("PJM") ? "PJM" : regions[0];
  regionDropdown.value = defaultRegion;
  populateSubstations();
}

// Function to populate substation dropdown based on selected region
function populateSubstations() {
  const regionDropdown = document.getElementById("regionDropdown");
  const ssDropdown = document.getElementById("ssDropdown");
  ssDropdown.innerHTML = "";

  const selectedRegion = regionDropdown.value;
  const substations = data.filter((item) => item.REGION === selectedRegion);

  substations.forEach((substation, index) => {
    const option = document.createElement("option");
    option.value = JSON.stringify({
      lat: substation.lat,
      lon: substation.lon,
      SS_ID: substation.SS_ID,
      SS_NAME: substation.SS_NAME,
      SS_OPERATOR: substation.SS_OPERATOR,
      SS_VOLTAGE: substation.SS_VOLTAGE,
      SS_TYPE: substation.SS_TYPE,
      REGION_ID: substation.REGION_ID,
      REGION: substation.REGION,
      connected_tl_id: substation.connected_tl_id,
      LINE_VOLTS: substation.LINE_VOLTS,
    });

    option.text = `${index + 1}) ${substation.SS_ID}`;
    ssDropdown.add(option);
  });

  if (substations.length > 0) {
    updateMapCenter(
      map,
      panorama,
      transmissionLinesLayer,
      allTransmissionLines
    );
  }
}

// Function to display transmission lines connected to the selected substation

// colors mapping transmission voltage class
export const voltageColorMapping = {
  345: "red",
  115: "green",
  230: "blue",
  161: "yellow",
  500: "purple",
  100: "orange",
  138: "pink",
  765: "brown",
  120: "cyan",
  220: "magenta"
}

// Function to get color based on voltage
export function getColor(voltage) {
  return voltageColorMapping[voltage] || "black"; // default to black if no match
}

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
  console.log(filteredLines)
  transmissionLinesLayer.addGeoJson({
    type: "FeatureCollection",
    features: filteredLines,
  });
  // Set the style of the transmission lines based on their voltage
  transmissionLinesLayer.setStyle((feature) => {
    const voltage = feature.getProperty('VOLTAGE');
    const color = getColor(voltage);
    return {
      strokeColor: color,
      strokeWeight: 2
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
  map
) {
  const toggleLines = document.getElementById("toggleLines");
  if (toggleLines.checked) {
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
  transmissionLinesLayer = currentTransmissionLinesLayer;
}

// Load Google Maps API asynchronously
function loadScript() {
  const script = document.createElement("script");
  script.src = `https://maps.googleapis.com/maps/api/js?key=${import.meta.env.VITE_GOOGLE_MAPS_API_KEY
    }&loading=async&libraries=drawing,marker&callback=initMap&v=weekly`;
  script.async = true;
  script.defer = true;
  document.head.appendChild(script);
}

window.onload = loadScript;

window.initMap = initMap;
