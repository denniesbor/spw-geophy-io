import "./style.css";
import { getMap } from "./src/Map";
import { getMarkers } from "./src/Markers";
import {
  updateMapCenter,
  toggleTransmissionLines,
  checkStreetView,
} from "./src/Utils";
import { RotateTilt } from "./src/RotateTilt";

let map, panorama, data, transmissionLinesLayer;
let allTransmissionLines = [];
let mapDivRect;

// Initialize map
function initMap() {
  map = getMap();

  // Get markers  from Markers.js
  getMarkers(map);
  RotateTilt(map);

  mapDivRect = document.getElementById("map").getBoundingClientRect();
  panorama = map.getStreetView();

  // Fetch and process data
  fetchData();
  fetchTransmissionLines();

  // Event listeners
  document
    .getElementById("regionDropdown")
    .addEventListener("change", populateSubstations);
  document
    .getElementById("ssDropdown")
    .addEventListener("change", () =>
      updateMapCenter(
        map,
        panorama,
        transmissionLinesLayer,
        allTransmissionLines
      )
    );
  document
    .getElementById("toggleLines")
    .addEventListener("change", () =>
      toggleTransmissionLines(transmissionLinesLayer, map)
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
    console.log("Data fetched successfully:"); // Debugging step
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

    console.log("Transmission lines fetched successfully:"); // Debugging step

    allTransmissionLines = geoJsonData.features; // Store all transmission lines
  } catch (error) {
    console.error("Error fetching transmission lines:", error);
  }
}

// Function to fetch unique regions and populate the region dropdown
function populateRegions() {
  const regionDropdown = document.getElementById("regionDropdown");
  const regions = [...new Set(data.map((item) => item.REGION))];
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

// Load Google Maps API asynchronously
function loadScript() {
  const script = document.createElement("script");
  script.src = `https://maps.googleapis.com/maps/api/js?key=${
    import.meta.env.VITE_GOOGLE_MAPS_API_KEY
  }&loading=async&libraries=drawing,marker&callback=initMap&v=weekly`;
  script.async = true;
  script.defer = true;
  document.head.appendChild(script);
}

window.onload = loadScript;

window.initMap = initMap;
