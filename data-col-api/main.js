import "./style.css";

let map, panorama, data, transmissionLinesLayer;
let currentMarkers = [];
let allTransmissionLines = [];

// Initialize map
function initMap() {
  map = new google.maps.Map(document.getElementById("map"), {
    center: { lat: 37.7749, lng: -122.4194 },
    zoom: 16,
    streetViewControl: true,
    mapId: "34e7058f5ed7d906",
  });

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
    .addEventListener("change", updateMapCenter);
  document
    .getElementById("toggleLines")
    .addEventListener("change", toggleTransmissionLines);

  map.addListener("click", function (event) {
    const location = {
      lat: event.latLng.lat(),
      lng: event.latLng.lng(),
    };
    checkStreetView(location);
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
      connected_tl_id: substation.connected_tl_id,
    });
    option.text = `${index + 1}) ${substation.SS_ID}`;
    ssDropdown.add(option);
  });

  if (substations.length > 0) {
    updateMapCenter();
  }
}

// Function to update the map center based on selected substation
function updateMapCenter() {
  const ssDropdown = document.getElementById("ssDropdown");
  const selectedSubstation = JSON.parse(ssDropdown.value);

  const location = {
    lat: parseFloat(selectedSubstation.lat),
    lng: parseFloat(selectedSubstation.lon),
  };

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

  // Check and toggle Street View
  checkStreetView(location);

  // Display transmission lines connected to the selected substation
  displayTransmissionLines(selectedSubstation.connected_tl_id);
}

// Function to check if Street View exists and toggle it
function checkStreetView(location) {
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

function toggleStreetView() {
  const toggle = panorama.getVisible();
  if (toggle == false) {
    panorama.setVisible(true);
  } else {
    panorama.setVisible(false);
  }
}

// Function to display transmission lines connected to the selected substation
function displayTransmissionLines(connected_tl_id) {
  if (transmissionLinesLayer) {
    transmissionLinesLayer.setMap(null);
  }
  transmissionLinesLayer = new google.maps.Data();
  const filteredLines = allTransmissionLines.filter((line) =>
    connected_tl_id.includes(line.properties.line_id)
  );

  console.log("Filtered transmission lines:", filteredLines); // Debugging step

  transmissionLinesLayer.addGeoJson({
    type: "FeatureCollection",
    features: filteredLines,
  });
  transmissionLinesLayer.setMap(map);
}

// Function to toggle transmission lines
function toggleTransmissionLines() {
  const toggleLines = document.getElementById("toggleLines");
  if (toggleLines.checked) {
    transmissionLinesLayer.setMap(map);
  } else {
    transmissionLinesLayer.setMap(null);
  }
}

// Load Google Maps API asynchronously
function loadScript() {
  const script = document.createElement("script");
  script.src = `https://maps.googleapis.com/maps/api/js?key=${
    import.meta.env.VITE_GOOGLE_MAPS_API_KEY
  }&callback=initMap&libraries=marker&v=weekly`;
  script.async = true;
  script.defer = true;
  document.head.appendChild(script);
}

window.onload = loadScript;

window.initMap = initMap;
