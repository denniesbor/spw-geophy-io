import "./style.css";

let map, panorama, data, transmissionLinesLayer;
let currentMarkers = [];
let allTransmissionLines = [];
let drawingManager;
let selectedArea;
let mapDivRect;

// Initialize map
function initMap() {
  map = new google.maps.Map(document.getElementById("map"), {
    center: {
      lat: 37.7893719,
      lng: -122.3942,
    },
    zoom: 16,
    heading: 320,
    tilt: 0,
    mapId: "90f87356969d889c",
  });

  const buttons = [
    ["Rotate Left", "rotate", 5, google.maps.ControlPosition.LEFT_CENTER],
    ["Rotate Right", "rotate", -5, google.maps.ControlPosition.RIGHT_CENTER],
    ["Tilt Down", "tilt", 5, google.maps.ControlPosition.TOP_CENTER],
    ["Tilt Up", "tilt", -5, google.maps.ControlPosition.BOTTOM_CENTER],
  ];

  buttons.forEach(([text, mode, amount, position]) => {
    const controlDiv = document.createElement("div");
    const controlUI = document.createElement("button");

    controlUI.classList.add("ui-button");
    controlUI.innerText = `${text}`;
    controlUI.addEventListener("click", () => {
      adjustMap(mode, amount);
    });
    controlDiv.appendChild(controlUI);
    map.controls[position].push(controlDiv);
  });

  const adjustMap = function (mode, amount) {
    switch (mode) {
      case "tilt":
        map.setTilt(map.getTilt() + amount);
        break;
      case "rotate":
        map.setHeading(map.getHeading() + amount);
        break;
      default:
        break;
    }
  };

  mapDivRect = document.getElementById("map").getBoundingClientRect();

  panorama = map.getStreetView();

  drawingManager = new google.maps.drawing.DrawingManager({
    drawingMode: google.maps.drawing.OverlayType.RECTANGLE,
    drawingControl: true,
    drawingControlOptions: {
      position: google.maps.ControlPosition.TOP_CENTER,
      drawingModes: ["rectangle"],
    },
    rectangleOptions: {
      fillColor: "#ffff00",
      fillOpacity: 0.1,
      strokeWeight: 2,
      clickable: false,
      editable: true,
      zIndex: 1,
    },
  });

  // drawingManager.setMap(map);

  // google.maps.event.addListener(
  //   drawingManager,
  //   "rectanglecomplete",
  //   function (rectangle) {
  //     if (selectedArea) {
  //       selectedArea.setMap(null);
  //     }
  //     selectedArea = rectangle;
  //     drawingManager.setDrawingMode(null); // Disable drawing mode after the rectangle is complete
  //   }
  // );

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
  // document
  //   .getElementById("captureButton")
  //   .addEventListener("click", captureMap);

  map.addListener("click", function (event) {
    const location = {
      lat: event.latLng.lat(),
      lng: event.latLng.lng(),
    };
    checkStreetView(location);
  });
}

function captureMap() {
  if (selectedArea) {
    const bounds = selectedArea.getBounds();
    const ne = bounds.getNorthEast();
    const sw = bounds.getSouthWest();

    // Convert geographic coordinates to pixel coordinates
    const projection = map.getProjection();
    const scale = Math.pow(2, map.getZoom());

    const neWorldPoint = projection.fromLatLngToPoint(ne);
    const swWorldPoint = projection.fromLatLngToPoint(sw);

    const nePixel = new google.maps.Point(
      neWorldPoint.x * scale,
      neWorldPoint.y * scale
    );
    const swPixel = new google.maps.Point(
      swWorldPoint.x * scale,
      swWorldPoint.y * scale
    );

    const mapDiv = document.getElementById("map");
    const mapRect = mapDiv.getBoundingClientRect();

    // Calculate the center pixel coordinates
    const mapCenter = map.getCenter();
    const mapCenterWorldPoint = projection.fromLatLngToPoint(mapCenter);
    const mapCenterPixel = new google.maps.Point(
      mapCenterWorldPoint.x * scale,
      mapCenterWorldPoint.y * scale
    );

    // Calculate offsets for the map container
    const containerStartX = mapCenterPixel.x - mapRect.width / 2;
    const containerStartY = mapCenterPixel.y - mapRect.height / 2;

    // Calculate the startX, startY, width, and height for the clipping area
    const startX = nePixel.x - containerStartX;
    const startY = nePixel.y - containerStartY;
    const width = swPixel.x - nePixel.x;
    const height = swPixel.y - nePixel.y;

    // Capture the entire map using html2canvas
    html2canvas(mapDiv, {
      allowTaint: true,
      useCORS: true,
      scale: window.devicePixelRatio,
    })
      .then(function (canvas) {
        // Create a new canvas to hold the clipped image
        const clippedCanvas = document.createElement("canvas");
        const context = clippedCanvas.getContext("2d");

        // Set the dimensions of the clipped canvas
        clippedCanvas.width = Math.abs(width);
        clippedCanvas.height = Math.abs(height);

        // Draw the clipped image onto the new canvas
        context.drawImage(
          canvas,
          startX, // x-coordinate where to start clipping
          startY, // y-coordinate where to start clipping
          width, // width of the clipped area
          height, // height of the clipped area
          0, // x-coordinate where to place the clipped image on the new canvas
          0, // y-coordinate where to place the clipped image on the new canvas
          Math.abs(width), // width of the clipped image on the new canvas
          Math.abs(height) // height of the clipped image on the new canvas
        );

        // Convert clipped canvas to an image and display or save it
        const link = document.createElement("a");
        link.download = "selected-area-screenshot.png";
        link.href = clippedCanvas.toDataURL();
        link.click();
      })
      .catch((error) => {
        console.error("Error capturing the map area:", error);
      });
  } else {
    alert("Please draw a rectangle first.");
  }
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
    updateMapCenter();
  }
}

// Function to update the info box based on selected substation
function updateInfoBox(substation) {
  document.getElementById("ssName").innerText = substation.SS_NAME;
  document.getElementById("ssOperator").innerText = substation.SS_OPERATOR;
  document.getElementById("ssVoltages").innerText = substation.SS_VOLTAGE;
  document.getElementById("lineVoltages").innerText =
    substation.LINE_VOLTS.join(", ");
}

// Function to update the map center based on selected substation
function updateMapCenter() {
  const ssDropdown = document.getElementById("ssDropdown");
  const selectedSubstation = JSON.parse(ssDropdown.value);

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

// Function to display transmission lines connected to the selected substation
function displayTransmissionLines(connected_tl_id) {
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
  }&loading=async&libraries=drawing,marker&callback=initMap&v=weekly`;
  script.async = true;
  script.defer = true;
  document.head.appendChild(script);
}

window.onload = loadScript;

window.initMap = initMap;
