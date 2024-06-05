// Markers component
import { saveMarkersToDatabase } from "./PersistMarkers";
import { selectedSubstation } from "./Utils";
import axios from "axios";

let markersObject = {}; // Initialize an empty object
let substationId;

const markerMessage = document.getElementById("markerMessage");
let created_by, updated_by;

// Object to hold the colors for each type of marker
let markerColors = {
  three_ph_transformer: "blue",
  circuit_breaker: "green",
  single_ph_transformer: "yellow",
  primary_power_line: "purple",
  sec_power_line: "pink",
  control_building: "brown",
};

let markerLabels = {
  three_ph_transformer: "3 Ph. Transf",
  circuit_breaker: "Circuit Breaker",
  single_ph_transformer: "Single Ph. Transf",
  primary_power_line: "Pri. Pwr Line",
  sec_power_line: "Sec. Pwr Line",
  control_building: "Control Room",
};

async function fetchMarkers(substationId) {
  try {
    const response = await axios.get(
      `https://denniesbor.com/gis/markers/?substation=${substationId}`
    );
    const markerData = response.data;

    if (markerData.length === 0) {
      markerMessage.textContent = "No markers available.";
      markerMessage.style.display = "block";
    } else {
      created_by = markerData[0].created_by;
      updated_by = markerData[0].updated_by;
      markerMessage.textContent = `Markers available. Created by: ${created_by}, Updated by: ${updated_by}`;
      markerMessage.style.display = "block";
      return markerData;
    }
  } catch (error) {
    markerMessage.textContent = "No markers available.";
  }
}

function addMarker(ss_id, marker) {
  // Initialize the markers array for the substation if it doesn't exist
  if (!markersObject[ss_id]) {
    markersObject[ss_id] = [];
  }

  // Push the new marker into the array associated with the ss_id key if the marker doesn't already exist
  if (
    !markersObject[ss_id].find(
      (m) =>
        m.getPosition().lat() === marker.getPosition().lat() &&
        m.getPosition().lng() === marker.getPosition().lng()
    )
  ) {
    markersObject[ss_id].push(marker);
  }
}

const toggleSubstationLabels = document.getElementById(
  "toggleSubstationLabels"
);

const markersContainer = document.getElementById("markers");

// create google map marker object
function createMarker(
  map,
  ss_id,
  location,
  color,
  label,
  labelColor = "black"
) {
  // Create a marker
  let marker = new google.maps.Marker({
    position: location,
    map: map,
    draggable: true, // This allows the user to move the marker
    icon: {
      url:
        "data:image/svg+xml;charset=UTF-8," +
        encodeURIComponent(`
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512" width="32" height="32" fill="${color}">
            <path d="M16 144a144 144 0 1 1 288 0A144 144 0 1 1 16 144zM160 80c8.8 0 16-7.2 16-16s-7.2-16-16-16c-53 0-96 43-96 96c0 8.8 7.2 16 16 16s16-7.2 16-16c0-35.3 28.7-64 64-64zM128 480V317.1c10.4 1.9 21.1 2.9 32 2.9s21.6-1 32-2.9V480c0 17.7-14.3 32-32 32s-32-14.3-32-32z"/></svg>
            `),
      scaledSize: new google.maps.Size(32, 32),
      labelOrigin: new google.maps.Point(50, -12),
    },
    label: {
      text: label,
      color: labelColor, // Set the color of the label
      fontSize: "18px",
    },
  });

  // Add an event listener for the dragend event on the marker
  marker.addListener("dragend", function (event) {
    // Update the location object with the new coordinates
    location.lat = event.latLng.lat();
    location.lng = event.latLng.lng();
  });

  // Add an event listener for the rightclick event on the marker to remove it
  marker.addListener("rightclick", function () {
    marker.setMap(null); // Remove the marker from the map
    markersObject[ss_id] = markersObject[ss_id].filter(
      (m) =>
        m.getPosition().lat() !== marker.getPosition().lat() ||
        m.getPosition().lng() !== marker.getPosition().lng()
    );
  });

  // Add the marker to the array
  addMarker(ss_id, marker);
}

export function getMarkers(map) {
  // Listen for toggleSubstationLabels change event
  toggleSubstationLabels.addEventListener("change", async () => {
    substationId = selectedSubstation.SS_ID;

    // marke markers container visible
    markersContainer.style.display = "flex";

    if (toggleSubstationLabels.checked) {
      // Clear existing markers for the substation
      clearMarkers(substationId);

      // Try fetching markers from the database
      let dbMarkers = await fetchMarkers(substationId);
      if (dbMarkers) {
        dbMarkers.forEach((marker) => {
          // Get the color for the marker
          let color = markerColors[marker.label];
          // Label
          let label = markerLabels[marker.label];

          let location = {
            lat: marker.latitude,
            lng: marker.longitude,
          };
          // Create a marker
          createMarker(map, substationId, location, color, label, "red");
        });

        // Make the marker message invisible
      }

      // Add click event listener to the map for adding new markers
      map.addListener("click", (event) => onMapClick(event, map));

      // Add click event to the marker divs
      let markerDivs = document.querySelectorAll(".marker");
      markerDivs.forEach((div) => {
        div.addEventListener("click", onMarkerDivClick);
      });

      // Add event listener to the disable button
      document
        .getElementById("disableMarkerAddingButton")
        .addEventListener("click", disableMarkerSelection);

      // Remove markers in case of a change of substation
      document.getElementById("ssDropdown").addEventListener("change", () => {
        clearMarkers(substationId);
      });

      // Save markers
      document.getElementById("saveMarkers").addEventListener("click", () => {
        saveMarkersToDatabase(
          markersObject[substationId],
          created_by,
          updated_by
        );
      });
    } else {
      // Hide the markers container
      markersContainer.style.display = "none";
      // Clear existing markers for the substation
      clearMarkers(substationId);
      markerMessage.style.display = "none";

      // Remove click event listener from the map
      google.maps.event.clearListeners(map, "click");
    }
  });
}

// Function to handle map click event
function onMapClick(event, map) {
  const location = {
    lat: event.latLng.lat(),
    lng: event.latLng.lng(),
  };

  // Get the selected item from the div
  let selectedItem = document.querySelector(".marker.selected");

  if (selectedItem) {
    // Get the color for the selected item
    let color = markerColors[selectedItem.id];

    // Label
    let label = markerLabels[selectedItem.id];

    // Create a marker
    createMarker(map, substationId, location, color, label);
  }
}

// Function to handle marker div click event
function onMarkerDivClick() {
  // Check if the selected marker is clicked again
  if (this.classList.contains("selected")) {
    // Disable all markers
    document.querySelectorAll(".marker").forEach((div) => {
      div.classList.add("disabled");
    });

    return; // Exit the function
  }

  // Remove selected class from all divs
  document
    .querySelectorAll(".marker")
    .forEach((div) => div.classList.remove("selected"));

  // Add selected class to the clicked div
  this.classList.add("selected");
}

// Function to clear markers
function clearMarkers(substationId) {
  if (markersObject[substationId]) {
    markersObject[substationId].forEach((marker) => marker.setMap(null));
    markersObject[substationId] = [];
  }
}

// Function to disable marker selection
function disableMarkerSelection() {
  let toggleButton = document.getElementById("disableMarkerAddingButton");
  toggleButton.textContent = "Done Adding Markers";

  // Deselect the selected marker
  let selectedMarker = document.querySelector(".marker.selected");
  if (selectedMarker) {
    selectedMarker.classList.remove("selected");
  }
}
