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
  if (!markersObject[ss_id]) {
    markersObject[ss_id] = [];
  }

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

function createMarker(
  map,
  ss_id,
  location,
  color,
  label,
  labelColor = "black"
) {
  let marker = new google.maps.Marker({
    position: location,
    map: map,
    draggable: true,
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
      color: labelColor,
      fontSize: "18px",
    },
  });

  marker.addListener("dragend", function (event) {
    location.lat = event.latLng.lat();
    location.lng = event.latLng.lng();
  });

  marker.addListener("rightclick", function () {
    marker.setMap(null);
    markersObject[ss_id] = markersObject[ss_id].filter(
      (m) =>
        m.getPosition().lat() !== marker.getPosition().lat() ||
        m.getPosition().lng() !== marker.getPosition().lng()
    );
  });

  addMarker(ss_id, marker);
}

export async function getMarkers(map) {
  substationId = selectedSubstation.SS_ID;

  console.log("ssDropdown changed");

  markersContainer.style.display = "flex";

  if (toggleSubstationLabels.checked) {
    clearMarkers(substationId);
    console.log("Toggled");

    let dbMarkers = await fetchMarkers(substationId);
    if (dbMarkers) {
      dbMarkers.forEach((marker) => {
        let color = markerColors[marker.label];
        let label = markerLabels[marker.label];

        let location = {
          lat: marker.latitude,
          lng: marker.longitude,
        };
        createMarker(map, substationId, location, color, label, "red");
      });
    }

    map.addListener("click", (event) => onMapClick(event, map));

    let markerDivs = document.querySelectorAll(".marker");
    markerDivs.forEach((div) => {
      div.addEventListener("click", onMarkerDivClick);
    });

    document
      .getElementById("disableMarkerAddingButton")
      .addEventListener("click", disableMarkerSelection);

    document.getElementById("ssDropdown").addEventListener("change", () => {
      clearMarkers(substationId);
    });

    document.getElementById("saveMarkers").addEventListener("click", () => {
      saveMarkersToDatabase(
        markersObject[substationId],
        created_by,
        updated_by
      );
    });
  } else {
    markersContainer.style.display = "none";
    clearMarkers(substationId);
    markerMessage.style.display = "none";

    google.maps.event.clearListeners(map, "click");
  }
}

function onMapClick(event, map) {
  const location = {
    lat: event.latLng.lat(),
    lng: event.latLng.lng(),
  };

  let selectedItem = document.querySelector(".marker.selected");

  if (selectedItem) {
    let color = markerColors[selectedItem.id];
    let label = markerLabels[selectedItem.id];

    if (!markerExists(substationId, location)) {
      createMarker(map, substationId, location, color, label);
    }
  }
}

function onMarkerDivClick() {
  if (this.classList.contains("selected")) {
    document.querySelectorAll(".marker").forEach((div) => {
      div.classList.add("disabled");
    });

    return;
  }

  document
    .querySelectorAll(".marker")
    .forEach((div) => div.classList.remove("selected"));

  this.classList.add("selected");
}

function clearMarkers(substationId) {
  if (markersObject[substationId]) {
    markersObject[substationId].forEach((marker) => marker.setMap(null));
    markersObject[substationId] = [];
  }
}

function disableMarkerSelection() {
  let toggleButton = document.getElementById("disableMarkerAddingButton");
  toggleButton.textContent = "Done Adding Markers";

  let selectedMarker = document.querySelector(".marker.selected");
  if (selectedMarker) {
    selectedMarker.classList.remove("selected");
  }
}

function markerExists(ss_id, location) {
  return markersObject[ss_id]?.some(
    (m) =>
      m.getPosition().lat() === location.lat &&
      m.getPosition().lng() === location.lng
  );
}
