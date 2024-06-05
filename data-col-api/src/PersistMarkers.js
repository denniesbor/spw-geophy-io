import axios from "axios";
import { saveMarkersAsGeoJSON } from "./WriteFile";
import { selectedSubstation } from "./Utils";

// Fetch users from the API
async function fetchUsers() {
  try {
    const response = await axios.get("https://denniesbor.com/gis/simpleusers/");
    return response.data;
  } catch (error) {
    return [];
  }
}

let markerLabels = {
  three_ph_transformer: "3 Ph. Transf",
  circuit_breaker: "Circuit Breaker",
  single_ph_transformer: "Single Ph. Transf",
  primary_power_line: "Pri. Pwr Line",
  sec_power_line: "Sec. Pwr Line",
  control_building: "Control Room",
};

const getKeyFromValue = (object, value) =>
  Object.keys(object).find((key) => object[key] === value);

// Function to save markers in a database
async function saveMarkers(payload) {
  try {
    const response = await axios.post(
      "https://denniesbor.com/gis/bulk_update/update_markers/",
      payload
    );
  } catch (error) {
    if (error.response) {
      console.error("Error saving markers:", error.response.data);
    } else {
      console.error("Error saving markers:", error.message);
    }
  }
}

let overlay = document.getElementById("overlay");
let savePopup = document.getElementById("savePopup");
let userContainer = document.getElementById("userContainer");
let cancelButton = document.getElementById("cancelButton");
let confirmSaveButton = document.getElementById("confirmSaveButton");
let errorMessage = document.getElementById("errorMessage");
let allowEmptyContainer = document.getElementById("allowEmptyContainer");
let allowEmptyCheckbox = document.getElementById("allowEmptyCheckbox");

// Function to save markers in a database or download
export async function saveMarkersToDatabase(markers, created_by, updated_by) {
  // Make the save popup and overlay visible
  overlay.style.display = "block";
  savePopup.style.display = "block";

  const users = await fetchUsers();
  if (users.length === 0) {
    return;
  }

  // Create the label
  const label = document.createElement("label");
  label.setAttribute("for", "userDropdown");
  label.textContent = created_by ? "Updated By:" : "Created By:";

  // Create the dropdown
  const dropdown = document.createElement("select");
  dropdown.id = "userDropdown";

  // Append label and dropdown to the container
  userContainer.innerHTML = ""; // Clear previous content
  userContainer.appendChild(label);
  userContainer.appendChild(dropdown);

  users.forEach((user) => {
    const option = document.createElement("option");
    option.value = user.username;
    option.textContent = user.username;
    dropdown.appendChild(option);
  });

  // Add event listener to the cancel button
  cancelButton.addEventListener("click", closePopup);

  // Ensure the confirm save button only has one event listener
  confirmSaveButton.replaceWith(confirmSaveButton.cloneNode(true));
  confirmSaveButton = document.getElementById("confirmSaveButton");

  // Add event listener to the allowEmptyCheckbox
  allowEmptyCheckbox.addEventListener("change", () => {
    confirmSaveButton.disabled = !allowEmptyCheckbox.checked;
  });

  confirmSaveButton.addEventListener("click", async () => {
    const downloadCheckbox =
      document.getElementById("downloadCheckbox").checked;
    const sendToDatabaseCheckbox = document.getElementById(
      "sendToDatabaseCheckbox"
    ).checked;

    // Check if markers are empty or if any marker has an invalid label
    let validMarkers = [];
    if (!markers || markers.length === 0) {
      validMarkers = [];
    } else {
      validMarkers = markers.filter((marker) => {
        const label = marker.getLabel()?.text;
        return typeof label === "string" && label.trim() !== "";
      });
    }

    if (
      (!markers || markers.length === 0 || validMarkers.length === 0) &&
      !allowEmptyCheckbox.checked
    ) {
      errorMessage.style.display = "block"; // Show error message
      allowEmptyContainer.style.display = "block"; // Show allow empty container
      confirmSaveButton.disabled = true; // Disable save button
      return;
    } else {
      errorMessage.style.display = "none"; // Hide error message
      allowEmptyContainer.style.display = "none"; // Hide allow empty container
      confirmSaveButton.disabled = false; // Enable save button
    }

    // If allow empty is checked, use all markers as valid
    const markersToSave = allowEmptyCheckbox.checked
      ? markers || []
      : validMarkers;

    // Create the GeoJSON object
    const geojson = {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          geometry: {
            type: "MultiPoint",
            coordinates: markersToSave.map((marker) => [
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
            threePhaseTransformerCount: 0,
            singlePhaseTransformerCount: 0,
            primaryPowerLineCount: 0,
            secondaryPowerLineCount: 0,
            totalTransformerCount: 0,
            markers: markersToSave.map((marker) => {
              const labelObj = marker.getLabel();
              const labelKey = getKeyFromValue(markerLabels, labelObj.text);

              return {
                label: labelKey,
                color: labelObj.color,
              };
            }),
          },
        },
      ],
    };

    // Count the types of markers
    markersToSave.forEach((marker) => {
      const labelKey = getKeyFromValue(markerLabels, marker.getLabel().text);
      switch (labelKey) {
        case "three_ph_transformer":
          geojson.features[0].properties.threePhaseTransformerCount++;
          geojson.features[0].properties.totalTransformerCount++;
          break;
        case "single_ph_transformer":
          geojson.features[0].properties.singlePhaseTransformerCount++;
          geojson.features[0].properties.totalTransformerCount++;
          break;
        case "primary_power_line":
          geojson.features[0].properties.primaryPowerLineCount++;
          break;
        case "sec_power_line":
          geojson.features[0].properties.secondaryPowerLineCount++;
          break;
      }
    });

    if (downloadCheckbox) {
      saveMarkersAsGeoJSON(geojson);
    }

    if (sendToDatabaseCheckbox) {
      // Create the payload
      const selectedUser = document.getElementById("userDropdown").value;
      const payload = {
        substation_id: selectedSubstation.SS_ID,
        created_by: created_by ? created_by : selectedUser,
        updated_by: selectedUser,
        markers: markersToSave.map((marker) => {
          const labelObj = marker.getLabel();
          const labelKey = getKeyFromValue(markerLabels, labelObj.text);

          return {
            label: labelKey,
            latitude: marker.getPosition().lat(),
            longitude: marker.getPosition().lng(),
          };
        }),
      };
      await saveMarkers(payload);
    }

    closePopup();
  });

  // Initial check if markers are empty
  if (!markers || markers.length === 0) {
    errorMessage.style.display = "block"; // Show error message
    allowEmptyContainer.style.display = "block"; // Show allow empty container
    confirmSaveButton.disabled = true; // Disable save button
  } else {
    errorMessage.style.display = "none"; // Hide error message
    allowEmptyContainer.style.display = "none"; // Hide allow empty container
    confirmSaveButton.disabled = false; // Enable save button
  }
}

function closePopup() {
  overlay.style.display = "none";
  savePopup.style.display = "none";
  errorMessage.style.display = "none"; // Hide error message when closing popup
  allowEmptyContainer.style.display = "none"; // Hide allow empty container when closing popup
  confirmSaveButton.disabled = false; // Enable save button when closing popup
}
