import { saveMarkersAsGeoJSON } from "./saveFileUtils";
import { getKeyFromValue } from "./utils";
import { getCurrentUser, getUserDetails } from "../services/authService";
import axiosInstance from "../services/axiosInstance";

const VITE_API_URL = import.meta.env.VITE_API_URL;

const markerLabels = {
  three_ph_transformer: "3 Ph. Transf",
  circuit_breaker: "Circuit Breaker",
  single_ph_transformer: "Single Ph. Transf",
  primary_power_line: "Pri. Pwr Line",
  sec_power_line: "Sec. Pwr Line",
  control_building: "Control Room",
};

async function saveMarkers(payload) {
  try {
    await axiosInstance.post(
      `${VITE_API_URL}/gis/bulk_update/update_markers/`,
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

export async function saveMarkersToDatabase(
  markers,
  selectedSubstation,
  allowEmpty,
  downloadChecked,
  sendToDatabaseChecked
) {
  const user = getCurrentUser();
  if (!user) {
    alert("User is not authenticated.");
    return;
  }

  let userDetails;
  try {
    userDetails = await getUserDetails();
  } catch (error) {
    alert("Failed to fetch user details.");
    return;
  }

  const validMarkers = markers[selectedSubstation.SS_ID].filter((marker) => {
    if (typeof marker.getLabel === "function") {
      const label = marker.getLabel().text;
      return typeof label === "string" && label.trim() !== "";
    } else {
      const label = marker.label.text;

      return typeof label === "string" && label.trim() !== "";
    }
  });

  console.log("validMarkers", markers[selectedSubstation.SS_ID]);

  if (
    (!markers[selectedSubstation.SS_ID] ||
      markers[selectedSubstation.SS_ID].length === 0 ||
      validMarkers.length === 0) &&
    !allowEmpty
  ) {
    console.error("Markers list is empty or contains invalid labels.");
    return;
  }

  const markersToSave = allowEmpty
    ? markers[selectedSubstation.SS_ID]
    : validMarkers;

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
            const labelKey = labelObj.text;
            const attributes = marker.attributes;
            return {
              label: labelKey,
              attributes: attributes,
              color: labelObj.color,
            };
          }),
        },
      },
    ],
  };

  // markersToSave.forEach((marker) => {
  //   const labelKey = getKeyFromValue(markerLabels, marker.getLabel().text);
  //   switch (labelKey) {
  //     case "three_ph_transformer":
  //       geojson.features[0].properties.threePhaseTransformerCount++;
  //       geojson.features[0].properties.totalTransformerCount++;
  //       break;
  //     case "single_ph_transformer":
  //       geojson.features[0].properties.singlePhaseTransformerCount++;
  //       geojson.features[0].properties.totalTransformerCount++;
  //       break;
  //     case "primary_power_line":
  //       geojson.features[0].properties.primaryPowerLineCount++;
  //       break;
  //     case "sec_power_line":
  //       geojson.features[0].properties.secondaryPowerLineCount++;
  //       break;
  //   }
  // });

  if (downloadChecked) {
    saveMarkersAsGeoJSON(geojson, selectedSubstation);
  }

  if (sendToDatabaseChecked) {
    const payload = {
      substation_id: selectedSubstation.SS_ID,
      created_by: userDetails.username, // Use the fetched user details
      updated_by: userDetails.username, // Use the fetched user details
      markers: markersToSave.map((marker) => {
        const labelObj = marker.getLabel();
        const labelKey = labelObj.text;
        const attributes = marker.attributes;
        return {
          label: labelKey,
          latitude: marker.getPosition().lat(),
          attributes: attributes,
          longitude: marker.getPosition().lng(),
        };
      }),
    };
    print("payload", payload);
    await saveMarkers(payload);
  }
}
