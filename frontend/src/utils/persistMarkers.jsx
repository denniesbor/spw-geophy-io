import { saveMarkersAsGeoJSON } from "./saveFileUtils";
import { getCurrentUser, getUserDetails } from "../services/authService";
import axiosInstance from "../services/axiosInstance";
import formatData from "./formatMarkers2GeoJSON";

const VITE_API_URL = "https://denniesbor.com";

async function saveMarkers(payload) {
  try {
    const response = await axiosInstance.post(
      `${VITE_API_URL}/gis/bulk_update/update_markers/`,
      payload
    );
    console.log("Markers saved successfully:", response.data);
    return { success: true };
  } catch (error) {
    if (error.response) {
      console.error("Error saving markers:", error.response.data);
      return { success: false, error: error.response.data };
    } else {
      console.error("Error saving markers:", error.message);
      return { success: false, error: error.message };
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

  const { geojson, markersToSave } = formatData(
    markers,
    selectedSubstation,
    allowEmpty
  );

  console.log("GEojson data", geojson);

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

    const result = await saveMarkers(payload);
    if (result.success) {
      alert("Markers saved successfully.");
    } else {
      alert(`Error saving markers: ${result.error}`);
    }
  }
}
