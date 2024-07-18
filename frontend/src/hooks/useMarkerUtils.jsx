// utils/MarkerUtils.js
import { useContext } from "react";
import { AppContext } from "../contexts/contextAPI";
import axiosInstance from "../services/axiosInstance";

let markersObject = {};

export function useMarkerUtils() {
  const { markers, setMarkers, setTempMarkers, setMarkerMessage, tempMarkers } =
    useContext(AppContext);

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
      setMarkers({ ...markersObject });
    }
  }

  // Clear markers from the map and the state
  function clearMarkers(ss_id) {
    if (markersObject[ss_id]) {
      markersObject[ss_id].forEach((marker) => marker.setMap(null));
      delete markersObject[ss_id];
      setMarkers({ ...markersObject });
    }

    if (markers[ss_id]) {
      markers[ss_id].forEach((marker) => marker.setMap(null));
      delete markers[ss_id];
      setMarkers({ ...markers });
    }
  }

  // Create a marker on the map
  function createMarker(
    map,
    ss_id,
    location,
    color,
    label,
    attributes,
    markerRefs,
    labelColor,
    setCurrentMarkerKey,
    setAllowAddMarker,
    isTemp = false
  ) {
    labelColor = attributes.type ? labelColor : "red";

    if (label === "Transformer") {
      labelColor = attributes.role === "" ? "red" : "black";
    }

    const marker = new google.maps.Marker({
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
        fontSize: "12px",
      },
    });

    marker.attributes = attributes;
    const markerKey = `${label}-${location.lat}-${location.lng}`;
    marker.markerKey = markerKey;
    setCurrentMarkerKey(markerKey);

    marker.addListener("dragend", function (event) {
      location.lat = event.latLng.lat();
      location.lng = event.latLng.lng();
    });

    marker.addListener("click", function () {
      const markerLabel = marker.getLabel().text;
      const markerDiv = markerRefs.current[markerLabel];
      if (markerDiv) {
        markerDiv.click();
        setCurrentMarkerKey(marker.markerKey);
      }
    });

    marker.addListener("rightclick", function () {
      marker.setMap(null);
      if (isTemp) {
        setTempMarkers(null);
        // set time out showing unsaved marker removed, then messahge disappears
        setMarkerMessage("Unsaved Marker removed.");
        setTimeout(() => {
          setMarkerMessage("");
        }, 1000);
      } else {
        const updatedMarkers = {
          ...markersObject,
          [ss_id]: markersObject[ss_id].filter(
            (m) => m.markerKey !== marker.markerKey
          ),
        };
        setMarkers(updatedMarkers);
      }
      setAllowAddMarker(false);
    });

    // If temporary marker exists, remove it to avoid duplicates or adding twice to the map
    if (isTemp) {
      // Set the new temporary marker
      setTempMarkers(marker);
      return;
    }

    addMarker(ss_id, marker);
  }

  async function fetchMarkers(substationId, setMarkerMessage) {
    try {
      const response = await axiosInstance.get(
        `https://denniesbor.com/gis/markers/?substation=${substationId}`
      );

      let markerData = response.data;

      if (markerData.length === 0) {
        setMarkerMessage("No markers available.");
      } else {
        const { created_by, updated_by } = markerData[0];
        setMarkerMessage(
          `Markers available. Created by: ${created_by}, Updated by: ${updated_by}`
        );
        return markerData;
      }
    } catch (error) {
      setMarkerMessage("No markers available.");
    }
  }

  return {
    addMarker,
    clearMarkers,
    createMarker,
    fetchMarkers,
  };
}
