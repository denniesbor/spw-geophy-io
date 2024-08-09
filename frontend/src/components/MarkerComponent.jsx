import React, { useState, useContext, useEffect, useRef } from "react";
import { AppContext } from "../contexts/contextAPI";
import SavePopup from "./SavePopup";
import { useMarkerUtils } from "../hooks/useMarkerUtils";
import { saveMarkersToDatabase } from "../utils/persistMarkers";
import RenderOptions from "./RenderOptions";

// Marker data
const markerData = {
  Transformer: {
    type: ["Single Phase", "Three Phase"],
    role: ["Generation", "Transmission"],
    insulative: ["Oil", "Dry", "Gas"],
  },
  "Circuit Breaker": {
    type: ["type1", "type2", "type3", "type4", "type5"],
  },
  "Power Lines": {
    type: ["intra-site", "extra-site"],
  },
  Controls: {
    type: ["Facility", "low voltage", "high voltage", "switchgear"],
  },
  Reactors: {
    type: ["shunt", "air core"],
  },
  "Alt. Energy": {
    type: ["battery", "capacitor", "wind component", "PV System"],
  },
  Other: {
    name: "",
    type: "",
    description: "",
  },
};

// Marker data colors
const markerColors = {
  Transformer: "blue",
  "Circuit Breaker": "green",
  "Power Lines": "purple",
  Controls: "brown",
  Reactors: "orange",
  "Alt. Energy": "yellow",
  Other: "red",
};

// Map previous markers to new markers
const mapPrevMarkers = (prevMarker) => {
  if (prevMarker === "three_ph_transformer") {
    return {
      label: "Transformer",
      color: "blue",
      attributes: {
        type: "Three Phase",
        role: "",
        transformer_fuel_type: "",
      },
    };
  }

  if (prevMarker === "single_ph_transformer") {
    return {
      label: "Transformer",
      color: "yellow",
      attributes: {
        type: "Single Phase",
        role: "",
        transformer_fuel_type: "",
      },
    };
  }

  if (prevMarker === "circuit_breaker") {
    return {
      label: "Circuit Breaker",
      color: "green",
      attributes: { type: "" },
    };
  }
  if (prevMarker === "primary_power_line") {
    return {
      label: "Power Lines",
      color: "purple",
      attributes: { type: "" },
    };
  }

  if (prevMarker === "sec_power_line") {
    return { label: "Power Lines", color: "pink", attributes: { type: "" } };
  }
  if (prevMarker === "control_building") {
    return { label: "Controls", color: "brown", attributes: { type: "" } };
  }
};

// Store updated markers
let updatedMarkers = [];

// Marker component
/**
 * MarkerComponent is a React component that handles the management and display of markers on a map.
 * It uses the AppContext to access the necessary state and functions for marker manipulation.
 *
 * @component
 * @example
 * return (
 *   <MarkerComponent />
 * )
 */
const MarkerComponent = () => {
  // Accessing state and functions from AppContext
  const {
    mapInstance,
    selectedSubstation,
    markers,
    setMarkers,
    editSubstationMarkers,
    allowEmptyChecked,
    setAllowEmptyChecked,
    sendToDatabaseChecked,
    setSendToDatabaseChecked,
    downloadChecked,
    setDownloadChecked,
    isOverlayVisible,
    setOverlayVisible,
    markerMessage,
    setMarkerMessage,
    previousSubstation,
    setPreviousSubstation,
    currentMarkerKey,
    setCurrentMarkerKey,
    tempMarkers,
    setTempMarkers,
  } = useContext(AppContext);

  // State variables
  const [selectedMarker, setSelectedMarker] = useState(null);
  const [details, setDetails] = useState({ name: "", attributes: {} });
  const [mapClickListener, setMapClickListener] = useState(null);
  const markerRefs = useRef({});
  const [allowAddMarker, setAllowAddMarker] = useState(false); // Add state for allowing marker adding
  const [isVisible, setIsVisible] = useState(true); // Close edit marker details
  const [markerAdded, setMarkerAdded] = useState(false);

  // Close marker details
  const closeDetails = () => {
    setIsVisible(false);
    setSelectedMarker(null);
    setAllowAddMarker(false);
    setDetails({ name: "", attributes: {} }); // Reset details
  };

  // Handle marker addition when currentMarkerKey changes
  useEffect(() => {
    if (markerAdded && currentMarkerKey) {
      handleAddMarker();
      setMarkerAdded(false); // Reset the flag
    }
  }, [markerAdded, currentMarkerKey]);

  // Marker utility functions
  const { clearMarkers, createMarker, fetchMarkers } = useMarkerUtils();

  // Fetch and display markers
  useEffect(() => {
    // Close marker details
    closeDetails();

    if (selectedSubstation && editSubstationMarkers) {
      setPreviousSubstation((prev) => {
        if (prev && prev.SS_ID !== selectedSubstation.SS_ID) {
          clearMarkers(prev.SS_ID);
        }
        return selectedSubstation;
      });
      fetchAndDisplayMarkers(selectedSubstation.SS_ID);
    } else if (!editSubstationMarkers && previousSubstation) {
      clearMarkers(previousSubstation.SS_ID);
    }

    if (!editSubstationMarkers) {
      if (selectedSubstation) {
        clearMarkers(selectedSubstation.SS_ID);
      }
    }
    setTempMarkers([]);
    setSelectedMarker(null);
  }, [selectedSubstation, editSubstationMarkers]);

  // Handle single marker added
  useEffect(() => {
    // Effect to handle marker removal
    return () => {
      if (tempMarkers.length > 0) {
        tempMarkers.setMap(null);
      }
    };
  }, [tempMarkers, selectedMarker]);

  // Fetch and display markers
  const fetchAndDisplayMarkers = async (substationId, dbMarkers) => {
    if (!dbMarkers) {
      dbMarkers = await fetchMarkers(substationId, setMarkerMessage);
    }
    if (dbMarkers) {
      // If markers of selected substation exist, set null and remove
      if (markers[substationId]) {
        markers[substationId].forEach((m) => {
          m.setMap(null);
        });
      }

      dbMarkers.forEach((marker) => {
        const label = mapPrevMarkers(marker.label)
          ? mapPrevMarkers(marker.label).label
          : marker.label;
        const attributes = mapPrevMarkers(marker.label)
          ? mapPrevMarkers(marker.label).attributes
          : marker.attributes;
        const location = { lat: marker.latitude, lng: marker.longitude };
        const color = markerColors[label];

        createMarker(
          mapInstance,
          substationId,
          location,
          color,
          label,
          attributes,
          markerRefs,
          "black",
          setCurrentMarkerKey,
          setAllowAddMarker,
          false
        );
      });
    }
  };

  // Handle map click event
  const onMapClick = (event) => {
    if (
      selectedMarker &&
      editSubstationMarkers &&
      allowAddMarker &&
      mapInstance
    ) {
      const location = {
        lat: event.latLng.lat(),
        lng: event.latLng.lng(),
      };

      let color = markerColors[selectedMarker];
      let label = selectedMarker;

      if (!markerExists(selectedSubstation.SS_ID, location)) {
        createMarker(
          mapInstance,
          selectedSubstation.SS_ID,
          location,
          color,
          label,
          {},
          markerRefs,
          "red",
          setCurrentMarkerKey,
          setAllowAddMarker,
          true
        );

        // Add message that lets a user know that a marker has been added and is temporary unless they fill the
        // attributes and click add marker then save to database or download

        setMarkerMessage(
          `Added ${label}. This is temporary. Click save markers to persist in a database or download.`
        );

        setTimeout(() => {
          setMarkerMessage("");
        }, 5000);

        setMarkerAdded(true);
      }
    }
  };

  // Check if marker exists at the given location
  const markerExists = (ss_id, location) => {
    return (
      markers[ss_id] &&
      markers[ss_id].some(
        (marker) =>
          marker.getPosition().lat() === location.lat &&
          marker.getPosition().lng() === location.lng
      )
    );
  };

  // Disable marker adding
  const handleDisableMarkerAdding = () => {
    setMarkerMessage("Marker adding disabled.");
    setSelectedMarker(null);
  };

  // Save markers
  const handleSaveMarkers = () => {
    if (markers.length === 0 && !allowEmptyChecked) {
      setMarkerMessage(
        "Cannot save because the markers list is empty or contains invalid labels."
      );
    } else {
      setOverlayVisible(true);
    }
  };

  // Confirm save
  const handleConfirmSave = async () => {
    await saveMarkersToDatabase(
      markers,
      selectedSubstation,
      allowEmptyChecked,
      downloadChecked,
      sendToDatabaseChecked
    );
    setOverlayVisible(false);
    setMarkerMessage("Markers saved successfully!");

    // Clear markers
    clearMarkers(selectedSubstation.SS_ID);

    // Refetch and populate markers
    fetchAndDisplayMarkers(selectedSubstation.SS_ID);
  };

  // Cancel save
  const handleCancelSave = () => {
    setOverlayVisible(false);
  };

  // Handle marker click
  const handleMarkerClick = (marker) => {
    setIsVisible(true);
    // Reset the marker key
    setCurrentMarkerKey(null);

    setSelectedMarker(marker);
    setAllowAddMarker(false);

    setDetails({
      name: marker,
      attributes: markerData[marker] ? { ...markerData[marker] } : {},
    });
  };

  // Handle option change
  const handleOptionChange = (e, option) => {
    const newAttributes = {
      ...details.attributes,
      [option]: e.target.value,
    };

    // Validate attributes
    let valid = true;
    Object.keys(newAttributes).forEach((key) => {
      if (typeof newAttributes[key] !== "string" || newAttributes[key] === "") {
        valid = false;
      }
    });
    setDetails((prevDetails) => ({
      ...prevDetails,
      attributes: newAttributes,
    }));

    setAllowAddMarker(valid);
  };

  // Handle input change
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setDetails((prevDetails) => ({
      ...prevDetails,
      [name]: value,
    }));

    let valid = true;
    Object.keys(details.attributes).forEach((key) => {
      if (typeof details[key] !== "string" || details[key] === "") {
        valid = false;
      }
    });
    setAllowAddMarker(valid);
  };

  // Update marker click event
  const updateMarkerClickEvent = (marker, selectedSubstation) => {
    // Remove existing listeners
    new google.maps.event.clearListeners(marker, "dragend");
    new google.maps.event.clearListeners(marker, "rightclick");
    new google.maps.event.clearListeners(marker, "click");

    // Add updated listeners
    marker.addListener("dragend", function (event) {
      marker.position.lat = event.latLng.lat();
      marker.position.lng = event.latLng.lng();
    });

    marker.addListener("rightclick", function () {
      marker.setMap(null);
      const updatedMarkers = {
        ...markers,
        [selectedSubstation.SS_ID]: markers[selectedSubstation.SS_ID].filter(
          (m) =>
            m.getPosition().lat() !== marker.getPosition().lat() ||
            m.getPosition().lng() !== marker.getPosition().lng()
        ),
      };
      setAllowAddMarker(false);
      setMarkers(updatedMarkers);
    });

    marker.addListener("click", function () {
      const markerLabel = marker.getLabel().text;
      const markerDiv = markerRefs.current[markerLabel];
      if (markerDiv) {
        markerDiv.click();
        setCurrentMarkerKey(marker.markerKey);
      }
    });

    return marker;
  };

  // Add marker
  const handleAddMarker = () => {
    setMarkerMessage(`Selected ${selectedMarker}`);

    if (updatedMarkers.length !== 0) {
      updatedMarkers.forEach((m) => m.setMap(null));
    }

    if (currentMarkerKey) {
      if (tempMarkers && tempMarkers.markerKey === currentMarkerKey) {
        // Add tempMarker to markers[selectedSubstation.SS_ID]
        const marker = updateMarkerClickEvent(tempMarkers, selectedSubstation);
        marker.attributes = details.attributes;

        updatedMarkers = markers[selectedSubstation.SS_ID]
          ? [...markers[selectedSubstation.SS_ID], marker]
          : [marker];

        setTempMarkers([]); // Clear tempMarkers
      } else {
        // Create a copy of the markers array for the selected substation
        updatedMarkers = markers[selectedSubstation.SS_ID].map((marker) => {
          if (marker.markerKey === currentMarkerKey) {
            marker.attributes = details.attributes;

            marker = updateMarkerClickEvent(marker, selectedSubstation);

            return marker;
          }
          return marker;
        });
      }

      if (markers[selectedSubstation.SS_ID]) {
        markers[selectedSubstation.SS_ID].forEach((m) => {
          if (m && typeof m.setMap === "function") {
            m.setMap(null);
          } else {
            console.error("Invalid marker or setMap function missing:", m);
          }
        });
      }

      // Update the state with the new markers array
      setMarkers((prevMarkers) => ({
        ...prevMarkers,
        [selectedSubstation.SS_ID]: updatedMarkers,
      }));

      if (mapInstance) {
        updatedMarkers.forEach((marker) => {
          marker.setMap(mapInstance);
        });
      }

      setTempMarkers([]);
    }
  };

  useEffect(() => {
    if (mapInstance && editSubstationMarkers) {
      if (mapClickListener) {
        google.maps.event.removeListener(mapClickListener);
      }
      const listener = mapInstance.addListener("click", onMapClick);
      setMapClickListener(listener);
      return () => {
        if (mapClickListener) {
          google.maps.event.removeListener(mapClickListener);
        }
      };
    }
  }, [mapInstance, selectedMarker, editSubstationMarkers, allowAddMarker]);

  const isSaveButtonDisabled =
    !selectedSubstation ||
    !selectedSubstation.SS_ID ||
    !markers[selectedSubstation.SS_ID] ||
    markers[selectedSubstation.SS_ID].length === 0;

  return (
    <div className="marker-component">
      <div className="marker-container">
        <div className="marker-message">
          <p id="markerMessage">{markerMessage}</p>
        </div>
        <div className="marker-list">
          {/* Render marker list */}
          {Object.keys(markerData).map((marker) => (
            <div
              key={marker}
              ref={(el) => (markerRefs.current[marker] = el)}
              className={`marker-item ${
                selectedMarker === marker ? "selected" : ""
              }`}
              onClick={() => handleMarkerClick(marker)}
            >
              {marker}
            </div>
          ))}
          {/* Button to disable marker adding */}
          <button
            onClick={handleDisableMarkerAdding}
            id="disableMarkerAddingButton"
          >
            Done Adding Markers
          </button>
          {/* Button to save markers */}
          <button
            onClick={handleSaveMarkers}
            id="saveMarkers"
            disabled={isSaveButtonDisabled}
          >
            Save Markers
          </button>
        </div>
        {/* Render marker details */}
        {isVisible && (
          <div className="marker-details">
            <button className="close-btn" onClick={closeDetails}>
              <i className="fas fa-times"></i>
            </button>
            <RenderOptions
              selectedMarker={selectedMarker}
              editSubstationMarkers={editSubstationMarkers}
              details={details}
              handleInputChange={handleInputChange}
              handleOptionChange={handleOptionChange}
              handleAddMarker={handleAddMarker}
              allowAddMarker={allowAddMarker}
              currentMarkerKey={currentMarkerKey}
              markerData={markerData}
              markers={markers}
              selectedSubstation={selectedSubstation}
            />
          </div>
        )}
      </div>

      {/* Save popup */}
      <SavePopup
        isVisible={isOverlayVisible}
        downloadChecked={downloadChecked}
        setDownloadChecked={setDownloadChecked}
        sendToDatabaseChecked={sendToDatabaseChecked}
        setSendToDatabaseChecked={setSendToDatabaseChecked}
        allowEmptyChecked={allowEmptyChecked}
        setAllowEmptyChecked={setAllowEmptyChecked}
        onConfirmSave={handleConfirmSave}
        onCancelSave={handleCancelSave}
        markers={markers}
        isAutthenticated={true}
      />
    </div>
  );
};

export default MarkerComponent;
