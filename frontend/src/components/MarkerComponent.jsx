import React, { useState, useContext, useEffect, useRef } from "react";
import { AppContext } from "../contexts/contextAPI";
import SavePopup from "./SavePopup";
import { useMarkerUtils } from "../hooks/useMarkerUtils";
import { saveMarkersToDatabase } from "../utils/persistMarkers";

// Marker data
const markerData = {
  Transformer: {
    type: ["Single Phase", "Three Phase"],
    role: ["Distribution", "Transmission"],
    transformer_fuel_type: ["Oil", "Dry", "Gas"],
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

//   Map previous markers to new markers
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

// store updated markers
let updatedMarkers = [];

// Marker component
const MarkerComponent = () => {
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

  const [selectedMarker, setSelectedMarker] = useState(null);
  const [details, setDetails] = useState({ name: "", attributes: {} });
  const [mapClickListener, setMapClickListener] = useState(null);
  const markerRefs = useRef({});
  const [allowAddMarker, setAllowAddMarker] = useState(false); // Add state for allowing marker adding

  //   Marker component functions
  const { clearMarkers, createMarker, fetchMarkers } = useMarkerUtils();

  //  fetch and display markers
  useEffect(() => {
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
    setTempMarkers(null);
    setSelectedMarker(null);
  }, [selectedSubstation, editSubstationMarkers]);

  // Use effect to handle singgle marker added
  useEffect(() => {
    // Effect to handle marker removal
    return () => {
      if (tempMarkers) {
        tempMarkers.setMap(null);
      }
    };
  }, [tempMarkers, selectedMarker]);

  //   Fetch and display markers
  const fetchAndDisplayMarkers = async (substationId, dbMarkers) => {
    if (!dbMarkers) {
      dbMarkers = await fetchMarkers(substationId, setMarkerMessage);
    }
    if (dbMarkers) {
      // if markers of selected substaion exist, set null and remove
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

  //   On map click
  const onMapClick = (event) => {
    if (selectedMarker && editSubstationMarkers) {
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

        // Add message that let's a user know that a marker has been added and is temporary unless they fill the
        // attributes and click add marker then save to database or download

        setMarkerMessage(
          `Added ${label}. This is temporary please fill the attributes, click Add Marker button and click Save Markers to persist in a database or download.`
        );

        setTimeout(() => {
          setMarkerMessage("");
        }, 5000);
      }
    }
  };

  //   Marker exists
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

  //   Disable marker adding
  const handleDisableMarkerAdding = () => {
    setMarkerMessage("Marker adding disabled.");
    setSelectedMarker(null);
  };

  //   Save markers
  const handleSaveMarkers = () => {
    if (markers.length === 0 && !allowEmptyChecked) {
      setMarkerMessage(
        "Cannot save because the markers list is empty or contains invalid labels."
      );
    } else {
      setOverlayVisible(true);
    }
  };

  //   Confirm save
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
  };

  //   Cancel save
  const handleCancelSave = () => {
    setOverlayVisible(false);
  };

  //   Handle marker click
  const handleMarkerClick = (marker) => {
    // Reset the marker key
    setCurrentMarkerKey(null);

    setSelectedMarker(marker);
    setAllowAddMarker(false);

    setDetails({
      name: marker,
      attributes: markerData[marker] ? { ...markerData[marker] } : {},
    });
  };

  //   Handle option change
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

  //   Handle input change
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

  // function to update marker click event
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

  //  Add marker
  const handleAddMarker = () => {
    setMarkerMessage(`Selected ${selectedMarker}`);

    if (updatedMarkers.length !== 0) {
      updatedMarkers.forEach((m) => m.setMap(null));
    }

    if (currentMarkerKey) {
      if (tempMarkers && tempMarkers.markerKey === currentMarkerKey) {
        // add tempMarker to markers[selectedSubstation.SS_ID]

        const marker = updateMarkerClickEvent(tempMarkers, selectedSubstation);
        marker.attributes = details.attributes;

        updatedMarkers = markers[selectedSubstation.SS_ID].concat(marker);
        setTempMarkers(null); // clear tempMarkers
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

      markers[selectedSubstation.SS_ID].forEach((m) => m.setMap(null));

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

      setSelectedMarker(null);
      setTempMarkers(null);
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

  const renderOptions = () => {
    if (!selectedMarker || !editSubstationMarkers) return <></>;

    if (selectedMarker === "Other") {
      return (
        <div className="marker-options">
          <h2>{selectedMarker}</h2>
          <input
            type="text"
            name="name"
            placeholder="Name"
            value={details.name || ""}
            onChange={handleInputChange}
            required
          />
          <input
            type="text"
            name="type"
            placeholder="Type"
            value={details.type || ""}
            onChange={handleInputChange}
          />
          <textarea
            name="description"
            placeholder="Description"
            value={details.description || ""}
            onChange={handleInputChange}
          ></textarea>
          <button
            onClick={handleAddMarker}
            disabled={!allowAddMarker || !currentMarkerKey}
            className="markers-btn"
          >
            Add Marker
          </button>
        </div>
      );
    }

    const options = markerData[selectedMarker];
    let attributes = {};

    if (markers && markers[selectedSubstation.SS_ID]) {
      attributes =
        markers[selectedSubstation.SS_ID].find(
          (m) => m.markerKey === currentMarkerKey
        )?.attributes || {};
    }

    return (
      <div className="marker-options">
        <h2>{selectedMarker}</h2>
        {Object.keys(options).map((option) => (
          <div key={option}>
            <label>
              {`${option} ${
                attributes && attributes[option]
                  ? `assigned: ${attributes[option]}`
                  : ""
              }`}
            </label>

            <select
              value={details.attributes[option] || attributes[option] || ""}
              onChange={(e) => handleOptionChange(e, option)}
            >
              <option value="">Select {option}</option>
              {options[option].map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          </div>
        ))}
        <button
          disabled={!allowAddMarker || !currentMarkerKey}
          className="markers-btn"
          onClick={handleAddMarker}
        >
          Add Marker
        </button>
      </div>
    );
  };

  return (
    <div className="marker-component">
      <div className="marker-container">
        <div className="marker-message">
          <p id="markerMessage">{markerMessage}</p>
        </div>
        <div className="marker-list">
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
          <button
            onClick={handleDisableMarkerAdding}
            id="disableMarkerAddingButton"
          >
            Done Adding Markers
          </button>
          <button onClick={handleSaveMarkers} id="saveMarkers">
            Save Markers
          </button>
        </div>
        {selectedMarker && (
          <div className="marker-details">{renderOptions()}</div>
        )}
      </div>

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
