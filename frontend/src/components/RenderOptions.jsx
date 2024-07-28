import React from "react";

function RenderOptions({
  selectedMarker,
  editSubstationMarkers,
  details,
  handleInputChange,
  handleOptionChange,
  handleAddMarker,
  allowAddMarker,
  currentMarkerKey,
  markerData,
  markers,
  selectedSubstation,
}) {
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
}

export default RenderOptions;
