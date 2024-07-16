import React, { useEffect, useContext, useState } from "react";
import { AppContext } from "../contexts/contextAPI";
import { updateMapCenter } from "../utils/updateMapCenter";
// import { getMarkers } from "../utils/getMarkers";
import { toggleTransmissionLines } from "../utils/toggleTransmissionLines";

const Controls = () => {
  const {
    data,
    mapInstance,
    transmissionLinesLayer,
    allTransmissionLines,
    selectedRegion,
    setSelectedRegion,
    selectedSubstation,
    setSelectedSubstation,
    currentMarkers,
    setCurrentMarkers,
    setTransmissionLinesLayer,
    setEditSubstationMarkers,
    toggleLinesChecked,
    setToggleLinesChecked,
  } = useContext(AppContext);

  useEffect(() => {
    if (data) {
      populateRegions();
    }
  }, [data]);

  const populateRegions = () => {
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
    setSelectedRegion(defaultRegion);
    populateSubstations(defaultRegion);
  };

  const populateSubstations = (selectedRegion) => {
    const ssDropdown = document.getElementById("ssDropdown");
    ssDropdown.innerHTML = "";

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
      const firstSubstation = JSON.parse(ssDropdown.options[0].value);
      setSelectedSubstation(firstSubstation);
      updateMapCenter(
        mapInstance,
        firstSubstation,
        currentMarkers,
        setCurrentMarkers
      );
    }
  };

  const handleRegionChange = (event) => {
    const selectedRegion = event.target.value;
    setSelectedRegion(selectedRegion);
    populateSubstations(selectedRegion);
  };

  const handleSubstationChange = (event) => {
    const selectedSubstation = JSON.parse(event.target.value);
    setSelectedSubstation(selectedSubstation);
    updateMapCenter(
      mapInstance,
      selectedSubstation,
      currentMarkers,
      setCurrentMarkers
    );
    if (toggleLinesChecked) {
      toggleTransmissionLines(
        selectedSubstation.connected_tl_id,
        transmissionLinesLayer,
        allTransmissionLines,
        mapInstance,
        true,
        setTransmissionLinesLayer
      );
    }
  };

  const handleToggleLines = (event) => {
    const isChecked = event.target.checked;
    setToggleLinesChecked(isChecked);
    toggleTransmissionLines(
      selectedSubstation.connected_tl_id,
      transmissionLinesLayer,
      allTransmissionLines,
      mapInstance,
      isChecked,
      setTransmissionLinesLayer
    );
  };

  const handleToggleSubstationLabels = (event) => {
    if (event.target.checked === true) {
      setEditSubstationMarkers(true);
    } else {
      setEditSubstationMarkers(false);
    }
  };

  return (
    <div id="controls">
      <select id="regionDropdown" onChange={handleRegionChange}></select>
      <select id="ssDropdown" onChange={handleSubstationChange}></select>
      <label>
        <input type="checkbox" id="toggleLines" onChange={handleToggleLines} />{" "}
        Show Transmission Lines
      </label>
      <label>
        <input
          type="checkbox"
          id="toggleSubstationLabels"
          onChange={handleToggleSubstationLabels}
        />{" "}
        Add/Edit Substation Labels
      </label>
      {/* <button id="captureButton">Capture Map</button> */}
    </div>
  );
};

export default Controls;
