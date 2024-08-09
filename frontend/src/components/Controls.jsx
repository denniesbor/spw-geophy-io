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

  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    if (data && !isInitialized) {
      initializeFromURL();
      setIsInitialized(true);
    }
  }, [data, isInitialized]);

  const initializeFromURL = () => {
    const urlParams = new URLSearchParams(window.location.search);
    const regionParam = urlParams.get("region");
    const substationIdParam = urlParams.get("substationid");

    populateRegions(regionParam);

    if (regionParam) {
      setSelectedRegion(regionParam);
      populateSubstations(regionParam, substationIdParam);
    }
  };

  const populateRegions = (defaultRegion = null) => {
    const regionDropdown = document.getElementById("regionDropdown");
    regionDropdown.innerHTML = ""; // Clear existing options
    const regions = [...new Set(data.map((item) => item.REGION))];
    regions.forEach((region) => {
      const option = document.createElement("option");
      option.value = region;
      option.text = region;
      regionDropdown.add(option);
    });

    // Set default region to the URL param, 'PJM' if it exists, or the first available region
    const selectedRegion =
      defaultRegion || (regions.includes("PJM") ? "PJM" : regions[0]);
    regionDropdown.value = selectedRegion;
    setSelectedRegion(selectedRegion);
  };

  const populateSubstations = (selectedRegion, defaultSubstationId = null) => {
    const ssDropdown = document.getElementById("ssDropdown");
    ssDropdown.innerHTML = "";

    const substations = data.filter((item) => item.REGION === selectedRegion);

    substations.forEach((substation, index) => {
      const option = document.createElement("option");
      option.value = JSON.stringify(substation);
      option.text = `${index + 1}) ${substation.SS_ID}`;
      ssDropdown.add(option);
    });

    let selectedSubstation;
    if (defaultSubstationId) {
      selectedSubstation = substations.find(
        (s) => s.SS_ID === defaultSubstationId
      );
      if (selectedSubstation) {
        ssDropdown.value = JSON.stringify(selectedSubstation);
      }
    }

    if (!selectedSubstation && substations.length > 0) {
      selectedSubstation = substations[0];
    }

    if (selectedSubstation) {
      setSelectedSubstation(selectedSubstation);
      updateMapCenter(
        mapInstance,
        selectedSubstation,
        currentMarkers,
        setCurrentMarkers
      );
      updateURL({
        region: selectedRegion,
        substationid: selectedSubstation.SS_ID,
      });
    }
  };

  const updateURL = (params) => {
    const searchParams = new URLSearchParams(window.location.search);
    Object.entries(params).forEach(([key, value]) => {
      searchParams.set(key, value);
    });
    const newUrl = `${window.location.pathname}?${searchParams.toString()}`;
    window.history.pushState({ path: newUrl }, "", newUrl);
  };

  const handleRegionChange = (event) => {
    const selectedRegion = event.target.value;
    setSelectedRegion(selectedRegion);
    populateSubstations(selectedRegion);
    updateURL({ region: selectedRegion });
  };

  const handleSubstationChange = (event) => {
    const selectedSubstation = JSON.parse(event.target.value);

    console.log("selectedSubstation", selectedSubstation);
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
    updateURL({
      region: selectedRegion,
      substationid: selectedSubstation.SS_ID,
    });
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
