import React, { useEffect, useContext, useState } from "react";
import { AppContext } from "../contexts/contextAPI";
import { updateMapCenter } from "../utils/updateMapCenter";
// import { getMarkers } from "../utils/getMarkers";
import { toggleTransmissionLines } from "../utils/toggleTransmissionLines";

/**
 * Controls component that handles the selection and interaction of regions, substations, and toggling of transmission lines.
 *
 * @component
 * @returns {JSX.Element} Controls component.
 */
/**
 * Controls component that handles the user interface for selecting regions, substations, and toggling options.
 */
const Controls = () => {
  /**
   * Destructuring the required values from the AppContext.
   */
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

  /**
   * State to track if the component has been initialized.
   */
  const [isInitialized, setIsInitialized] = useState(false);

  /**
   * useEffect hook to initialize the component when data is available and it hasn't been initialized yet.
   */
  useEffect(() => {
    if (data && !isInitialized) {
      initializeFromURL();
      setIsInitialized(true);
    }
  }, [data, isInitialized]);

  /**
   * Updates the URL with the provided parameters.
   *
   * @param {Object} params - The parameters to update the URL with.
   */
  const updateURL = (params) => {
    const searchParams = new URLSearchParams(window.location.search);
    Object.entries(params).forEach(([key, value]) => {
      searchParams.set(key, value);
    });
    const newUrl = `${window.location.pathname}?${searchParams.toString()}`;
    window.history.pushState({ path: newUrl }, "", newUrl);
  };

  /**
   * Initializes the component based on the URL parameters.
   */
  const initializeFromURL = () => {
    const urlParams = new URLSearchParams(window.location.search);
    const regionParam = urlParams.get("region");
    const substationIdParam = urlParams.get("substationid");

    if (regionParam) {
      setSelectedRegion(regionParam);
      populateRegions(regionParam, substationIdParam);
      populateValidationDropdown();
    } else {
      populateRegions(null, substationIdParam);
    }
  };

  /**
   * Populates the regions dropdown based on the data.
   *
   * @param {string} defaultRegion - The default region to select.
   * @param {string} defaultSubstationId - The default substation ID to select.
   */
  const populateRegions = (
    defaultRegion = null,
    defaultSubstationId = null
  ) => {
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

    populateSubstations(selectedRegion, defaultSubstationId);
  };

  /**
   * Populates the substations dropdown based on the selected region.
   *
   * @param {string} selectedRegion - The selected region.
   * @param {string} defaultSubstationId - The default substation ID to select.
   */
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

    // Convert to number if defaultSubstationId is a string
    const convertedId = isNaN(Number(defaultSubstationId))
      ? defaultSubstationId
      : Number(defaultSubstationId);

    let selectedSubstation;
    if (defaultSubstationId) {
      selectedSubstation = substations.find((s) => s.SS_ID === convertedId);
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

  /**
   * Populates the validation dropdown based on the selected region.
   */
  // Implement a simple seeded random number generator
  class SeededRandom {
    constructor(seed) {
      this.seed = seed;
    }

    // Generate a random number between 0 and 1
    random() {
      const x = Math.sin(this.seed++) * 10000;
      return x - Math.floor(x);
    }
  }

  // Use the current date as the seed
  const seed = new Date().getMonth() + new Date().getFullYear() * 12;
  const seededRandom = new SeededRandom(seed);

  const populateValidationDropdown = () => {
    const validationDropdown = document.getElementById("validationDropdown");
    validationDropdown.innerHTML = "";

    // Randomly select 300 substations from all regions using the seeded random
    const selectedSubstations = getRandomSubset(data, 300, seededRandom);

    selectedSubstations.forEach((substation, index) => {
      const option = document.createElement("option");
      option.value = JSON.stringify(substation);
      option.text = `${index + 1}) ${substation.SS_ID} (${substation.REGION})`;
      validationDropdown.add(option);
    });
  };

  // Modified helper function to use the seeded random number generator
  const getRandomSubset = (array, size, seededRandom) => {
    let shuffled = array.slice(0),
      i = array.length,
      temp,
      index;
    while (i--) {
      index = Math.floor(seededRandom.random() * (i + 1));
      temp = shuffled[index];
      shuffled[index] = shuffled[i];
      shuffled[i] = temp;
    }
    return shuffled.slice(0, size);
  };

  /**
   * Event handler for region dropdown change.
   *
   * @param {Event} event - The change event.
   */
  const handleRegionChange = (event) => {
    const selectedRegion = event.target.value;
    setSelectedRegion(selectedRegion);
    populateSubstations(selectedRegion);
    updateURL({ region: selectedRegion });
  };

  /**
   * Event handler for substation dropdown change.
   *
   * @param {Event} event - The change event.
   */
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
    updateURL({
      region: selectedRegion,
      substationid: selectedSubstation.SS_ID,
    });
  };

  /**
   * Event handler for toggle lines checkbox change.
   *
   * @param {Event} event - The change event.
   */
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

  /**
   * Event handler for toggle substation labels checkbox change.
   *
   * @param {Event} event - The change event.
   */
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
      <select
        id="validationDropdown"
        onChange={handleSubstationChange}
      ></select>
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
