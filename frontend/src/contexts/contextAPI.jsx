import React, { createContext, useState, useEffect } from "react";
import axios from "axios";
import useFetchData from "../hooks/useFetchData";

export const AppContext = createContext();

export const ContextProvider = ({ children }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [mapInstance, setMapInstance] = useState(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [transmissionLinesLayer, setTransmissionLinesLayer] = useState(null);
  const [allTransmissionLines, setAllTransmissionLines] = useState([]);
  const [selectedRegion, setSelectedRegion] = useState(null);
  const [selectedSubstation, setSelectedSubstation] = useState(null);
  const [currentMarkers, setCurrentMarkers] = useState([]);
  const [markers, setMarkers] = useState({});
  const [markerMessage, setMarkerMessage] = useState("");
  const [isOverlayVisible, setOverlayVisible] = useState(false);
  const [downloadChecked, setDownloadChecked] = useState(true);
  const [sendToDatabaseChecked, setSendToDatabaseChecked] = useState(true);
  const [allowEmptyChecked, setAllowEmptyChecked] = useState(false);
  const [editSubstationMarkers, setEditSubstationMarkers] = useState(false);
  const [toggleLinesChecked, setToggleLinesChecked] = useState(false);
  const [previousSubstation, setPreviousSubstation] = useState(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [currentMarkerKey, setCurrentMarkerKey] = useState(null);
  // Set temporary state for markers added to the map
  const [tempMarkers, setTempMarkers] = useState(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

    const dataUrl =
      "https://gist.githubusercontent.com/denniesbor/81dfc2b05d3c7ee0f02dfc20ec15dce8/raw/a9c6c3b0f9fdc604d4884ded950b1d5035c90e22/tm_ss_df_gt_300v.json";
    const transmissionLinesUrl =
      "https://gist.githubusercontent.com/denniesbor/f55327cd9a7ba2c7da2725c5b03b17f0/raw/ece03a294a758201597da9c80a50759726425b09/tm_lines_within_ferc.geojson";

    useFetchData(dataUrl, setData);
    useFetchData(transmissionLinesUrl, setAllTransmissionLines);

  const sub_data_url =
    "https://gist.githubusercontent.com/denniesbor/81dfc2b05d3c7ee0f02dfc20ec15dce8/raw/a9c6c3b0f9fdc604d4884ded950b1d5035c90e22/tm_ss_df_gt_300v.json";

  useEffect(() => {
    const fetchData = async () => {
      try {
        const url = sub_data_url;
        const response = await axios.get(url);
        setData(response.data);
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false);
      }
    };

    const checkLoginStatus = () => {
      const token = JSON.parse(localStorage.getItem("user"))?.access;
      setIsLoggedIn(!!token);
    };

    fetchData();
    checkLoginStatus();
  }, []);

  return (
    <AppContext.Provider
      value={{
        dataUrl,
        transmissionLinesUrl,
        data,
        setData,
        loading,
        mapInstance,
        setMapInstance,
        mapLoaded,
        setMapLoaded,
        transmissionLinesLayer,
        setTransmissionLinesLayer,
        allTransmissionLines,
        setAllTransmissionLines,
        selectedRegion,
        selectedSubstation,
        setSelectedRegion,
        setSelectedSubstation,
        currentMarkers,
        setCurrentMarkers,
        markers,
        setMarkers,
        markerMessage,
        setMarkerMessage,
        isOverlayVisible,
        setOverlayVisible,
        downloadChecked,
        setDownloadChecked,
        sendToDatabaseChecked,
        setSendToDatabaseChecked,
        allowEmptyChecked,
        setAllowEmptyChecked,
        editSubstationMarkers,
        setEditSubstationMarkers,
        toggleLinesChecked,
        setToggleLinesChecked,
        previousSubstation,
        setPreviousSubstation,
        username,
        setUsername,
        password,
        setPassword,
        currentMarkerKey,
        setCurrentMarkerKey,
        tempMarkers,
        setTempMarkers,
        isLoggedIn,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};
