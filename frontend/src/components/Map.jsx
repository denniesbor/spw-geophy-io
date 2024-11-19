import React, { useEffect, useContext, useState } from "react";
import { AppContext } from "../contexts/contextAPI";
import withGoogleMaps from "../utils/withGoogleMaps";

const GOOGLE_MAPS_API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;

function Map() {
  const { selectedSubstation, setMapInstance, setMapLoaded } =
    useContext(AppContext);
  const [map, setMap] = useState(null);

  useEffect(() => {
    if (typeof google !== "undefined" && map === null && selectedSubstation) {
      initMap({ lat: selectedSubstation.lat, lng: selectedSubstation.lon });
    }
  }, [map, selectedSubstation]);

  const initMap = (location) => {
    const map = new google.maps.Map(document.getElementById("map"), {
      center: location,
      zoom: 15,
      mapTypeId: "satellite",
    });

    const marker = new google.maps.Marker({
      position: location,
      map: map,
      title: selectedSubstation.SS_NAME,
    });

    setMap(map);
    setMapInstance(map);
    setMapLoaded(true);
  };

  // Ensure initMap is available globally for the callback
  useEffect(() => {
    if (selectedSubstation) {
      window.initMap = () =>
        initMap({ lat: selectedSubstation.lat, lng: selectedSubstation.lon });
    }
  }, [selectedSubstation]);

  return (
    <div
      id="map"
      style={{ width: "100%", height: "70vh", position: "relative" }}
    ></div>
  );
}

export default withGoogleMaps(Map, GOOGLE_MAPS_API_KEY);
