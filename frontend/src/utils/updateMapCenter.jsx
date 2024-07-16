export function updateMapCenter(
  map,
  selectedSubstation,
  currentMarkers,
  setCurrentMarkers
) {
  const location = {
    lat: parseFloat(selectedSubstation.lat),
    lng: parseFloat(selectedSubstation.lon),
  };

  // Clear existing markers
  currentMarkers.forEach((marker) => marker.setMap(null));
  currentMarkers = [];

  if (map) {
    // Add a red marker at the center location
    const marker = new google.maps.Marker({
      position: location,
      map: map,
      title: "Your Location",
    });

    currentMarkers.push(marker); // Store the marker

    //   Set map center to the selected substation
    map.setCenter(location);

    setCurrentMarkers(currentMarkers); // Update the current markers state
  }
}
