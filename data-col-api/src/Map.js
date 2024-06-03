// map.js
let map;

export function initMap() {
  map = new google.maps.Map(document.getElementById("map"), {
    center: {
      lat: 37.7893719,
      lng: -122.3942,
    },
    zoom: 16,
    heading: 320,
    tilt: 0,
    mapId: "90f87356969d889c",
  });

  return map;
}

export function getMap() {
  return initMap();
}
