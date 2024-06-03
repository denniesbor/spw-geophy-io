// Markers component
import { saveMarkersAsGeoJSON } from "./WriteFile";

let isMarkerAddingEnabled = true;
let markers = []; // Array to hold all markers

export function getMarkers(map) {
  // Object to hold the colors for each type of marker
  let markerColors = {
    three_ph_transformer: "blue",
    circuit_breaker: "green",
    single_ph_transformer: "yellow",
    primary_power_line: "purple",
    sec_power_line: "pink",
    control_building: "brown",
  };

  map.addListener("click", function (event) {
    // if (!isMarkerAddingEnabled) {
    //   return; // If marker adding is disabled, exit the function
    // }

    const location = {
      lat: event.latLng.lat(),
      lng: event.latLng.lng(),
    };

    // Get the selected item from the div
    let selectedItem = document.querySelector(".marker.selected");

    if (selectedItem) {
      // Get the color for the selected item
      let color = markerColors[selectedItem.id];

      // Label
      let label = selectedItem.id;

      // Create a marker
      let marker = new google.maps.Marker({
        position: location,
        map: map,
        draggable: true, // This allows the user to move the marker
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          fillColor: color,
          fillOpacity: 1,
          strokeWeight: 0,
          scale: 10,
        },
        label: label,
      });

      // Add the marker to the array
      markers.push(marker);

      // Add an event listener for the dragend event on the marker
      marker.addListener("dragend", function (event) {
        // Update the location object with the new coordinates
        location.lat = event.latLng.lat();
        location.lng = event.latLng.lng();
      });

      // Add an event listener for the rightclick event on the marker to remove it
      marker.addListener("rightclick", function () {
        marker.setMap(null); // Remove the marker from the map
        markers = markers.filter((m) => m !== marker); // Remove the marker from the array
      });
    }
  });

  // Add click event to the marker divs
  let markerDivs = document.querySelectorAll(".marker");

  markerDivs.forEach((div) => {
    div.addEventListener("click", function () {
      // Check if the selected marker is clicked again
      if (this.classList.contains("selected")) {
        // Disable all markers
        markerDivs.forEach((div) => {
          div.classList.add("disabled");
        });

        return; // Exit the function
      }

      // Remove selected class from all divs
      markerDivs.forEach((div) => div.classList.remove("selected"));

      // Add selected class to the clicked div
      this.classList.add("selected");
    });
  });

  document
    .getElementById("disableMarkerAddingButton")
    .addEventListener("click", disableMarkerSelection);

  document
    .getElementById("saveMarkers")
    .addEventListener("click", () => saveMarkersAsGeoJSON(markers));
}

// Modify the function to disable marker selection
function disableMarkerSelection() {
  let toggleButton = document.getElementById("disableMarkerAddingButton");
  toggleButton.textContent = "Done Adding Markers";

  // Deselect the selected marker
  let selectedMarker = document.querySelector(".marker.selected");
  if (selectedMarker) {
    selectedMarker.classList.remove("selected");
  }
}

// Function to save the markers to localStorage
