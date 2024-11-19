// colors mapping transmission voltage class
export const voltageColorMapping = {
  345: "red",
  115: "green",
  230: "blue",
  161: "yellow",
  500: "purple",
  100: "orange",
  138: "pink",
  765: "brown",
  120: "cyan",
  220: "magenta",
};

// Function to get color based on voltage
export function getColor(voltage) {
  return voltageColorMapping[voltage] || "black"; // default to black if no match
}

// Function to check if Street View exists and toggle it
export function checkStreetView(location, panorama) {
  const sv = new google.maps.StreetViewService();

  sv.getPanorama({ location: location, radius: 50 }, function (data, status) {
    if (status === "OK") {
      panorama.setPano(data.location.pano);
      panorama.setPov({
        heading: 270, // Adjust heading as needed
        pitch: 0,
      });
      panorama.setVisible(true);
    } else {
      panorama.setVisible(false);
    }
  });
}

// Get key from labels
export const getKeyFromValue = (object, value) =>
  Object.keys(object).find((key) => object[key] === value);
