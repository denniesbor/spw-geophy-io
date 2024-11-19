import { selectedSubstation } from "./Utils";

export function saveMarkersAsGeoJSON(geojson) {
  // Convert GeoJSON object to string
  const geojsonString = JSON.stringify(geojson, null, 2);

  // Create a Blob from the GeoJSON string
  const blob = new Blob([geojsonString], { type: "application/geo+json" });

  // Create a link element to trigger the download
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${selectedSubstation.REGION}_${
    selectedSubstation.SS_ID
  }_${selectedSubstation.lat.toFixed(4)}_${selectedSubstation.lon.toFixed(
    4
  )}.geojson`; // Set the file name

  // Append the link to the body
  document.body.appendChild(link);

  // Trigger the download
  link.click();

  // Clean up by removing the link
  document.body.removeChild(link);
}
