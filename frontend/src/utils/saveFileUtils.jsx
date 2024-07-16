// utils/SaveFileUtils.js
export function saveMarkersAsGeoJSON(geojson, selectedSubstation) {
  const geojsonString = JSON.stringify(geojson, null, 2);
  const blob = new Blob([geojsonString], { type: "application/geo+json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${selectedSubstation.REGION}_${
    selectedSubstation.SS_ID
  }_${selectedSubstation.lat.toFixed(4)}_${selectedSubstation.lon.toFixed(
    4
  )}.geojson`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
