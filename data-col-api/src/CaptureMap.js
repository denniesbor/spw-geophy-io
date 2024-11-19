// Function that captures the map and saves it as an image
function captureMap() {
  if (selectedArea) {
    const bounds = selectedArea.getBounds();
    const ne = bounds.getNorthEast();
    const sw = bounds.getSouthWest();

    // Convert geographic coordinates to pixel coordinates
    const projection = map.getProjection();
    const scale = Math.pow(2, map.getZoom());

    const neWorldPoint = projection.fromLatLngToPoint(ne);
    const swWorldPoint = projection.fromLatLngToPoint(sw);

    const nePixel = new google.maps.Point(
      neWorldPoint.x * scale,
      neWorldPoint.y * scale
    );
    const swPixel = new google.maps.Point(
      swWorldPoint.x * scale,
      swWorldPoint.y * scale
    );

    const mapDiv = document.getElementById("map");
    const mapRect = mapDiv.getBoundingClientRect();

    // Calculate the center pixel coordinates
    const mapCenter = map.getCenter();
    const mapCenterWorldPoint = projection.fromLatLngToPoint(mapCenter);
    const mapCenterPixel = new google.maps.Point(
      mapCenterWorldPoint.x * scale,
      mapCenterWorldPoint.y * scale
    );

    // Calculate offsets for the map container
    const containerStartX = mapCenterPixel.x - mapRect.width / 2;
    const containerStartY = mapCenterPixel.y - mapRect.height / 2;

    // Calculate the startX, startY, width, and height for the clipping area
    const startX = nePixel.x - containerStartX;
    const startY = nePixel.y - containerStartY;
    const width = swPixel.x - nePixel.x;
    const height = swPixel.y - nePixel.y;

    // Capture the entire map using html2canvas
    html2canvas(mapDiv, {
      allowTaint: true,
      useCORS: true,
      scale: window.devicePixelRatio,
    })
      .then(function (canvas) {
        // Create a new canvas to hold the clipped image
        const clippedCanvas = document.createElement("canvas");
        const context = clippedCanvas.getContext("2d");

        // Set the dimensions of the clipped canvas
        clippedCanvas.width = Math.abs(width);
        clippedCanvas.height = Math.abs(height);

        // Draw the clipped image onto the new canvas
        context.drawImage(
          canvas,
          startX, // x-coordinate where to start clipping
          startY, // y-coordinate where to start clipping
          width, // width of the clipped area
          height, // height of the clipped area
          0, // x-coordinate where to place the clipped image on the new canvas
          0, // y-coordinate where to place the clipped image on the new canvas
          Math.abs(width), // width of the clipped image on the new canvas
          Math.abs(height) // height of the clipped image on the new canvas
        );

        // Convert clipped canvas to an image and display or save it
        const link = document.createElement("a");
        link.download = "selected-area-screenshot.png";
        link.href = clippedCanvas.toDataURL();
        link.click();
      })
      .catch((error) => {
        console.error("Error capturing the map area:", error);
      });
  } else {
    alert("Please draw a rectangle first.");
  }
}

export default captureMap;
