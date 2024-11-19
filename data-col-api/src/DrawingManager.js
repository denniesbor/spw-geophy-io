export function DrawingManager(map) {
  let drawingManager = new google.maps.drawing.DrawingManager({
    drawingMode: google.maps.drawing.OverlayType.RECTANGLE,
    drawingControl: true,
    drawingControlOptions: {
      position: google.maps.ControlPosition.TOP_CENTER,
      drawingModes: ["rectangle"],
    },
    rectangleOptions: {
      fillColor: "#ffff00",
      fillOpacity: 0.1,
      strokeWeight: 2,
      clickable: false,
      editable: true,
      zIndex: 1,
    },
  });

  drawingManager.setMap(map);

  google.maps.event.addListener(
    drawingManager,
    "rectanglecomplete",
    function (rectangle) {
      if (selectedArea) {
        selectedArea.setMap(null);
      }
      selectedArea = rectangle;
      drawingManager.setDrawingMode(null); // Disable drawing mode after the rectangle is complete
    }
  );
}
