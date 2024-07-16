import React from "react";
import Controls from "./Controls";
import Map from "./Map";
import InfoBox from "./InfoBox";
import MarkerContainer from "./old/MarkerContainer";
import MarkerComponent from "./MarkerComponent";

const MainBody = () => {
  return (
    <>
      <div className="main-body">
        <Controls />
        <InfoBox />
        {/* <MarkerContainer />
         */}
      </div>
      <div className="map-container">
        <MarkerComponent />
        <Map />
      </div>
    </>
  );
};

export default MainBody;
