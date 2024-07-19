import React, {useContext} from "react";
import Controls from "./Controls";
import Map from "./Map";
import InfoBox from "./InfoBox";
import MarkerComponent from "./MarkerComponent";
import { AppContext } from "../contexts/contextAPI";

const MainBody = () => {

  const {isLoggedIn} = useContext(AppContext);
  return (
    <>
    {
      isLoggedIn ? (
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
      ) : (
        <div className="login-message">
          <h2>Please log in to access the application</h2>
        </div>

      )
    }
    </>


  );
};

export default MainBody;
