import React, { useEffect, useState } from "react";
import loadGoogleMaps from "./loadGoogleMaps";

const withGoogleMaps = (WrappedComponent, apiKey) => {
  return (props) => {
    const [isLoaded, setIsLoaded] = useState(false);

    useEffect(() => {
      loadGoogleMaps(apiKey)
        .then(() => setIsLoaded(true))
        .catch((error) =>
          console.error("Error loading Google Maps API:", error)
        );
    }, [apiKey]);

    return isLoaded ? <WrappedComponent {...props} /> : <div>Loading...</div>;
  };
};

export default withGoogleMaps;
