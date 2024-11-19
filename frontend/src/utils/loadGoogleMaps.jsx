let googleMapsPromise;

const loadGoogleMaps = (apiKey) => {
  if (!googleMapsPromise) {
    googleMapsPromise = new Promise((resolve, reject) => {
      const existingScript = document.querySelector(
        `script[src*="maps.googleapis.com"]`
      );
      if (existingScript) {
        resolve();
        return;
      }

      const script = document.createElement("script");
      script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=geometry,drawing,places`;
      script.async = true;
      script.defer = true;
      script.onload = () => {
        if (typeof google !== "undefined") {
          resolve();
        } else {
          reject(new Error("Google Maps API could not be loaded."));
        }
      };
      script.onerror = reject;
      document.body.appendChild(script);
    });
  }
  return googleMapsPromise;
};

export default loadGoogleMaps;
