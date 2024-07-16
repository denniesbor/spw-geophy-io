import { useContext, useEffect, useState } from "react";
import { AppContext } from "./contexts/contextAPI";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import useFetchData from "./hooks/useFetchData";
import MainBody from "./components/MainBody";

function App() {
  const { setData, setAllTransmissionLines } = useContext(AppContext);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    // Check if user is logged in by looking for a specific item in localStorage
    const token = JSON.parse(localStorage.getItem("user"))?.access;
    setIsLoggedIn(!!token);
  }, []);
  const dataUrl =
    "https://gist.githubusercontent.com/denniesbor/81dfc2b05d3c7ee0f02dfc20ec15dce8/raw/a9c6c3b0f9fdc604d4884ded950b1d5035c90e22/tm_ss_df_gt_300v.json";
  const transmissionLinesUrl =
    "https://gist.githubusercontent.com/denniesbor/f55327cd9a7ba2c7da2725c5b03b17f0/raw/ece03a294a758201597da9c80a50759726425b09/tm_lines_within_ferc.geojson";

  useFetchData(dataUrl, setData);
  useFetchData(transmissionLinesUrl, setAllTransmissionLines);

  return (
    <>
      <Navbar />
      {isLoggedIn ? (
        <>
          <MainBody />
          <Footer />
        </>
      ) : (
        <div className="login-message">
          <h2>Please log in to access the application</h2>
        </div>
      )}
      {/* <MainBody />
      <Footer /> */}
    </>
  );
}

export default App;
