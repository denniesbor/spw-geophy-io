import { useContext } from "react";
import { AppContext } from "./contexts/contextAPI";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import MainBody from "./components/MainBody";

function App() {
  const { isLoggedIn } = useContext(AppContext);

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
    </>
  );
}

export default App;
