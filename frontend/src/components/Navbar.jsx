import React, { useState, useEffect, useContext } from "react";
import {
  getCurrentUser,
  logout,
  getUserDetails,
} from "../services/authService";
import UserInfo from "./UserInfo";
import LoginPopup from "./LoginPopup";

function Navbar() {
  const [isPopupOpen, setIsPopupOpen] = useState(false);
  const [user, setUser] = useState(null);

  useEffect(() => {
    const fetchUserDetails = async () => {
      const currentUser = getCurrentUser();
      if (currentUser && currentUser.access) {
        try {
          const userDetails = await getUserDetails();
          setUser(userDetails);
        } catch (error) {
          logout();
        }
      }
    };

    fetchUserDetails();
  }, []);

  const togglePopup = () => {
    setIsPopupOpen(!isPopupOpen);
  };

  const handleLogout = () => {
    logout();
    setUser(null);
  };

  return (
    <div className="navbar">
      <h2>geo-spw-io</h2>
      {user ? (
        <UserInfo user={user} handleLogout={handleLogout} />
      ) : (
        <button onClick={togglePopup}>Login</button>
      )}
      {isPopupOpen && <LoginPopup togglePopup={togglePopup} />}
    </div>
  );
}

export default Navbar;
