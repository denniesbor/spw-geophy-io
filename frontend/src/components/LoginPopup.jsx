import React, { useState, useContext } from "react";
import { login } from "../services/authService";
import { AppContext } from "../contexts/contextAPI";

function LoginPopup({ togglePopup }) {
  const [error, setError] = useState(null);
  const {username, setUsername, password, setPassword} = useContext(AppContext);

  const handleLogin = async () => {
    try {
      const user = await login(username, password);
      togglePopup();
      window.location.reload(); // Reload to update the Navbar
    } catch (err) {
      setError("Login failed. Please check your credentials.");
    }
  };

  const handleUsernameChange = (e) => {
    setUsername(e.target.value);
    if (error) {
      setError(null);
    }
  };

  const handlePasswordChange = (e) => {
    setPassword(e.target.value);
    if (error) {
      setError(null);
    }
  };

  return (
    <div className="popup visible">
      <h2>Login</h2>
      {error && <p className="error">{error}</p>}
      <input
        type="text"
        placeholder="Username"
        value={username}
        onChange={handleUsernameChange}
      />
      <input
        type="password"
        placeholder="Password"
        value={password}
        onChange={handlePasswordChange}
      />
      <button onClick={handleLogin}>Login</button>
      <button onClick={togglePopup}>Cancel</button>
    </div>
  );
}

export default LoginPopup;
