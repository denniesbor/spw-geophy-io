import React from "react";

const UserInfo = ({ user, handleLogout }) => {
  const capitalizeFirstName = (name) => {
    if (!name) return "";
    return name.charAt(0).toUpperCase() + name.slice(1);
  };


  return (
    <div className="user-info">
      <div className="user-details">
        <i className="fas fa-user user-icon"></i>
        <span className="username">{capitalizeFirstName(user.username)}</span>
      </div>
      <button onClick={handleLogout} className="logout-btn">
        Logout
      </button>
    </div>
  );
};

export default UserInfo;
