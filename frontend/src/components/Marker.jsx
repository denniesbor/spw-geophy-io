// components/Marker.jsx
import React from "react";

const Marker = ({ id, label, onClick }) => (
  <div className="marker" id={id} onClick={() => onClick(id)}>
    {label}
  </div>
);

export default Marker;
