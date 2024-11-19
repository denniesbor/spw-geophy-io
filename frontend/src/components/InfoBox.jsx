import React, { useContext, useEffect, useState } from "react";
import { AppContext } from "../contexts/contextAPI";
import { voltageColorMapping, getColor } from "../utils/utils";

const InfoBox = () => {
  const { selectedSubstation, toggleLinesChecked } = useContext(AppContext);
  const [lineVoltages, setLineVoltages] = useState([]);

  useEffect(() => {
    if (selectedSubstation) {
      updateInfoBox(selectedSubstation);
    }
  }, [selectedSubstation]);

  const updateInfoBox = (substation) => {
    setLineVoltages(substation.LINE_VOLTS);
  };

  return (
    <div id="infoBox" className="info-box">
      <div className="sub-info-box">
        <p>
          <strong>Substation Information:</strong>
        </p>

        {selectedSubstation?.SS_NAME && (
          <p>
            <strong>Name:</strong>{" "}
            <span id="ssName">{selectedSubstation?.SS_NAME}</span>
          </p>
        )}

        {selectedSubstation?.SS_OPERATOR && (
          <p>
            <strong>Operator:</strong>{" "}
            <span id="ssOperator">{selectedSubstation?.SS_OPERATOR}</span>
          </p>
        )}
        {selectedSubstation?.SS_VOLTAGE && (
          <p>
            <strong>Voltages:</strong>{" "}
            <span id="ssVoltages">{selectedSubstation?.SS_VOLTAGE}</span>
          </p>
        )}
      </div>
      {toggleLinesChecked && (
        <div className="line-info-box">
          <p>
            <strong>Connected Line Voltages:</strong>
            <span id="lineVoltages">
              {lineVoltages.map((voltage, index) => (
                <React.Fragment key={index}>
                  <span
                    style={{
                      display: "inline-block",
                      width: "10px",
                      height: "10px",
                      backgroundColor: getColor(voltage),
                      marginRight: "5px",
                    }}
                  ></span>
                  <span style={{ color: "black" }}>{voltage}</span>
                  {index < lineVoltages.length - 1 && <span>, </span>}
                </React.Fragment>
              ))}
            </span>
          </p>
        </div>
      )}
    </div>
  );
};

export default InfoBox;
