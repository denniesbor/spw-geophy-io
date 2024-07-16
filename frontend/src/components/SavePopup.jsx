// components/SavePopup.jsx
import React from "react";

const SavePopup = ({
  isVisible,
  downloadChecked,
  setDownloadChecked,
  sendToDatabaseChecked,
  setSendToDatabaseChecked,
  allowEmptyChecked,
  setAllowEmptyChecked,
  onConfirmSave,
  onCancelSave,
  markers,
}) => (
  <>
    {isVisible && (
      <>
        <div id="overlay" className="overlay"></div>
        <div id="savePopup" className="popup" style={{display:"block"}}>
          <h3>Save Options</h3>
          <div>
            <input
              type="checkbox"
              id="downloadCheckbox"
              checked={downloadChecked}
              onChange={(e) => setDownloadChecked(e.target.checked)}
            />
            <label htmlFor="downloadCheckbox">Download JSON File</label>
          </div>
          <div>
            <input
              type="checkbox"
              id="sendToDatabaseCheckbox"
              checked={sendToDatabaseChecked}
              onChange={(e) => setSendToDatabaseChecked(e.target.checked)}
            />
            <label htmlFor="sendToDatabaseCheckbox">Send to Database</label>
          </div>
          {markers.length === 0 && (
            <div id="allowEmptyContainer">
              <label htmlFor="allowEmptyCheckbox">Allow Save When Empty:</label>
              <input
                type="checkbox"
                id="allowEmptyCheckbox"
                checked={allowEmptyChecked}
                onChange={(e) => setAllowEmptyChecked(e.target.checked)}
              />
            </div>
          )}
          <button
            id="confirmSaveButton"
            onClick={onConfirmSave}
            disabled={markers.length === 0 && !allowEmptyChecked}
          >
            Save
          </button>
          <button id="cancelButton" onClick={onCancelSave}>
            Cancel
          </button>
        </div>
      </>
    )}
  </>
);

export default SavePopup;
