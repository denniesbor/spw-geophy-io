import React, { useContext } from "react";
import axios from "axios";
import axiosInstance from "../services/axiosInstance";
import { AppContext } from "../contexts/contextAPI";

function Footer() {
  const { dataUrl, transmissionLinesUrl } = useContext(AppContext);

  const handleDownload = async (url, filename, useAxiosInstance = false) => {
    try {
      const response = await (useAxiosInstance ? axiosInstance : axios).get(
        url,
        {
          responseType: "blob",
        }
      );

      // Create a link element
      const blobUrl = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = blobUrl;
      link.setAttribute("download", filename); // Set the download attribute with a filename
      document.body.appendChild(link);
      link.click(); // Programmatically click the link to trigger the download

      // Clean up and remove the link
      link.parentNode.removeChild(link);
    } catch (error) {
      console.error("Error downloading the CSV file", error);
    }
  };

  return (
    <div className="footer">
      <h3>Mapping Transmission Lines and Substations in the United States</h3>

      {/* <div className="download-data">
        <div className="download-grid-mapping">
          <button
            onClick={() =>
              handleDownload(
                "/gis/substations/export-csv/",
                "grid_mapping.csv",
                true
              )
            }
          >
            Download Grid Mapping
          </button>
        </div>

        <div className="download-substation">
          <button onClick={() => handleDownload(dataUrl, "substations.csv")}>
            Download Substations CSV
          </button>
        </div>

        <div className="download-transmission-lines">
          <button
            onClick={() =>
              handleDownload(transmissionLinesUrl, "transmission_lines.csv")
            }
          >
            Download Transmission Lines
          </button>
        </div>
      </div> */}
    </div>
  );
}

export default Footer;
