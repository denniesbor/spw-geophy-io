import { useState, useEffect, useContext } from "react";
import axios from "axios";

const useFetchData = (url, setData) => {
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(url);
        setData(response.data);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };
    fetchData();
  }, [url, setData]);
};

export default useFetchData;
