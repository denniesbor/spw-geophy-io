'''
Module to estimate induced voltages in the transmission line during a geomagnetic storm.
The module leverages the bezpy package.
'''

import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import bezpy
from pysecs import SECS
import numpy as np
import pandas as pd
import random


class PredictBEFields:
    """
    A class to estimate the electric and magnetic fields in the transmission lines.
    
    Parameters:
    -----------
    data_manager : DataManager
        An instance of the DataManager class that manages the data.
    start_time : datetime.datetime, optional
        The start time of the data. If not provided, the first time in the magnetic data will be used.
    end_time : datetime.datetime, optional
        The end time of the data. If not provided, the last time in the magnetic data will be used.
    """
    
    def __init__(self, data_manager, start_time=None, end_time=None,  line_resistivity=0.99):
        self.data_manager = data_manager
        self.transmission_lines = data_manager.tl_data.transmission_lines
        self.filtered_site_xys = data_manager.emtf_data.filtered_site_xys
        self.apply_bezpy()
        self.obsv_xarrays = data_manager.mag_data.obsv_xarrays
        self.filtered_MT_sites = data_manager.emtf_data.filtered_MT_sites
        self.line_resistivity = line_resistivity
        # if start and end time are not given use the first and last time in the magnetic data
        if not start_time:
            self.start_time = data_manager.mag_data.start_time
        if not end_time:
            self.end_time = data_manager.mag_data.end_time
        self.interpolate_B_E_fields()
        self.R_earth = 6371e3 # Earth radius in meters
        self.B_pred = None
        
        # Predict the magnetic field in mt sites
        self.calculate_SECS()
        self.E_pred = np.zeros((len(self.B_obs), len(self.filtered_site_xys), 2))
        self.calculate_E_field()
        self.calculate_voltages(time=random.choice(self.obsv_xarrays["BOU"].Timestamp))
        
    def apply_bezpy(self):
        """
        Apply the bezpy library to the transmission lines dataset.
        """
        self.transmission_lines['obj'] = self.transmission_lines.apply(bezpy.tl.TransmissionLine, axis=1)
        self.transmission_lines["length"] = self.transmission_lines["obj"].apply(lambda x: x.length)
        
        # Apply delaunay triangulation
        self.transmission_lines.obj.apply(lambda x: x.set_delaunay_weights(self.filtered_site_xys))
        print("Done filling interpolation weights: {0} s")

        # Remove lines with bad integration
        E_test = np.ones((1, len(self.filtered_site_xys), 2))

        arr_delaunay = np.zeros(shape=(1, len(self.transmission_lines)))
        for i, tLine in enumerate(self.transmission_lines.obj):
            arr_delaunay[:,i] = tLine.calc_voltages(E_test, how='delaunay')

        # Filter the transmission_lines
        self.transmission_lines = self.transmission_lines[~np.isnan(arr_delaunay[0, :])]
    @staticmethod
    def process_dataset(name, dataset, start_time, end_time):
        data = dataset.loc[{'Timestamp': slice(start_time, end_time)}].interpolate_na('Timestamp')
        
        # No data here... skip ahead
        if len(data['Timestamp']) == 0:
            return None, None, None
        
        data = np.array(data.loc[{'Timestamp': slice(start_time, end_time)}].to_array().T)
        if np.any(np.isnan(data)):
            return None, None, None

        data = data - np.median(data, axis=0)
        return name, (dataset.Latitude, dataset.Longitude), data

    def interpolate_B_E_fields(self):
        """
        Interpolate the magnetic field data and calculate the observed magnetic field.
        """
        obs_xy = []
        B_obs = []

        start_time = self.start_time
        end_time = self.end_time

        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self.process_dataset, name, self.obsv_xarrays[name], start_time, end_time): name
                for name in self.obsv_xarrays
            }

            for future in as_completed(futures):
                name, location, data = future.result()
                if name is not None and location is not None and data is not None:
                    obs_xy.append(location)
                    B_obs.append(data)

        self.obs_xy = np.squeeze(np.array(obs_xy))
        self.B_obs = np.array(B_obs).swapaxes(0, 1)

    def calculate_SECS(self):
        """
        Calculate the SECS output magnetic field.
        """
        if self.obs_xy.shape[0] != self.B_obs.shape[1]:
            raise ValueError("Number of observation points doesn't match B input")

        obs_lat_lon_r = np.zeros((len(self.obs_xy), 3))
        obs_lat_lon_r[:,0] = self.obs_xy[:,0]
        obs_lat_lon_r[:,1] = self.obs_xy[:,1]
        obs_lat_lon_r[:,2] = self.R_earth

        B_std = np.ones_like(self.B_obs)
        B_std[..., 2] = np.inf  # Ignoring Z component

        # specify the SECS grid
        lat, lon, r = np.meshgrid(np.linspace(15,85,36),
                                np.linspace(-175,-25,76),
                                self.R_earth+110000, indexing='ij')
        secs_lat_lon_r = np.hstack((lat.reshape(-1,1),
                                    lon.reshape(-1,1),
                                    r.reshape(-1,1)))

        secs = SECS(sec_df_loc=secs_lat_lon_r)

        secs.fit(obs_loc=obs_lat_lon_r, obs_B=self.B_obs, obs_std=B_std, epsilon=0.05)

        # Create prediction points
        pred_lat_lon_r = np.zeros((len(self.filtered_site_xys), 3))
        pred_lat_lon_r[:,0] = self.filtered_site_xys[:,0]
        pred_lat_lon_r[:,1] = self.filtered_site_xys[:,1]
        pred_lat_lon_r[:,2] = self.R_earth

        self.B_pred = secs.predict_B(pred_lat_lon_r)
    @staticmethod
    def calculate_E_field_for_site(i, site, B_pred_i_0, B_pred_i_1, dt=60):
        Ex, Ey = site.convolve_fft(B_pred_i_0, B_pred_i_1, dt)
        return i, Ex, Ey

    def calculate_E_field(self):
        """
        Calculate the electric field from the magnetic field.
        """
        results = []

        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self.calculate_E_field_for_site, i, site, self.B_pred[:, i, 0], self.B_pred[:, i, 1]): i
                for i, site in enumerate(self.filtered_MT_sites)
            }

            for future in as_completed(futures):
                i, Ex, Ey = future.result()
                results.append((i, Ex, Ey))

        for i, Ex, Ey in results:
            self.E_pred[:, i, 0] = Ex
            self.E_pred[:, i, 1] = Ey
        
    @staticmethod        
    def calculate_voltage_for_line(i, tLine, E_pred):
        voltages = tLine.calc_voltages(E_pred, how='delaunay')
        return i, voltages

    def calculate_voltages(self):
        print("Starting to calculate Vs in transmission lines...")
        n_trans_lines = len(self.transmission_lines)
        mag_times = pd.date_range(start=self.start_time, end=self.end_time, freq='1Min')
        random_time = mag_times[random.randint(0, len(mag_times))]
        
        n_times = len(mag_times)
        arr_delaunay = np.zeros((n_times, n_trans_lines))
        
        for i, tLine in enumerate(self.transmission_lines.obj):
            i, voltages = self.calculate_voltage_for_line(i, tLine, self.E_pred)
            arr_delaunay[:, i] = voltages
            
            if i % 100 == 0:
                print(f"Done calculating Vs of {i} lines,")
        
        df_voltage = pd.DataFrame(index=mag_times, columns=self.transmission_lines.line_id, data=arr_delaunay)
        print("Done with all the Vs calculations.")

        voltages = np.ma.masked_invalid(df_voltage.loc[random_time, :].abs())
        line_E = voltages / (self.line_resistivity * self.transmission_lines['length'].values)
        self.random_time = random_time
        self.transmission_lines['voltage'] = voltages
        self.transmission_lines['E_field'] = line_E