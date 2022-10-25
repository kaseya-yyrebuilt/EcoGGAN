#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 22:32:01 2021

@author: xzy
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from pynwb import NWBFile
from pynwb import TimeSeries
from pynwb import NWBHDF5IO

from process_nwb.utils import generate_synthetic_data
from process_nwb.resample import resample
from process_nwb.linenoise_notch import apply_linenoise_notch
from process_nwb.common_referencing import subtract_CAR
from process_nwb.wavelet_transform import wavelet_transform
from sklearn.preprocessing import normalize
#%%
io = NWBHDF5IO('EC9_B53.nwb', 'r')
nwbfile_in = io.read()
test_timeseries_in = nwbfile_in.acquisition
trial = nwbfile_in.trials

raw_data = test_timeseries_in["ElectricalSeries"].data
raw_data_read = raw_data[:]
sample_rate = test_timeseries_in["ElectricalSeries"].rate # Hz
new_sample_rate = 500
duration = raw_data.shape[0]/sample_rate
rs_data = resample(raw_data_read, new_sample_rate, sample_rate)
t = np.linspace(0, duration, rs_data.shape[0])

nth_data = apply_linenoise_notch(rs_data, new_sample_rate)
freq, car_pwr = welch(rs_data[:, 0], fs=new_sample_rate, nperseg=1024)
_, nth_pwr = welch(nth_data[:, 0], fs=new_sample_rate, nperseg=1024)

car_data = subtract_CAR(nth_data)
normed = normalize(car_data, axis=1, norm='l1')
#%%
index = 0
start = trial.start_time[index]
end = trial.stop_time[index]
#condition = trial.condition[index]
a = rs_data[int(start*new_sample_rate):int(end*new_sample_rate),:]
if (len(a)<128):
    a = np.pad(a, [(0, 128-len(a)), (0, 0)],'constant')
b = a[:128,:]
chopped_data = b[:,:, np.newaxis]


for index in range(len(trial)):
    start = trial.start_time[index]
    end = trial.stop_time[index]
    condition = trial.condition[index]
    a = rs_data[int(start*new_sample_rate):int(end*new_sample_rate),:]
    if (len(a)<128):
        a = np.pad(a, [(0, 128-len(a)), (0, 0)],'constant')
    b = a[:128,:,np.newaxis]
    chopped_data = np.append(chopped_data,b,axis = 2)

chopped_data_trans = chopped_data.T
#%%  
np.save('data1.npy', chopped_data)
