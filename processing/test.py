#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 9 16:59:52 2021

@author: xzy
"""
# %%
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
# %%
io = NWBHDF5IO('EC9_B53.nwb', 'r')
nwbfile_in = io.read()
test_timeseries_in = nwbfile_in.acquisition
trial = nwbfile_in.trials
index = 0
start = trial.start_time[index]
end = trial.stop_time[index]

raw_data = test_timeseries_in["ElectricalSeries"].data
raw_data_read = raw_data[:]

raw_data0 = raw_data[:,index] 

#plot one channel of original data
plt.plot(raw_data[:, index])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (au)')
_ = plt.title('One channel of neural data')
# %%
#Resample
#As indicated in previous research, high gamma band is between 70-170Hz, so 
#we resample data to 500 Hz
sample_rate = test_timeseries_in["ElectricalSeries"].rate # Hz
new_sample_rate = 500
duration = raw_data.shape[0]/sample_rate
rs_data = resample(raw_data_read, new_sample_rate, sample_rate)
t = np.linspace(0, duration, rs_data.shape[0])
#%%
plt.plot(t[int(start*new_sample_rate):int(end*new_sample_rate)],rs_data[int(start*new_sample_rate):int(end*new_sample_rate),0])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (au)')
_ = plt.title('One channel of neural data after resampling')
# %%
#  Notch filtering
# ----------------
# Notch filtering is used to remove the 60 Hz line noise and harmonics on all
# channels.
rs_data1 = rs_data[int(start*new_sample_rate):int(end*new_sample_rate*5),0:64]
nth_data = apply_linenoise_notch(rs_data1, new_sample_rate)
# %%
freq, car_pwr = welch(rs_data[:, 0], fs=new_sample_rate, nperseg=1024)
_, nth_pwr = welch(nth_data[:, 0], fs=new_sample_rate, nperseg=1024)

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)
axs[0].semilogy(freq, car_pwr)
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('Power density (V^2/Hz)')
axs[0].set_xlim([1, 150])
axs[0].set_title('Pre notch filtering')

axs[1].semilogy(freq, nth_pwr)
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_xlim([1, 150])
axs[1].set_title('Post notch filtering')
_ = fig.tight_layout()

# %%
# Estimate of the mean
# across all channels (for each timepoint). This quantity is then subtracted all
# channels. By default, this function takes the mean over the center 95% of
# the electrodes.

car_data = subtract_CAR(nth_data)

plt.plot(t[:500], car_data[:500, 0])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (au)')
_ = plt.title('One channel of data after re-referencing from resample')

# %%
#  Time-frequency decomposition with wavelets
# 
# decompose the neural time series into 6 different frequency subbands
# in the high gamma range using a wavelet transform. The wavelet transform
# amplitude is complex valued and here we take the absolute value.

tf_data, _, ctr_freq, bw = wavelet_transform(car_data, new_sample_rate,
                                             filters='rat', hg_only=True)
# Z scoring the amplitude instead of the complex waveform
tf_data = abs(tf_data)

# %%
num_tf_signals = len(ctr_freq)
fig, axs = plt.subplots(num_tf_signals, 1, sharex=True, sharey=True, figsize=(15, 10))
fig.subplots_adjust(hspace=0.4)
fig.tight_layout()


for idx in range(num_tf_signals):
    sig = tf_data[:, 0, idx]
    axs[idx].plot(t[:5000], sig[:5000])
    axs[idx].set_title('Frequency = {0:.0f} ; Bandwidth = {1:0.0f}'.format(ctr_freq[idx], bw[idx]))
    axs[idx].set_ylabel('Amp. (au)')
# %%
# Normalizing power by zscoring

mean = tf_data[:].mean(axis=0, keepdims=True)
std = tf_data[:].std(axis=0, keepdims=True)
tf_norm_data = (tf_data - mean) / std
high_gamma = tf_norm_data.mean(axis=-1)

# %%
# plot 4 channels of the gamma band
ig, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(10, 10))
fig.subplots_adjust(hspace=0.4)
fig.tight_layout()

for idx in range(4):
    sig = high_gamma[:, idx]
    axs[idx].plot(sig)
    axs[idx].set_title('Channel {0:.0f}'.format(idx))
    axs[idx].set_ylabel('σ')
    axs[idx].set_ylim(-4, 4)
