from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile
import numpy as np
from pynwb import TimeSeries
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt 

io = NWBHDF5IO('data/EC9_B53.nwb', 'r')
nwbfile_in = io.read()
test_timeseries_in = nwbfile_in.acquisition
a = test_timeseries_in["ElectricalSeries"].data
# a = np.load('SampleData_class0.npy')
# with open('data/B53_trunc.npy', 'wb') as f:
#     np.save(f, a)

#nwbfile_in.trials.start_time[index] = start
#nwbfile_in.trials.stop_time[index] = end
#nwbfile_in.trials.speak[index] -> true
#nwbfile_in.trials.condition[index] -> syllable
#data = a[int(start*rate), int(end*rate)] -> size different*
#transpose!!!!