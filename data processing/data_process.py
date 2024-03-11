import os

import numpy as np
from matplotlib import pyplot as plt  # graphic library, for plots
plt.rcParams['figure.figsize'] = [8, 7] #default configuration of plt size [w, h] in inches

from metavision_core.event_io import EventDatReader
from metavision_ml.preprocessing import histo
from metavision_ml.preprocessing.viz import viz_histo

path = "spinner.dat"
from metavision_core.utils import get_sample

get_sample(path, folder=".")

record = EventDatReader(path)
height, width = record.get_size()
print('record dimensions: ', height, width)
start_ts = 1 * 1e6 # in us
record.seek_time(start_ts) # seek the file in the recording after 1 sec

delta_t = 50000 #sampling duration  in us = 50ms
events = record.load_delta_t(delta_t)  # load 50 milliseconds worth of events
events['t'] -= int(start_ts)

tbins=4

volume = np.zeros((tbins, 2, height, width), dtype=np.float32)
histo(events, volume, delta_t)

im = viz_histo(volume[1])
plt.imshow(im)
plt.tight_layout()
plt.title('Histogram', fontsize=20)



from metavision_ml.preprocessing import timesurface
from metavision_ml.preprocessing.viz import filter_outliers

volume = np.zeros((tbins, 2, height, width))

timesurface(events, volume, delta_t, normed=True)

plt.imshow(filter_outliers(volume[1,0], 5))
plt.tight_layout()
plt.colorbar()
plt.title('Linear Time Surface', fontsize=20)
plt.show()

