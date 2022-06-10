import pyrealtime as prt
import serial
import numpy as np
import struct

from controller_lib import get_device_data
from segmentation2 import segment
import time

BYTES_PER_SAMPLE = 3
RX_CHANNELS_ADC = 3
TX_CHANNELS = 3

BAUD = 960000
FS = 136
FILTER_CUTOFF = 18
FILTER_ORDER = 15

SEGMENT = False
FILTER = False







@prt.transformer
def analyze(data):
    print(np.var(data), np.var(data)/np.mean(data))

@prt.transformer
def ratios(data):
    return np.array([data[:,1] / data[:,0], data[:,2] / data[:,0]]).T


def main():
    data = get_device_data()
    # prt.RecordLayer(data)
    # analyze(prt.AggregateLayer(data, flush_counter=100, empty_on_flush=True, in_place=True))
    # prt.PrintLayer(data)
    # data_filt = prt.ExponentialFilter(data, alpha=1, batch=True)
    prt.TimePlotLayer(data, window_size=5000, n_channels=3, ylim=(-1.2, 1.2), lw=1)
    # prt.TimePlotLayer(ratios(data_filt), window_size=5000, n_channels=2, ylim=(-1.2, 1.2), lw=1)
    # prt.RecordLayer(data)
    if SEGMENT:
        segmented = segment_data(data)
        split = prt.SplitLayer(segmented, print_fps=True)

        if FILTER:
            segmented = prt.ExponentialFilter(split, alpha=.4)
            # sos_filter = signal.butter(FILTER_ORDER, [FILTER_CUTOFF/FS*2], output='sos')
            # segmented = prt.SOSFilter(segmented, sos=sos_filter, axis=0, shape=(RX_CHANNELS*TX_CHANNELS,))

        prt.TimePlotLayer(segmented, window_size=600, n_channels=RX_CHANNELS_ADC*TX_CHANNELS, ylim=(-1.2, 1.2), lw=1)

    prt.LayerManager.session().run(show_monitor=False)


if __name__ == "__main__":
    main()
