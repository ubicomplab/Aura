import pyrealtime as prt
import serial
import math
import numpy as np
import struct
from scipy import signal

from debug_raw import segment_data
from prt_natnet import NatNetLayer

BUFFER_SIZE = 10
FS = 226
FILTER_CUTOFF = 18
FILTER_ORDER = 7
RX_CHANNELS = 3
TX_CHANNELS = 3
AGGREGATE = 10

USE_NATNET = True
RECORD = False
HEAD_RIGID_BODY_NAME = b"head"
HAND_RIGID_BODY_NAME = b"mag_controller_v8"


# @prt.transformer
def process(data):
    try:
        data = struct.unpack('<' +'h' * RX_CHANNELS * AGGREGATE, data)
        data = np.array([float(x) for x in data])
        data = data.reshape((AGGREGATE, RX_CHANNELS))
        #for i in range(len(data)//3):
        #if len(data) == CHANNELS:
        return data
    except ValueError:
        pass
    return None


def setup_fig(fig):
    if USE_NATNET:
        ax1 = fig.add_subplot(321)
        ax2 = fig.add_subplot(322)
        ax3 = fig.add_subplot(323)
        ax4 = fig.add_subplot(324)
        ax5 = fig.add_subplot(313)
        return {'head_pos': ax1, 'head_rot': ax2, 'hand_pos': ax3, 'hand_rot': ax4, 'mag': ax5}
    else:
        ax5 = fig.add_subplot(111)
        return {'mag': ax5}


@prt.transformer(multi_output=True)
def parse(data):
    return {'pos': np.array(data[0]), 'rot': np.array(data[1])}

def get_pos(data):
    return np.array(data[0])

def get_rot(data):
    return np.array(data[1])

def get_opti(data):
    return np.hstack((get_pos(data), get_rot(data)))


@prt.transformer
def rescale(data):
    return data / np.hstack(([30000] * 9, [30000] * 9, [1]*3, [1]*3, [1]*4, [1]*4))


@prt.transformer
def encode(data):
    return np.hstack((data['mag'][0,:], get_opti(data['optitrack'][HEAD_RIGID_BODY_NAME]), get_opti(data['optitrack'][HAND_RIGID_BODY_NAME])))
    # return np.hstack((data['mag'], data['mag_filt'],get_opti(data['optitrack'][HEAD_RIGID_BODY_NAME]), get_opti(data['optitrack'][HAND_RIGID_BODY_NAME])))


def main():
    serial_port = serial.Serial(prt.find_serial_port('COM24'), 115200, timeout=5)
    serial_buffer = prt.FixedBuffer(BUFFER_SIZE, use_np=True, shape=(RX_CHANNELS,), axis=0)
    raw_data = prt.ByteSerialReadLayer.from_port(serial=serial_port, decoder=process, print_fps=False, preamble=b'UW', num_bytes=2 * RX_CHANNELS * AGGREGATE, buffer=serial_buffer)
    # serial_buffer = prt.FixedBuffer(BUFFER_SIZE, use_np=True, shape=(CHANNELS,))
    # raw_data = prt.ByteSerialReadLayer.from_port(serial=serial_port, decoder=process, print_fps=True, preamble=b'UW', num_bytes=4*CHANNELS, buffer=serial_buffer)
    segmented = segment_data(raw_data)
    # segmented = prt.SplitLayer(segmented)
    # filtered = prt.ExponentialFilter(segmented, alpha=0.4, print_fps=True)
    # sos_filter = signal.butter(FILTER_ORDER, [FILTER_CUTOFF/FS*2], output='sos')
    # filtered = prt.SOSFilter(data, sos=sos_filter, axis=0, shape=(CHANNELS,))
    # filtered = data
    # prt.PrintLayer(data)
    fm = prt.FigureManager(setup_fig)
    prt.TimePlotLayer(segmented, window_size=500, n_channels=RX_CHANNELS*TX_CHANNELS, ylim=(0, 18600), lw=1, fig_manager=fm, plot_key='mag')
    # prt.TimePlotLayer(filtered, window_size=500, n_channels=RX_CHANNELS * TX_CHANNELS, ylim=(0, 18600))

    if USE_NATNET:
        natnet_split = NatNetLayer(bodies_to_track=[HEAD_RIGID_BODY_NAME, HAND_RIGID_BODY_NAME], multi_output=True)
        natnet = prt.MergeLayer(None, trigger=prt.LayerTrigger.SLOWEST, discard_old=True, multi_output=True, print_fps=True)
        natnet.set_input(natnet_split.get_port(HEAD_RIGID_BODY_NAME), HEAD_RIGID_BODY_NAME)
        natnet.set_input(natnet_split.get_port(HAND_RIGID_BODY_NAME), HAND_RIGID_BODY_NAME)
        # prt.PrintLayer(parse(natnet.get_port(HAND_RIGID_BODY_NAME)))
        parsed_head = parse(natnet.get_port(HEAD_RIGID_BODY_NAME))
        parsed_hand = parse(natnet.get_port(HAND_RIGID_BODY_NAME))
        prt.TimePlotLayer(parsed_head.get_port('pos'), ylim=(-2,2), n_channels=3, plot_key="head_pos", fig_manager=fm)
        prt.TimePlotLayer(parsed_head.get_port('rot'), ylim=(-2,2), n_channels=4, plot_key="head_rot", fig_manager=fm)
        prt.TimePlotLayer(parsed_hand.get_port('pos'), ylim=(-2,2), n_channels=3, plot_key="hand_pos", fig_manager=fm)
        prt.TimePlotLayer(parsed_hand.get_port('rot'), ylim=(-2,2), n_channels=4, plot_key="hand_rot", fig_manager=fm)

        if RECORD:
            # concat = prt.MergeLayer(None, trigger=prt.LayerTrigger.SLOWEST, discard_old=True)
            # # concat.set_input(filtered, 'mag_filt')
            # concat.set_input(segmented, 'mag')
            # concat.set_input(natnet, 'optitrack')
            # # prt.PrintLayer(concat)
            # flat = encode(concat)
            # prt.RecordLayer(flat, file_prefix="combo", append_time=True)
            # prt.RecordLayer(raw_data, file_prefix="mag-raw", append_time=True)
            prt.RecordLayer(segmented, file_prefix="mag-segmented", append_time=True)
            prt.RecordLayer(natnet_split, file_prefix="natnet", append_time=True)
        # prt.PrintLayer(rescale(flat))
        # prt.TimePlotLayer(rescale(flat), ylim=(-2,2), n_channels=32)

    prt.LayerManager.session().run()



if __name__ == "__main__":
    main()
