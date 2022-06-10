import pyrealtime as prt
import serial
import math
import numpy as np
import struct
import time

CHANNELS = 9

BUFFER_SIZE = 2000
FS = 136
FILTER_CUTOFF = 18
FILTER_ORDER = 15

FILTER = False


# @prt.transformer
def process(data):
    try:
        data = struct.unpack('<' +'f' * CHANNELS, data)
        data = np.array([float(x) for x in data])
        # print(data)
        return np.cbrt(data)
    except ValueError:
        pass
    return None


def main():
    while True:
        try:
            serial_port = serial.Serial(prt.find_serial_port('COM11'), 115200, timeout=5)
            break
        except serial.SerialException as e:
            print(e)
            time.sleep(1)
    # serial_buffer = prt.FixedBuffer(BUFFER_SIZE, use_np=True, shape=(CHANNELS,), axis=0)
    data = prt.ByteSerialReadLayer.from_port(serial=serial_port, decoder=process, print_fps=True, preamble=b'UW', num_bytes=4 * CHANNELS)#, buffer=serial_buffer)

    if FILTER:
        data = prt.ExponentialFilter(data, alpha=.3)
        # sos_filter = signal.butter(FILTER_ORDER, [FILTER_CUTOFF/FS*2], output='sos')
        # segmented = prt.SOSFilter(segmented, sos=sos_filter, axis=0, shape=(RX_CHANNELS*TX_CHANNELS,))

    prt.TimePlotLayer(data, window_size=600, n_channels=CHANNELS, ylim=(-3, np.cbrt(18600)), lw=2)

    prt.LayerManager.session().run(show_monitor=False)



if __name__ == "__main__":
    main()
