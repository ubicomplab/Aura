import pyrealtime as prt
import serial
import math
import numpy as np
import struct

N_CHANNELS = 3

BUFFER_SIZE = 100
OFFSET = 0  # handled in adc sampling
OFF_THRESHOLD = 300
LAST_MAG_DROP_THRESHOLD = -.5
OFF_COUNT_THRESHOLD = 5

DATA_PER_CHANNEL = 24
CHANNELS = 3


def process(data):
    try:
        data = struct.unpack('<'+'h'*CHANNELS, data)
        data = np.array([float(x) for x in data])
        if len(data) == 3:
            return data
    except ValueError:
        pass
    return None


all_buffer = np.zeros((DATA_PER_CHANNEL, CHANNELS, 3))
all_buffer_idx = 0
is_on = False
tx_channel = 0
counts = [0] * CHANNELS
off_counts = 0
last_mag = 5000
last_peak = 5000
peaks = [0, 0, 0]


@prt.transformer
def analyze(data):
    global all_buffer_idx, all_buffer, is_on, tx_channel, counts, off_counts, last_mag, last_peak, peaks
    magnitude = np.linalg.norm(data, 2, axis=1)  # np.max(data, axis=1)#
    to_return_all = []
    flush = False
    for i in range(data.shape[0]):
        current_data = data[i]
        # print(magnitude[i])
        # if magnitude[i] < OFF_THRESHOLD or (magnitude[i] - last_mag)/last_peak < LAST_MAG_DROP_THRESHOLD or \
        #         (not is_on and (magnitude[i] < last_mag)):
        drops = (current_data - np.array(peaks)) / np.array(peaks)
        print(drops)
        if np.all(drops > .3):
            # off state
            # print('off')
            if is_on:
                is_on = False
                peaks = [0, 0, 0]
                counts[channel] = all_buffer_idx
                # if mag[i] >= OFF_THRESHOLD:
                #     print(mag[i])
                #     print(all_buffer_idx)
                #     print(last_mag)
                #     print(last_peak)
                all_buffer_idx = 0
                channel += 1
                if channel == CHANNELS:
                    flush = True
                    channel = 0
            off_counts += 1
            # print(mag[i])
        else:
            # on state
            # print('on')
            if not is_on:
                # print(off_counts)
                if off_counts > OFF_COUNT_THRESHOLD:
                    channel = 0
                off_counts = 0
                is_on = True
            for j in range(3):
                if current_data[j] > peaks[i]:
                    peaks[i] = current_data[j]
            if all_buffer_idx < DATA_PER_CHANNEL:
                all_buffer[all_buffer_idx, channel, :] = data[i]
                last_peak = magnitude[i]
            else:
                print(all_buffer_idx)
                print("overflow")
            all_buffer_idx += 1
        last_mag = magnitude[i]
        if flush:
            flush = False
            # print(counts)
            truncate = max(min(counts), 2)
            to_return = np.reshape(all_buffer[0:truncate,:,:], (-1, CHANNELS * 3))

            # to_return = np.max(to_return, axis=0)
            # to_return = np.median(to_return, axis=0)
            good_data = to_return[3:-3,:]
            print(good_data.shape)
            to_return = np.mean(good_data, axis=0)
            to_return_all.append(to_return)
            # print(to_return.shape)
    if len(to_return_all) == 0:
        return None
    return np.vstack(to_return_all)

    # return 2
    # if abs(data[channel]) < 3500:
    #     channel = (channel + 1) % 3
    #     return channel


@prt.transformer
def mag(data):
    return np.sort(np.sqrt(np.sum(np.power(np.split(data, 4), 2), axis=1)))


@prt.transformer
def measure_noise(data):
    return np.mean(np.std(data, axis=0))


@prt.transformer
def get_mean(data):
    return np.hstack((data, np.mean(data, axis=1, keepdims=True)))

def main():
    serial_port = serial.Serial(prt.find_serial_port('COM11'), 115200, timeout=5)
    serial_buffer = prt.FixedBuffer(BUFFER_SIZE, use_np=True, shape=(CHANNELS,))
    data = prt.ByteSerialReadLayer.from_port(serial=serial_port, decoder=process, print_fps=False, preamble=b'UW',
                                             num_bytes=2 * CHANNELS, buffer=serial_buffer)
    with_mean = get_mean(data)
    # prt.PrintLayer(with_mean)
    # data = prt.SerialReadLayer.from_port(serial=serial_port, decoder=process, print_fps=False)
    channel_data = analyze(data, print_fps=True)
    # noise_buffer = prt.Buffer(channel_data, in_buffer=prt.FixedBuffer(buffer_size=100, shape=(9,), axis=0, use_np=True))
    # prt.PrintLayer(measure_noise(noise_buffer))
    # prt.PrintLayer(buffer)
    # analyzer = analyze_data(data)
    # prt.PrintLayer(analyzer)
    filtered = channel_data# prt.ExponentialFilter(channel_data, alpha=.2)

    # prt.SerialWriteLayer.from_port(analyzer, serial=serial_port, encoder=lambda x: ('c'+str(x)).encode("UTF-8"))
    # prt.PrintLayer(analyzer)

    prt.TimePlotLayer(data, window_size=600*2, n_channels=CHANNELS, ylim=(-800, 30000), lw=1)
    # prt.TimePlotLayer(channel_data, window_size=600, n_channels=2, ylim=(-800, 30000), lw=1)
    prt.TimePlotLayer(filtered, window_size=600, n_channels=3*CHANNELS, ylim=(0, 20000), lw=1)
    #
    # out = prt.OutputLayer(buffer)
    # prt.TimePlotLayer(data.get_port('12'), window_size=400, n_channels=N_CHANNELS, ylim=(0, 20000), lw=1)
    # prt.TimePlotLayer(mag(filtered), window_size=40, n_channels=4, ylim=(0, 5000), lw=2)
    prt.LayerManager.session().run()
    # data = out.get_output()
    # prt.LayerManager.session().stop()
    print(data)



if __name__ == "__main__":
    main()
