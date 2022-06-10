import serial
import numpy as np
import struct

TX_CHANNELS = 3
RX_CHANNELS = 1

BUFFER_SIZE = 100
OFFSET = 0  # handled in adc sampling
OFF_COUNT_THRESHOLD = 15
MIN_OFF = 6
MAX_OFF = 10
MAX_OFF_LONG = 22
MIN_ON = 15
DROP_THRESH = -.4
RISE_THRESH = .3

DATA_PER_CHANNEL = 40

all_buffer = np.zeros((DATA_PER_CHANNEL, TX_CHANNELS, RX_CHANNELS))
all_buffer_idx = 0
is_on = False
tx_channel = 0
counts = [0] * TX_CHANNELS
off_counts = 0
last_peaks = [0] * RX_CHANNELS * TX_CHANNELS
troughs = [0] * RX_CHANNELS * TX_CHANNELS  # troughs come before peaks, index-wise
next_max_off = MAX_OFF
auto_on = False


def segment(current_data):
    global all_buffer_idx, all_buffer, is_on, tx_channel, counts, off_counts, last_peaks, troughs, next_max_off, auto_on
    to_return_all = []
    flush = False
    # print(data.shape)

    if is_on:
        # drops = (current_data - np.array(last_peaks[channel*3:channel*3+3])) / (np.array(last_peaks[channel*3:channel*3+3]) - np.array(troughs[channel*3:channel*3+3]))
        drops = np.sum((current_data - np.array(
            last_peaks[tx_channel * RX_CHANNELS:(tx_channel + 1) * RX_CHANNELS]))) / np.sum(
            np.array(last_peaks[tx_channel * RX_CHANNELS:(tx_channel + 1) * RX_CHANNELS]) - np.array(
                troughs[tx_channel * RX_CHANNELS:(tx_channel + 1) * RX_CHANNELS]))
        print("on", drops, current_data, last_peaks[tx_channel * RX_CHANNELS:(tx_channel+1) * RX_CHANNELS], troughs[tx_channel * RX_CHANNELS:(tx_channel+1) * RX_CHANNELS])
        if (all_buffer_idx > MIN_ON and np.min(drops) < DROP_THRESH) or (auto_on and all_buffer_idx > 30):
            # print("all_buffer_idx", all_buffer_idx)
            print(all_buffer_idx)
            auto_on = False
            is_on = False
            counts[tx_channel] = all_buffer_idx
            all_buffer_idx = 0
            tx_channel += 1
            next_max_off = MAX_OFF
            if tx_channel == TX_CHANNELS - 0:
                next_max_off = MAX_OFF_LONG
            if tx_channel == TX_CHANNELS:
                flush = True
                tx_channel = 0

            for j in range(RX_CHANNELS):
                troughs[j + tx_channel * RX_CHANNELS] = current_data[j]
        else:
            for j in range(RX_CHANNELS):
                if current_data[j] > last_peaks[j + tx_channel * RX_CHANNELS]:
                    last_peaks[j + tx_channel * RX_CHANNELS] = current_data[j]
            if all_buffer_idx < DATA_PER_CHANNEL:
                all_buffer[all_buffer_idx, tx_channel, :] = current_data
                # last_peak = magnitude[i]
            else:
                print(all_buffer_idx)
                print("overflow")
            all_buffer_idx += 1
    else:
        rises = (current_data - np.array(troughs[tx_channel * RX_CHANNELS:(tx_channel + 1) * RX_CHANNELS])) / (
                np.array(last_peaks[tx_channel * RX_CHANNELS:(tx_channel + 1) * RX_CHANNELS]) - np.array(
            troughs[tx_channel * RX_CHANNELS:(tx_channel + 1) * RX_CHANNELS]))
        # rises = np.sum((current_data - np.array(troughs[tx_channel * RX_CHANNELS:(tx_channel+1) * RX_CHANNELS]))) / np.sum(current_data)

        print("off", rises, current_data, last_peaks[tx_channel * RX_CHANNELS:(tx_channel+1) * RX_CHANNELS], troughs[tx_channel * RX_CHANNELS:(tx_channel+1) * RX_CHANNELS])
        # print(next_max_off)
        auto_on = off_counts > next_max_off
        if auto_on:
            print("auto on")
        if (off_counts > MIN_OFF and np.max(rises) > RISE_THRESH) or auto_on:
            # for j in range(3):
            #     peaks[j+channel*3] = current_data[j]
            # print(off_counts)
            if off_counts > OFF_COUNT_THRESHOLD:
                tx_channel = 0
                truncate = max(min(counts), 2)
                reshaped_data = np.reshape(all_buffer[0:truncate, :, :], (-1, RX_CHANNELS * TX_CHANNELS))
                good_data = reshaped_data[3:-3, :]
                to_return = np.mean(good_data, axis=0)
                if not np.isnan(to_return).any():
                    last_peaks = to_return.copy()
            print(off_counts)
            off_counts = 0
            is_on = True
        else:
            for j in range(RX_CHANNELS):
                if current_data[j] < troughs[j + tx_channel * RX_CHANNELS]:
                    troughs[j + tx_channel * RX_CHANNELS] = current_data[j]
                # if current_data[j] > last_peaks[j + tx_channel * RX_CHANNELS]:
                #     last_peaks[j + tx_channel * RX_CHANNELS] = current_data[j] + 1

            off_counts += 1

    if flush:
        flush = False
        print(counts)
        if min(counts) > 6:
            truncate = max(min(counts), 2)
            reshaped_data = np.reshape(all_buffer[0:truncate, :, :], (-1, RX_CHANNELS * TX_CHANNELS))

            # to_return = np.max(to_return, axis=0)
            # to_return = np.median(to_return, axis=0)
            # print(counts)
            good_data = reshaped_data[3:-3, :]
            # print(good_data.shape)
            to_return = np.mean(good_data, axis=0)
            # if to_return[0] < 58000:
            #     print(to_return)
            to_return_all.append(to_return.copy())
            if not np.isnan(to_return).any():
                last_peaks = to_return.copy()
            # print("***********************",to_return)
            # print(Fto_return.shape)

    if len(to_return_all) == 0:
        return None
    return np.vstack(to_return_all)


