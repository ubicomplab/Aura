import serial
import numpy as np
import struct

TX_CHANNELS = 3
RX_CHANNELS = 3

ON_COUNT = 8
PRE_COUNT = 3
OFF_COUNT = 4
FULL_CYCLE = 43


STATE_OFF = 0
STATE_WAIT = 1
STATE_PRE_ON_1 = 2
STATE_ON_1 = 3
STATE_PRE_ON_2 = 4
STATE_ON_2 = 5
STATE_PRE_ON_3 = 6
STATE_ON_3 = 7


def get_simple_state(state):
    if is_state_pre_on(state):
        return 2
    if is_state_on(state):
        return 3
    else:
        return state

state = STATE_WAIT

sorted_data = np.zeros((RX_CHANNELS, TX_CHANNELS, ON_COUNT))
sorted_data_index = np.zeros((TX_CHANNELS,), dtype=int)
state_count = 0

mean_buffer = np.zeros((OFF_COUNT,))
mean_buffer_index = 0

MEANS_BUFFER_LEN = FULL_CYCLE - 10
running_means = np.zeros((MEANS_BUFFER_LEN)) + 1
running_means_index = 0

tx_channel = 0


def is_state_on(_state):
    return (_state == STATE_ON_1) or (_state == STATE_ON_2) or (_state == STATE_ON_3)


def is_state_pre_on(_state):
    return (_state == STATE_PRE_ON_1) or (_state == STATE_PRE_ON_2) or (_state == STATE_PRE_ON_3)


def is_state_normal_off(_state):
    return _state == STATE_OFF


global_count = 0
last_mean = 0


def segment(current_data):
    global state, tx_channel, state_count, global_count, mean_buffer, mean_buffer_index, running_means, running_means_index, last_mean
    flush = False
    # print(global_count)

    mean_buffer[mean_buffer_index] = np.linalg.norm(np.abs(current_data)-0.006)# np.sum(np.abs(current_data))
    mean_buffer_index += 1
    mean_buffer_index = mean_buffer_index % OFF_COUNT

    new_mean = np.sum(mean_buffer)
    # new_mean = running_means[running_means_index-1] + np.mean(mean_buffer[mean_buffer_index-1, :] - mean_buffer[mean_buffer_index, :])# np.mean(mean_buffer)
    running_means[running_means_index] = new_mean
    running_means_index += 1
    running_means_index = running_means_index % MEANS_BUFFER_LEN

    # if new_mean < last_mean:
    #     state = STATE_WAIT

    global_count += 1

    if state == STATE_WAIT:
        samples_since_last_min = ((running_means_index - 1) - np.argmin(running_means)) % MEANS_BUFFER_LEN
        if samples_since_last_min == 2:
            tx_channel = 0
            state = STATE_PRE_ON_1
    if is_state_pre_on(state):
        if state == STATE_PRE_ON_1 and state_count == 0:
            # samples_since_last_min = ((running_means_index - 1) - np.argmin(running_means)) % MEANS_BUFFER_LEN
            # state_count = samples_since_last_min
            # last_mean = np.min(running_means)
            # if state_count > PRE_COUNT or state_count == 0:
            #     state_count = -1
            # print(global_count, samples_since_last_min, np.min(running_means), state_count)
            tx_channel = 0
        state_count += 1

        if state_count >= PRE_COUNT:
            state += 1
            state_count = 0

    elif is_state_on(state):

        if 0 <= state_count < ON_COUNT:
            sorted_data[:, tx_channel, state_count] = current_data
        # sorted_data_index[tx_channel] += 1
        state_count += 1
        if state_count >= ON_COUNT:
            state += 1
            sorted_data_index[tx_channel] = state_count
            state_count = 0
            tx_channel += 1
            tx_channel = tx_channel % TX_CHANNELS
            state = state % (STATE_ON_3+1)
        # elif should_turn_off(current_data):
        #     state = 0
        #     sorted_data_index[tx_channel] = state_count
        #     tx_channel = 0
        #     state_count = 0

    elif is_state_normal_off(state):

        state_count += 1

        # turn_on = should_turn_on(current_data)
        if state_count == OFF_COUNT - 2:# or
            # if state_count >= OFF_COUNT:
            tx_channel = 0
            state = STATE_WAIT
            flush = True
            state_count = 0

    if flush:
        results = np.zeros((RX_CHANNELS, TX_CHANNELS))
        for i in range(TX_CHANNELS):
            extracted_data = sorted_data[:,i,0:sorted_data_index[i]]
            if extracted_data.shape[1] > 4:
                extracted_data = extracted_data[:,:-2]
            results[:,i] = np.sign(np.mean(extracted_data, axis=1)) * np.mean(np.abs(extracted_data), axis=1)

        return results.flatten(), state, np.mean(mean_buffer)

    return None, state, np.mean(mean_buffer)



