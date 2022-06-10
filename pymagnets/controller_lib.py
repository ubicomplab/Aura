from segmentation2 import segment
from settings import PORT, BAUD
import serial
import pyrealtime as prt
import struct
import numpy as np
import time

OSR = 256
BYTES_PER_SAMPLE = 2
RX_CHANNELS_ADC = 3
BUFFER_SIZE = 100
AGGREGATE = 1
RX_CHANNELS = 3

NUM_SIGN_BYTES = 1

NUM_BYTES = 2 + BYTES_PER_SAMPLE * RX_CHANNELS_ADC * AGGREGATE + NUM_SIGN_BYTES
PACKET_FORMAT = '<H' + 'B' * RX_CHANNELS_ADC * AGGREGATE * BYTES_PER_SAMPLE + ("B" * NUM_SIGN_BYTES)

USE_NATIVE = True

VICON_PORT = 9987

FRAME_FORMAT = "IHhhh"
FRAME_SIZE = 12
FRAMES_PER_PACKET = 68

def parse_24(data):
    return float((data[0] << 0) + (data[1] << 8) + (data[2] << 16))


def parse_16(data):
    return float((data[0] << 0) + (data[1] << 8))


def convert_to_volts(data):
    center = 0x800000 >> int(((256 / OSR)-1) * 3)
    if BYTES_PER_SAMPLE == 2:
        center = center >> 8
    return np.sign(data) * (np.abs(data) - center) / (center-1) * -1.2


def fix_signs(data, counts, buffer_size):
    signs = [(1 if x > buffer_size/2 else -1) for x in counts]
    return data * signs


def fix_signs_single_byte(data, sign_byte):
    values = [(sign_byte>>0) & 0x03, (sign_byte>>2) & 0x03, (sign_byte>>4) & 0x03]
    # print(values)
    signs = [(1 if x >=2 else -1) for x in values]
    return data * signs


def process(use_abs):
    def process_data(data):
        try:
            decoded_data = struct.unpack(PACKET_FORMAT, data)
            print(decoded_data[0])
            if BYTES_PER_SAMPLE == 2:
                data = np.array([parse_16(decoded_data[1:3]), parse_16(decoded_data[3:5]), parse_16(decoded_data[5:7])])
            elif BYTES_PER_SAMPLE == 3:
                data = np.array([parse_24(decoded_data[0:3]), parse_24(decoded_data[3:6]), parse_24(decoded_data[6:9])])
            # data = [parse_16(data[0:2]), parse_16(data[2:4]), parse_16(data[4:6])]
            # print(decoded_data[-1])
            # print(data)
            data = data.reshape((AGGREGATE, RX_CHANNELS_ADC))
            data = data[:,:RX_CHANNELS]
            data = convert_to_volts(data)
            if NUM_SIGN_BYTES == 1:
                data = fix_signs_single_byte(data, decoded_data[-1])
            elif NUM_SIGN_BYTES == 4:
                data = fix_signs(data, decoded_data[-4:-1], decoded_data[-1])
            # if np.max(data) > 1.2 or np.min(data) < 0:
            #     print(data)
                # return None

            # data = (data - 0x20000) / 0x10000
            # for i in range(len(data)//3):
            # if len(data) == CHANNELS:
            if use_abs:
                data = np.abs(data)
            return data
        except ValueError:
            print("parse error")
            pass
        return None
    return process_data


def decode_native_packet(data):
    if len(data) != FRAME_SIZE * FRAMES_PER_PACKET:
        print("Invalid packet size: ", len(data))
        return None
    all_raw_data = []
    all_data = []
    all_counts = []
    for i in range(FRAMES_PER_PACKET):
        data_frame = data[i*FRAME_SIZE:(i+1) * FRAME_SIZE]
        frame_num, raw_frame_id, x1, x2, x3 = struct.unpack(FRAME_FORMAT, data_frame)
        parsed_data = convert_to_volts(np.array([x1, x2, x3]))
        all_raw_data.append(np.hstack((frame_num, raw_frame_id, parsed_data)))
        all_data.append(parsed_data)
        # all_counts.append(count)

    return {'raw_data': np.array(all_raw_data), 'data': np.array(all_data)} # , 'count': np.array(all_counts)


def get_device_data(use_abs=False, show_plot=True):
    if USE_NATIVE:
        data = prt.UDPReadLayer(port=VICON_PORT, parser=decode_native_packet, print_fps=True, multi_output=True, bufsize=2048)
        plot_data = data.get_port("data")
    else:
        while True:
            try:
                serial_port = serial.Serial(prt.find_serial_port(PORT), BAUD, timeout=5)
                break
            except serial.SerialException as e:
                print(e)
                time.sleep(1)

        serial_buffer = prt.FixedBuffer(BUFFER_SIZE, use_np=True, shape=(RX_CHANNELS,), axis=0)
        data = prt.ByteSerialReadLayer.from_port(serial=serial_port, decoder=process(use_abs), print_fps=True, preamble=b'UW',
                                                 num_bytes=NUM_BYTES, buffer=serial_buffer)
        plot_data = data
    if show_plot:
        prt.TimePlotLayer(plot_data, window_size=1000, n_channels=RX_CHANNELS, ylim=(-1.2, 1.2), lw=1)

    return data


@prt.transformer
def segment_data_layer(data):
    to_return = []
    for i in range(data.shape[0]):
        segmented, state, off = segment(data[i,:])
        if segmented is not None:
            to_return.append(segmented)
    if len(to_return) == 0:
        return None
    return np.array(to_return)


def segment_data(data, show_plot=False):
    segmented = segment_data_layer(data)  # TODO: If using native, need to add .get_port('data')
    if show_plot:
        prt.TimePlotLayer(segmented, window_size=1000, n_channels=9, ylim=(-1.2, 1.2), lw=1)
