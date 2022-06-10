import struct

import pyrealtime as prt
import serial
import time

from controller_lib import PACKET_FORMAT, parse_16, BUFFER_SIZE, RX_CHANNELS, NUM_BYTES
from settings import PORT, BAUD


def process(data):
    decoded_data = struct.unpack(PACKET_FORMAT, data)
    return parse_16(decoded_data[0:2])



def main():
    while True:
        try:
            serial_port = serial.Serial(prt.find_serial_port(PORT), BAUD, timeout=5)
            break
        except serial.SerialException as e:
            print(e)
            prt.time.sleep(1)

    data = prt.ByteSerialReadLayer.from_port(serial=serial_port, decoder=process, print_fps=True, preamble=b'UW',
                                             num_bytes=NUM_BYTES)
    # prt.PrintLayer(data)
    out = prt.OutputLayer(data)
    prt.LayerManager.session().start()
    time.sleep(5)
    prt.LayerManager.session().stop()
    while True:
        print(out.get_output())

if __name__ == "__main__":
    main()
