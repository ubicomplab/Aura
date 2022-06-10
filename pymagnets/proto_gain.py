import pyrealtime as prt
import serial
import math


N_CHANNELS = 3


def parse_data(data):
    try:
        parsed = [int(x) for x in data.split(',')]
        return parsed if len(parsed) == N_CHANNELS else None
    except:
        return None


def combine_iq(data):
    return [math.sqrt(data[0]**2 + data[1]**2), math.sqrt(data[2]**2 + data[3]**2), math.sqrt(data[4]**2 + data[5]**2)]


def main():
    serial_port = serial.Serial(prt.find_serial_port('KitProg'), 115200, timeout=5)
    print(serial_port)
    data = prt.SerialReadLayer.from_port(serial=serial_port, parser=parse_data)

    keyboard = prt.InputLayer(frame_generator=lambda counter: input("Enter Gain: "))
    prt.SerialWriteLayer.from_port(keyboard, serial=serial_port, encoder=lambda x: x.encode("UTF-8"))

    prt.TimePlotLayer(data, window_size=1000, n_channels=N_CHANNELS, ylim=(2**19, 1000000))
    # iq = prt.TransformLayer(data, transformer=combine_iq)
    # prt.TimePlotLayer(iq, window_size=1000, n_channels=N_CHANNELS, ylim=(0, 1000000))
    # prt.PrintLayer(data)
    prt.LayerManager.session().run(show_monitor=False)


if __name__ == "__main__":
    main()