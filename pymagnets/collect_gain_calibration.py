import pyrealtime as prt
import numpy as np
from controller_lib import get_device_data
import time

CALIBRATION_MATRIX = np.eye(3)
DO_CALIBRATION = False
DO_RECORD = False


@prt.transformer
def decimate(data):
    return data[0]


def get_threshold(voltage):
    return min(0.0015, (voltage-.2) / 800 + .00001)


voltage = 0
last_value = None
@prt.transformer
def differentiate(data):
    global last_value
    if last_value is None:
        last_value = data
    diff = data - last_value
    last_value = data
    return np.hstack((diff, get_threshold(voltage)))

last_inc_time = time.perf_counter()+2
@prt.transformer
def get_voltage(data):
    global voltage, last_inc_time
    if max(data) > get_threshold(voltage) and time.perf_counter() - last_inc_time > 0.25:
        voltage += 0.1
        last_inc_time = time.perf_counter()
        return voltage
    return None


class SampleExtractor(prt.TransformMixin, prt.ThreadLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extract_next = True
        self.extract_time = None


    def transform(self, data):
        if self.extract_next:
            self.extract_time = time.perf_counter()
            self.extract_next = False
            return None
        if self.extract_time is not None and time.perf_counter() - self.extract_time > .25:
            self.extract_time = None
            return np.hstack((voltage, data))
        return None

    def handle_signal(self, signal):
        self.extract_next = True

def main():

    data = get_device_data(use_abs=True)

    # fm = prt.FigureManager(create_fig=create_fig)

    data_filt = prt.ExponentialFilter(data, alpha=.05, batch=True)
    downsampled = decimate(data_filt)
    diff = differentiate(downsampled)
    # decimated = prt.DecimateLayer(, keep_every=5)
    # prt.TimePlotLayer(data_filt, window_size=5000, n_channels=3, ylim=(-1.2, 1.2), lw=1)
    prt.TimePlotLayer(data, window_size=5000, n_channels=3, ylim=(-1.2, 1.2), lw=1)
    voltage = get_voltage(diff)
    extractor = SampleExtractor(downsampled)
    extractor.set_signal_in(voltage)
    prt.PrintLayer(extractor)

    prt.TimePlotLayer(diff, window_size=500, n_channels=4, ylim=(-.0125, .0125), lw=1)
    # prt.TimePlotLayer(voltage, window_size=100, n_channels=1, ylim=(0, .5), lw=1)

    if DO_RECORD:
        prt.RecordLayer(extractor, file_prefix="calibrations/gain_calibration")
    prt.LayerManager.session().run(show_monitor=False)


if __name__ == "__main__":
    main()
