import pyrealtime as prt
import numpy as np
from controller_lib import get_device_data

CALIBRATION_MATRIX = np.eye(3)
DO_CALIBRATION = False
DO_RECORD = False
# CALIBRATION_MATRIX = np.array([[2.21689997,  0.20745002,  0.18046373], [-0.05727706,  2.24714853,
#         0.08938468], [-0.02623188,  0.08895261,  2.00035859]])

@prt.transformer
def calibrate(data):
    return np.matmul(data, CALIBRATION_MATRIX)

@prt.transformer
def get_mag(data):
    return np.linalg.norm(data, axis=1)

def create_fig(fig):
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,3)
    ax3 = fig.add_subplot(1,2,2)

    return {"ax1": ax1, "ax2": ax2, "ax3": ax3}

@prt.transformer
def decimate(data):
    return data[0]

def main():

    data = get_device_data()

    fm = prt.FigureManager(create_fig=create_fig)

    data_filt = prt.ExponentialFilter(data, alpha=1, batch=True)
    prt.PrintLayer(prt.DecimateLayer(decimate(data_filt), keep_every=50))
    prt.TimePlotLayer(data_filt, window_size=5000, n_channels=3, ylim=(-1.2, 1.2), lw=1, fig_manager=fm, plot_key="ax1")
    if DO_CALIBRATION:
        calibrated_data = calibrate(data_filt)
        mag = get_mag(calibrated_data)
        prt.TimePlotLayer(calibrated_data, window_size=5000, n_channels=3, ylim=(-1.2, 1.2), lw=1, fig_manager=fm, plot_key="ax2")
        prt.TimePlotLayer(mag, window_size=5000, n_channels=1, ylim=(0, 2), lw=1, fig_manager=fm, plot_key="ax3")
    if DO_RECORD:
        prt.RecordLayer(data_filt, file_prefix="calibrations/helmholz", split_axis=0)
    prt.LayerManager.session().run(show_monitor=False)


if __name__ == "__main__":
    main()
