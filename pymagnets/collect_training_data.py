import pyrealtime as prt

from controller_lib import get_device_data, segment_data
from optitrack_lib import get_opti_source
from vicon_lib import get_vicon_source


USE_MOCAP = True
MOCAP_IS_VICON = False
RECORD = True
SEGMENT = False


def main():
    data = get_device_data(show_plot=True)
    # prt.TimePlotLayer(data.get_port("count"), n_channels=1, window_size=1000, ylim=(0,50))
    if SEGMENT:
        segmented = segment_data(data, show_plot=True)
    # if RECORD:
    #     prt.RecordLayer(data.get_port("raw_data"), file_prefix="mag-raw")

    if USE_MOCAP:
        if MOCAP_IS_VICON:
            mocap_vicon = get_vicon_source(show_plot=False)
            mocap = mocap_vicon.get_port("data")
        else:
            mocap = get_opti_source(show_plot=True)

        if RECORD:
            prt.RecordLayer(mocap, file_prefix=r"C:\Users\farsh\Dropbox\aura_data\recordings_native\mocap")

    prt.LayerManager.session().run()



if __name__ == "__main__":
    main()
