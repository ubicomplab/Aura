import struct
import numpy as np
import array
import pyrealtime as prt

from controller_lib import convert_to_volts, get_device_data





def main():

    vicon_data = get_device_data()
    # vicon_data_down = prt.DecimateLayer(vicon_data, keep_every=5)
    # prt.PrintLayer(vicon_data)
    prt.TimePlotLayer(vicon_data, window_size=5000, n_channels=3, ylim=(-1.2, 1.2))
    prt.LayerManager.session().run()


if __name__ == "__main__":
    main()

# time
# 32: frame num
# 33: head frame num
# 34-36: head pos
# 37-41: head q
# 42: hand frame num
# 43-45: hand pos
# 46-49: hand q

#ssh swetko@svlws
#pw: s

#nano source/lowlevel/vicon.cpp
#line 194

#cd build
#ninja a.out
