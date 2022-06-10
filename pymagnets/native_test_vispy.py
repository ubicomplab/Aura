import struct
import numpy as np
import array
import pyrealtime as prt
from vispy import scene, app

from controller_lib import convert_to_volts
from vispy_plot import OscilloscopeWrapper

VICON_PORT = 9987

FRAME_FORMAT = "Ihhhc"
FRAME_SIZE = 11
FRAMES_PER_PACKET = 10


def decode_vicon_packet(data):
    if len(data) != FRAME_SIZE * FRAMES_PER_PACKET:
        print("Invalid packet size: ", len(data))
        return None
    all_parsed = []
    for i in range(FRAMES_PER_PACKET):
        data_frame = data[i*FRAME_SIZE:(i+1) * FRAME_SIZE]
        frame_num, x1, x2, x3, raw_frame_id = struct.unpack(FRAME_FORMAT, data_frame)
        parsed_data = convert_to_volts(np.array([x1, x2, x3]))
        all_parsed.append(parsed_data)
    # thumb data, finger data
    # time, x, y, z, w, x, y, z
    return np.array(all_parsed)


def main():

    vicon_data = prt.UDPReadLayer(port=VICON_PORT, parser=decode_vicon_packet, print_fps=True)
    # vicon_data_down = prt.DecimateLayer(vicon_data, keep_every=5)
    # prt.PrintLayer(vicon_data)
    win = scene.SceneCanvas(keys='interactive', show=True, fullscreen=False)
    grid = win.central_widget.add_grid()
    # view3 = grid.add_view(row=0, col=0, col_span=2, camera='panzoom',
    #                       border_color='grey')
    # image = ScrollingImage((1 + fft_samples // 2, 4000), parent=view3.scene)
    # image.transform = scene.LogTransform((0, 10, 0))
    # # view3.camera.rect = (0, 0, image.size[1], np.log10(image.size[0]))
    # view3.camera.rect = (3493.32, 1.85943, 605.554, 1.41858)
    view1 = grid.add_view(row=0, col=0, camera='panzoom', border_color='grey')

    view1.camera.rect = (-.01, -0.6, 0.02, 1.2)
    gridlines = scene.GridLines(color=(1, 1, 1, 0.5), parent=view1.scene)
    OscilloscopeWrapper(vicon_data, n_channels=3)
    # prt.TimePlotLayer(vicon_data, window_size=5000, n_channels=3, ylim=(-1.2, 1.2))
    prt.LayerManager.session().start(main_thread=OscilloscopeWrapper)
    app.run()


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
