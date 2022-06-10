import numpy as np
import array
import pyrealtime as prt


HEAD_RIGID_BODY_NAME = b"head"
HAND_RIGID_BODY_NAME = b"hand"
VICON_PORT = 9909

def decode_vicon_packet(data):
    data_double = np.array(list(array.array('d', data)))
    # print(data_double)
    return parse_data_array(data_double)


prev_hand_q = None
prev_head_q = None

def parse_data_array(data):
    global prev_hand_q, prev_head_q
    time = data[0]
    markers = data[1:31]
    frame_num = data[32]
    head_frame_num = data[33]
    hand_frame_num = data[41]
    # print(head_frame_num, hand_frame_num)
    # if head_frame_num != hand_frame_num:
    #     return
    head_pos = data[34:37]
    head_q = data[37:41]
    hand_pos = data[42:45]
    hand_q = data[45:49]

    if prev_head_q is not None and np.linalg.norm(head_q - prev_head_q) > np.linalg.norm(-head_q - prev_head_q):
        head_q *= -1

    if prev_hand_q is not None and np.linalg.norm(hand_q - prev_hand_q) > np.linalg.norm(-hand_q - prev_hand_q):
        hand_q *= -1


    prev_hand_q = hand_q
    prev_head_q = head_q
    data = np.hstack((time, frame_num, head_frame_num, head_pos, head_q, hand_frame_num, hand_pos, hand_q))

    marker_data = np.hstack((head_pos, head_q, hand_pos, hand_q, markers))
    return {'head_pos': head_pos, 'head_rot': head_q, 'hand_pos': hand_pos, 'hand_rot': hand_q, 'markers': marker_data, 'data': data}



def setup_fig(fig):
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    return {'head_pos': ax1, 'head_rot': ax2, 'hand_pos': ax3, 'hand_rot': ax4}


def get_vicon_source(show_plot=True):
    vicon_data = prt.UDPReadLayer(port=VICON_PORT, parser=decode_vicon_packet, print_fps=True, multi_output=True)

    if show_plot:
        fm = prt.FigureManager(setup_fig, fps=10)
        prt.TimePlotLayer(vicon_data.get_port('head_pos'), ylim=(-1000, 1000), n_channels=3, plot_key="head_pos",
                          fig_manager=fm)
        prt.TimePlotLayer(vicon_data.get_port('head_rot'), ylim=(-1.1, 1.1), n_channels=4, plot_key="head_rot",
                          fig_manager=fm)
        prt.TimePlotLayer(vicon_data.get_port('hand_pos'), ylim=(-1000, 1000), n_channels=3, plot_key="hand_pos",
                          fig_manager=fm)
        prt.TimePlotLayer(vicon_data.get_port('hand_rot'), ylim=(-1.1, 1.1), n_channels=4, plot_key="hand_rot",
                          fig_manager=fm)



    return vicon_data