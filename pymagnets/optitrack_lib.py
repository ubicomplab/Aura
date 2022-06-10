from prt_natnet import NatNetLayer
import pyrealtime as prt
import numpy as np


HEAD_RIGID_BODY_NAME = b"hmd"
HAND_RIGID_BODY_NAME = b"controller"


@prt.transformer(multi_output=True)
def parse(data):
    return {'pos': np.array(data[0]), 'rot': np.array(data[1])}


def get_pos(data):
    return np.array(data[0])


def get_rot(data):
    return np.array(data[1])


def get_opti(data):
    return np.hstack((get_pos(data), get_rot(data)))


def setup_fig(fig):
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    return {f'{HEAD_RIGID_BODY_NAME}_pos': ax1, f'{HEAD_RIGID_BODY_NAME}_rot': ax2, f'{HAND_RIGID_BODY_NAME}_pos': ax3, f'{HAND_RIGID_BODY_NAME}_rot': ax4}


@prt.transformer()
def encode(data):
    return np.hstack([data['frame_num'], data['head']['pos'], data['head']['rot'], data['hand']['pos'], data['hand']['rot']])


def get_opti_source(show_plot=True):
    natnet = NatNetLayer(bodies_to_track=[HEAD_RIGID_BODY_NAME, HAND_RIGID_BODY_NAME], multi_output=True, print_fps=True)

    frame_num = natnet.get_port("frame_num")
    parsed_head = parse(natnet.get_port(HEAD_RIGID_BODY_NAME))
    parsed_hand = parse(natnet.get_port(HAND_RIGID_BODY_NAME))
    if show_plot:
        fm = prt.FigureManager(setup_fig)
        prt.TimePlotLayer(parsed_head.get_port('pos'), ylim=(-2, 2), n_channels=3, plot_key=f'{HEAD_RIGID_BODY_NAME}_pos', fig_manager=fm)
        prt.TimePlotLayer(parsed_head.get_port('rot'), ylim=(-2, 2), n_channels=4, plot_key=f'{HEAD_RIGID_BODY_NAME}_rot', fig_manager=fm)
        prt.TimePlotLayer(parsed_hand.get_port('pos'), ylim=(-2, 2), n_channels=3, plot_key=f'{HAND_RIGID_BODY_NAME}_pos', fig_manager=fm)
        prt.TimePlotLayer(parsed_hand.get_port('rot'), ylim=(-2, 2), n_channels=4, plot_key=f'{HAND_RIGID_BODY_NAME}_rot', fig_manager=fm)

    data = prt.MergeLayer(None)
    data.set_input(frame_num, "frame_num")
    data.set_input(parsed_head, "head")
    data.set_input(parsed_hand, "hand")

    return encode(data)