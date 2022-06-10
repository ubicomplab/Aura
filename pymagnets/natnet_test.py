import pyrealtime as prt
import numpy as np

from prt_natnet import NatNetLayer

HEAD_RIGID_BODY_NAME = b"head"
HAND_RIGID_BODY_NAME = b"hand"

@prt.transformer(multi_output=True)
def parse(data):
    return {'pos':np.array(data[0]), 'rot':np.array(data[1])}


def setup_fig(fig):
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(313)
    return {'head_pos': ax1, 'head_rot': ax2, 'hand_pos': ax3, 'hand_rot': ax4, 'x5': ax5}

def main():
    natnet = NatNetLayer(bodies_to_track=[HEAD_RIGID_BODY_NAME, HAND_RIGID_BODY_NAME], multi_output=True, print_fps=True)
    # prt.PrintLayer(parse(natnet.get_port(RIGID_BODY_NAME)))
    parsed_head = parse(natnet.get_port(HEAD_RIGID_BODY_NAME))
    parsed_hand = parse(natnet.get_port(HAND_RIGID_BODY_NAME))
    fm = prt.FigureManager(setup_fig)
    prt.TimePlotLayer(parsed_head.get_port('pos'), ylim=(-2,2), n_channels=3, plot_key="head_pos", fig_manager=fm)
    prt.TimePlotLayer(parsed_head.get_port('rot'), ylim=(-2,2), n_channels=4, plot_key="head_rot", fig_manager=fm)
    prt.TimePlotLayer(parsed_hand.get_port('pos'), ylim=(-2,2), n_channels=3, plot_key="hand_pos", fig_manager=fm)
    prt.TimePlotLayer(parsed_hand.get_port('rot'), ylim=(-2,2), n_channels=4, plot_key="hand_rot", fig_manager=fm)
    prt.LayerManager.session().run()

if __name__ == "__main__":
    main()