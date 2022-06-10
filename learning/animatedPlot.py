from pyquaternion import Quaternion

from utils import load_data, \
    load_predictions
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import pyquaternion

matplotlib.use('TKAgg')

TRAIL_LEN = 50
FILE_KEY = 'nostop'
NUM_QUAT = 2
DIST = .2


def generate_data():
    ground_truth_data = load_data(FILE_KEY)
    predicted_data = load_predictions(FILE_KEY)
    actual_data = ground_truth_data[['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']].as_matrix()

    for i in range(TRAIL_LEN, actual_data.shape[0]):
        yield actual_data[i-TRAIL_LEN:i, 0:3], Quaternion(actual_data[i, 3:7]), \
              predicted_data[i-TRAIL_LEN:i, 0:3], Quaternion(predicted_data[i, 3:7])
    yield None

def main():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))

    generator = generate_data()

    h_scatter = ax.scatter([0], [0], [0], s=1)
    h_scatter_hat = ax.scatter([0], [0], [0], s=1, c='r')

    # use a different color for each axis
    colors = ['r', 'g', 'b']

    # set up lines and points
    lines_qs = [[ax.plot([], [], [], c=c)[0] for c in colors] for i in range(NUM_QUAT)]
    [l.set_alpha(0.5) for l in lines_qs[0]]

    startpoints_qs = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    endpoints_qs = np.array([[DIST, 0, 0], [0, DIST, 0], [0, 0, DIST]])

    def init():
        for lines_q in lines_qs:
            for line in lines_q:
                line.set_data([], [])
                line.set_3d_properties([])

        return lines_q

    def update_graph(num):

        pos, rot, pos_hat, rot_hat = next(generator)
        print(rot)
        h_scatter._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
        h_scatter_hat._offsets3d = (pos_hat[:,0], pos_hat[:,1], pos_hat[:,2])
        title.set_text('3D Test, time={}'.format(num))

        for lines_q, p, r in zip(lines_qs, [pos, pos_hat], [rot, rot_hat]):
            for line, start, end in zip(lines_q, startpoints_qs, endpoints_qs):
                start = r.rotate(start)
                end = r.rotate(end)

                start += p[-1,:]
                end += p[-1,:]

                line.set_data([start[0], end[0]], [start[1], end[1]])
                line.set_3d_properties([start[2], end[2]])
        return [h_scatter, title] + lines_qs

    ani = animation.FuncAnimation(fig, update_graph, None, init_func=init,
                                  interval=30, blit=False)
    # ani.save('hand_to_head.html',fps=30)
    print("show")
    plt.show(block=True)
    print("done")


if __name__ == "__main__":
    main()
