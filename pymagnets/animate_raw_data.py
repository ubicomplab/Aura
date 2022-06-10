import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.signal
import seaborn as sns

DECIMATE = 8
FS = 152 / DECIMATE

class Scope(object):
    def __init__(self, ax, channels, maxt=5, dt=1/FS):
        self.ax = ax
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.ydata = []
        self.lines = []
        self.channels = channels
        current_palette = sns.hls_palette(channels)
        for channel in range(channels):
            self.ydata.append([0])
            self.lines.append(Line2D(self.tdata, [self.ydata[channel]], color=current_palette[channel]))
            self.ax.add_line(self.lines[-1])
        self.ax.set_ylim(0, 16000)
        self.ax.set_xlim(0, self.maxt)

    def update(self, y):
        lastt = self.tdata[-1]
        if lastt > self.tdata[0] + self.maxt:  # reset the arrays
            self.tdata = [self.tdata[-1]]
            for channel in range(self.channels):
                self.ydata[channel] = [self.ydata[channel][-1]]
            self.ax.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)
            self.ax.figure.canvas.draw()

        t = self.tdata[-1] + self.dt
        self.tdata.append(t)
        for channel in range(self.channels):
            self.ydata[channel].append(y[channel])
            self.lines[channel].set_data(self.tdata, self.ydata[channel])
        return self.lines


def main():
    data = np.load(r"debug_recordings\demo_video\recording_18_02_28_22_34_42-processed.npy")
    data_filt = []
    last_row = data[0,:]
    for row in data:
        last_row = last_row * .95 + row * .05
        data_filt.append(last_row)
    data_filt = np.array(data_filt)
    print(data_filt.shape)

    data_filt = data_filt[0::DECIMATE, :]
    print(data_filt.shape)

    fig, ax = plt.subplots()
    scope = Scope(ax, 9)

    def emitter():
        for i, row in enumerate(data_filt):
            print(i)
            yield row

    ani = animation.FuncAnimation(fig, scope.update, emitter, interval=1000/FS,
                                  blit=True, save_count=data_filt.shape[0])
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=FS, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('data.mp4')
    # plt.show()

if __name__ == "__main__":
    main()