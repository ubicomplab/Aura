# -*- coding: utf-8 -*-
# vispy: testskip
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
"""
An oscilloscope, spectrum analyzer, and spectrogram.

This demo uses pyaudio to record data from the microphone. If pyaudio is not
available, then a signal will be generated instead.
"""

from __future__ import division

import threading
import atexit
import numpy as np
from vispy import app, scene, gloo, visuals
from vispy.util.filter import gaussian_filter

try:
    import pyaudio

    class MicrophoneRecorder(object):
        def __init__(self, rate=44100, chunksize=1024):
            self.rate = rate
            self.chunksize = chunksize
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunksize,
                                      stream_callback=self.new_frame)
            self.lock = threading.Lock()
            self.stop = False
            self.frames = []
            atexit.register(self.close)

        def new_frame(self, data, frame_count, time_info, status):
            data = np.fromstring(data, 'int16')
            with self.lock:
                self.frames.append(data)
                if self.stop:
                    return None, pyaudio.paComplete
            return None, pyaudio.paContinue

        def get_frames(self):
            with self.lock:
                frames = self.frames
                self.frames = []
                return frames

        def start(self):
            self.stream.start_stream()

        def close(self):
            with self.lock:
                self.stop = True
            self.stream.close()
            self.p.terminate()

except ImportError:
    class MicrophoneRecorder(object):
        def __init__(self):
            self.chunksize = 1024
            self.rate = rate = 44100.
            t = np.linspace(0, 10, rate*10)
            self.data = (np.sin(t * 10.) * 0.3).astype('float32')
            self.data += np.sin((t + 0.3) * 20.) * 0.15
            self.data += gaussian_filter(np.random.normal(size=self.data.shape)
                                         * 0.2, (0.4, 8))
            self.data += gaussian_filter(np.random.normal(size=self.data.shape)
                                         * 0.005, (0, 1))
            self.data += np.sin(t * 1760 * np.pi)  # 880 Hz
            self.data = (self.data * 2**10 - 2**9).astype('int16')
            self.ptr = 0

        def get_frames(self):
            if self.ptr + 1024 > len(self.data):
                end = 1024 - (len(self.data) - self.ptr)
                frame = np.concatenate((self.data[self.ptr:], self.data[:end]))
            else:
                frame = self.data[self.ptr:self.ptr+1024]
            self.ptr = (self.ptr + 1024) % (len(self.data) - 1024)
            return [frame]

        def start(self):
            pass


class Oscilloscope(scene.ScrollingLines):
    """A set of lines that are temporally aligned on a trigger.

    Data is added in chunks to the oscilloscope, and each new chunk creates a
    new line to draw. Older lines are slowly faded out until they are removed.

    Parameters
    ----------
    n_lines : int
        The maximum number of lines to draw.
    line_size : int
        The number of samples in each line.
    dx : float
        The x spacing between adjacent samples in a line.
    color : tuple
        The base color to use when drawing lines. Older lines are faded by
        decreasing their alpha value.
    trigger : tuple
        A set of parameters (level, height, width) that determine how triggers
        are detected.
    parent : Node
        An optional parent scenegraph node.
    """
    def __init__(self, n_lines=100, line_size=1024, dx=1e-4,
                 color=(20, 255, 50), trigger=(0, 0.002, 1e-4), parent=None):

        self._trigger = trigger  # trigger_level, trigger_height, trigger_width

        # lateral positioning for trigger
        self.pos_offset = np.zeros((n_lines, 3), dtype=np.float32)

        # color array to fade out older plots
        self.color = np.empty((n_lines, 4), dtype=np.ubyte)
        self.color[:, :3] = [list(color)]
        self.color[:, 3] = 0
        self._dim_speed = 0.01 ** (1 / n_lines)

        self.frames = []  # running list of recently received frames
        self.plot_ptr = 0

        scene.ScrollingLines.__init__(self, n_lines=n_lines,
                                      line_size=line_size, dx=dx,
                                      color=self.color,
                                      pos_offset=self.pos_offset,
                                      parent=parent)
        self.set_gl_state('additive', line_width=2)

    def new_frame(self, data):
        self.frames.append(data)

        # see if we can discard older frames
        while len(self.frames) > 10:
            self.frames.pop(0)

        if self._trigger is None:
            dx = 0
        else:
            # search for next trigger
            th = int(self._trigger[1])  # trigger window height
            tw = int(self._trigger[2] / self._dx)  # trigger window width
            thresh = self._trigger[0]

            trig = np.argwhere((data[tw:] > thresh + th) &
                               (data[:-tw] < thresh - th))
            if len(trig) > 0:
                m = np.argmin(np.abs(trig - len(data) / 2))
                i = trig[m, 0]
                y1 = data[i]
                y2 = data[min(i + tw * 2, len(data) - 1)]
                s = y2 / (y2 - y1)
                i = i + tw * 2 * (1-s)
                dx = i * self._dx
            else:
                # default trigger at center of trace
                # (optionally we could skip plotting instead, or place this
                # after the most recent trace)
                dx = self._dx * len(data) / 2.

        # if a trigger was found, add new data to the plot
        self.plot(data, -dx)

    def plot(self, data, dx=0):
        self.set_data(self.plot_ptr, data)

        np.multiply(self.color[..., 3], 0.98, out=self.color[..., 3],
                    casting='unsafe')
        self.color[self.plot_ptr, 3] = 50
        self.set_color(self.color)
        self.pos_offset[self.plot_ptr] = (dx, 0, 0)
        self.set_pos_offset(self.pos_offset)

        self.plot_ptr = (self.plot_ptr + 1) % self._data_shape[0]


mic = MicrophoneRecorder()

win = scene.SceneCanvas(keys='interactive', show=True, fullscreen=False)
grid = win.central_widget.add_grid()

view1 = grid.add_view(row=0, col=0, camera='panzoom', border_color='grey')
view1.camera.rect = (-0.01, -0.6, 0.02, 1.2)
gridlines = scene.GridLines(color=(1, 1, 1, 0.5), parent=view1.scene)
print(mic.chunksize, 1.0/mic.rate)
# scope = Oscilloscope(line_size=mic.chunksize, dx=1.0/mic.rate, parent=view1.scene)



mic.start()

scope2 = scene.ScrollingLines(n_lines=1, line_size=mic.chunksize, dx=1.0/mic.rate,
        parent=view1.scene, columns=1, cell_size=(1,1), pos_offset=np.zeros((1, 3), dtype=np.float32))

def update(ev):
    global scope2, spectrum, mic
    data = mic.get_frames()
    for frame in data:

        # scope2.new_frame(frame)
        print(frame.shape)
        scope2.roll_data(frame/1000)




timer = app.Timer(interval='auto', connect=update)
timer.start()

if __name__ == '__main__':
    app.run()
