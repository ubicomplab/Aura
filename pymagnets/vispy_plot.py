import numpy as np
from vispy import app, scene, gloo, visuals
from vispy.util.filter import gaussian_filter
import pyrealtime as prt


class OscilloscopeWrapper(prt.TransformMixin, prt.ThreadLayer):
    def __init__(self, port_in, n_channels=1, line_size=1024, dx=1e-4,
                 color=(20, 255, 50), trigger=(0, 0.002, 1e-4), parent=None):
        super().__init__(port_in)
        # self.scope = Oscilloscope(n_lines=n_lines, line_size=line_size, dx=dx,
        #          color=color, trigger=trigger, parent=parent)
        self.scope = scene.ScrollingLines(n_lines=1, line_size=line_size, dx=0.8/line_size/10,
                                          pos_offset=np.zeros((1, 3)), columns=1, parent=parent)

    def transform(self, data):
        # self.scope.update_scope(data)
        data = np.random.randn(1024)
        print(data.shape)
        # self.scope.roll_data(np.atleast_2d(data).T)
        self.scope.roll_data(data*100)