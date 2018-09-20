from Tkinter import Button, Label, PhotoImage, Tk
import numpy as np

series_length = 100
trait_image = '/home/gabi/PycharmProjects/visualizing-traits/src/live_demo/plots/traits_tmp.png'


class GuiVar(object):
    def __init__(self):
        self._root = Tk()
        self._window_title = 'deepimpression 1.0'
        self._window_width = 500
        self._window_height = 500 + 300
        self._window_position = 200

        self._camera_image = PhotoImage(file='/home/gabi/PycharmProjects/visualizing-traits/src/live_demo/eye.png')
        self._camera = None

        self._trait_image_all = PhotoImage(file=trait_image)
        self._trait_all = None

        self._run_button = None
        self._keep_going = False

        self._model = None

        self._series_o = [0.5] * series_length
        self._series_c = [0.5] * series_length
        self._series_e = [0.5] * series_length
        self._series_a = [0.5] * series_length
        self._series_s = [0.5] * series_length

        self._label_o = None
        self._label_c = None
        self._label_e = None
        self._label_a = None
        self._label_s = None

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        self._root = value

    @property
    def window_title(self):
        return self._window_title

    @window_title.setter
    def window_title(self, value):
        self._window_title = value

    @property
    def window_width(self):
        return self._window_width

    @window_width.setter
    def window_width(self, value):
        self._window_width = value

    @property
    def window_height(self):
        return self._window_height

    @window_height.setter
    def window_height(self, value):
        self._window_height = value

    @property
    def window_position(self):
        return self._window_position

    @window_position.setter
    def window_position(self, value):
        self._window_position = value

    @property
    def camera_image(self):
        return self._camera_image

    @camera_image.setter
    def camera_image(self, value):
        self._camera_image = value

    @property
    def camera(self):
        return self._camera

    @camera.setter
    def camera(self, value):
        self._camera = value

    @property
    def run_button(self):
        return self._run_button

    @run_button.setter
    def run_button(self, value):
        self._run_button = value

    @property
    def keep_going(self):
        return self._keep_going

    @keep_going.setter
    def keep_going(self, value):
        self._keep_going = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def series_o(self):
        return self._series_o

    @series_o.setter
    def series_o(self, value):
        self._series_o = value

    @property
    def series_c(self):
        return self._series_c

    @series_c.setter
    def series_c(self, value):
        self._series_c = value

    @property
    def series_e(self):
        return self._series_e

    @series_e.setter
    def series_e(self, value):
        self._series_e = value

    @property
    def series_a(self):
        return self._series_a

    @series_a.setter
    def series_a(self, value):
        self._series_a = value

    @property
    def series_s(self):
        return self._series_s

    @series_s.setter
    def series_s(self, value):
        self._series_s = value

    @property
    def trait_image_all(self):
        return self._trait_image_all

    @trait_image_all.setter
    def trait_image_all(self, value):
        self._trait_image_all = value

    @property
    def trait_all(self):
        return self._trait_all

    @trait_all.setter
    def trait_all(self, value):
        self._trait_all = value

    @property
    def label_o(self):
        return self._label_o

    @label_o.setter
    def label_o(self, value):
        self._label_o = value

    @property
    def label_c(self):
        return self._label_c

    @label_c.setter
    def label_c(self, value):
        self._label_c = value

    @property
    def label_e(self):
        return self._label_e

    @label_e.setter
    def label_e(self, value):
        self._label_e = value

    @property
    def label_a(self):
        return self._label_a

    @label_a.setter
    def label_a(self, value):
        self._label_a = value

    @property
    def label_s(self):
        return self._label_s

    @label_s.setter
    def label_s(self, value):
        self._label_s = value