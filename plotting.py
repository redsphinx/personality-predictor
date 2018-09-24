import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time


def update_series(srs, value):
    srs.insert(0, value)
    srs.pop()
    return srs


def make_plot(gv):
    x = np.arange(100)

    monitor_dpi = 96 # https://www.infobyip.com/detectmonitordpi.php
    # fig = plt.figure(figsize=(500/monitor_dpi, 800/monitor_dpi), dpi=monitor_dpi)
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(650/monitor_dpi, 350/monitor_dpi), dpi=monitor_dpi)
    fig.subplots_adjust(hspace=0)

    axs[0].plot(x, gv.series_e)
    axs[0].tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    axs[0].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
    axs[1].plot(x, gv.series_a)
    axs[1].tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    axs[1].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
    axs[2].plot(x, gv.series_c)
    axs[2].tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    axs[2].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
    axs[3].plot(x, gv.series_s)
    axs[3].tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    axs[3].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
    axs[4].plot(x, gv.series_o)
    axs[4].tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    axs[4].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)

    location = 'media/traits.png'
    fig.savefig(location)
    # time.sleep(0.1)

    return location