import audio_dspy as adsp
import numpy as np
import matplotlib.pyplot as plt

import utils

def create_filter_block(sample_rate, show_plot=False):
    filter_config = utils.util.get_config() 

    eq = adsp.EQ(sample_rate)
    for f in filter_config:
        if f['type'] == 'LPF':
            eq.add_LPF(f['fc'], f['Q'])
        if f['type'] == 'lowshelf':
            eq.add_lowshelf(f['fc'], f['Q'], f['gain'])
        if f['type'] == 'highshelf':
            eq.add_highshelf(f['fc'], f['Q'], f['gain'])

    if show_plot:
        eq.plot_eq_curve(worN=np.logspace(1, 3.3, num=1000, base=20))
        plt.grid()
        plt.ylim(-5, 8)
        # plt.xlim(0, sample_rate//2)
        plt.show()
    return eq
