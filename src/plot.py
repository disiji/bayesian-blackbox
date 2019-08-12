import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

COLUMN_WIDTH = 3.25  # Inches
TEXT_WIDTH = 6.299213  # Inches
GOLDEN_RATIO = 1.61803398875
DPI = 300
FONT_SIZE = 8

def plot_active_learning(pickle_filename):
    figname = ('../figures/active_learning/' + pickle_filename.split('/')[-1])[:-3] + 'pdf'
    #figname = ('~/Dropbox/Apps/ShareLaTex/AAAI 2020 Bayesian Assessment of Blackbox Models/new_figures/active_learning/' + pickle_filename.split('/')[-1])[:-3] + 'pdf'
    success_rate_dict = pickle.load( open(pickle_filename, "rb" ) )
    plt.figure(figsize=(COLUMN_WIDTH, COLUMN_WIDTH / GOLDEN_RATIO), dpi=300)
    for method_name in ['random', 'TTTS']:
        success_rate = success_rate_dict[method_name]
        plt.plot(success_rate, label=method_name)
    plt.xlabel('Time')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.ylim(0.0, 1.0)
    plt.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')


dir = '../output/active_learning'
for file in os.listdir(dir):
    if file.endswith(".pkl"):
        filename = os.path.join(dir, file)
        print(filename)
        plot_active_learning(filename)
