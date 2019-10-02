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
    for method_name in ['random', 'TTTS_uniform', 'TTTS_informed']:
        success_rate = success_rate_dict[method_name]
        plt.plot(success_rate, label=method_name)
    plt.xlabel('#Queries')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.ylim(0.0, 1.0)
    plt.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')


def plot_reliability_diagram_error(N_list, filename_bayesian, filename_frequentist, figname):
    bayesian_estimation_error = np.genfromtxt(filename_bayesian)
    frequentist_estimation_error = np.genfromtxt(filename_frequentist)
    plt.figure(figsize=(COLUMN_WIDTH, COLUMN_WIDTH / GOLDEN_RATIO), dpi=300)
    plt.errorbar(N_list, bayesian_estimation_error.mean(axis=0), bayesian_estimation_error.std(axis=0), linestyle='None',
                marker='^', label='Bayesian')
    plt.errorbar(N_list, frequentist_estimation_error.mean(axis=0), frequentist_estimation_error.std(axis=0),
                linestyle='None', marker='*', label='Frequentist')
    plt.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')

# dir = '../output/active_learning'
# for file in os.listdir(dir):
#     if file.endswith(".pkl"):
#         filename = os.path.join(dir, file)
#         print(filename)
#         plot_active_learning(filename)

#
# dir = '../output/accuracy_estimation_error'
# filelist = os.listdir(dir)
# filelist.sort()
# for i in range(int(len(filelist)/2)):
#     filename_bayesian = os.path.join(dir, filelist[2*i])
#     filename_frequentist = os.path.join(dir, filelist[2*i+1])
#     print(filelist[2*i], filelist[2*i+1])
#     figname = '../figures/reliability_diagram_error/' + '_'.join(filelist[2*i].split('_')[:-1]) + '.pdf'
#     dataset = filelist[2*i].split("_")[2]
#     if dataset == "cifar100":  # 10,000
#         N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
#                   10000]
#     elif dataset == 'imagenet':  # 50,000
#         N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
#                   10000, 20000, 30000, 40000, 50000]
#     elif dataset == 'imagenet2':  # 10,000
#         N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
#                   10000]
#     elif dataset == '20newsgroup':  # 5607
#         N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 5607]
#     elif dataset == 'svhn':
#         N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
#                   10000, 20000, 26032]
#     plot_reliability_diagram_error(N_list, filename_bayesian, filename_frequentist, figname)
#     print(filename_bayesian, filename_frequentist)
