import os
import pylab
import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def main(argv=None):
    # Arguments
    parser = argparse.ArgumentParser(description='Incremental Learning Framework - Experiment 3 Plotting')
    parser.add_argument('--path', type=str, default='/data/experiments/LLL/exp3a', help='(default=%(default)d)')
    parser.add_argument('--experiment', type=str, default='exp3a', help='(default=%(default)s)')
    parser.add_argument('--show_plot', action='store_true', help='(default=%(default)s)')
    args, extra_args = parser.parse_known_args(argv)
    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))

    font = {'family': 'serif',
            'weight': 'normal',
            'size': 11}
    matplotlib.rc('font', **font)

    # Iterate through folders in the experiment path and create ApprPlotter per approach
    exp_folders = [item for item in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, item))]
    approaches = []
    for exp in exp_folders:
        # get approach name
        approach = exp.split("_")[1]
        # check if it has results and get the latest version
        exp_path = os.path.join(args.path, exp, 'results')
        exp_file = [elem for elem in os.listdir(exp_path) if 'wavg_accs_tag' in elem]
        exp_file.sort()
        if not exp_file:
            print('Warning: ' + exp + ' has no results file.')
            continue
        else:
            exp_file = exp_file[-1]
        # create approach plotter
        approaches.append(ApprPlotter(approach, os.path.join(exp_path, exp_file)))

    # Adapt axis to current experiment
    x_axis, num_classes = [], 0
    if args.experiment in ['exp3a', 'exp3b', 'exp3c', 'exp3d']:
        # 1 task per dataset
        x_axis = [102, 169, 369, 565, 665, 705]
        num_classes = 705
    elif args.experiment in ['exp3e', 'exp3f','exp3g', 'exp3h']:
        # 4 tasks per dataset
        x_axis = [26, 52, 77, 102, 119, 136, 153, 169, 219, 269, 319, 369, 418, 467, 516, 565, 590, 615, 640, 665, 675,
                  685, 695, 705]
        num_classes = 705
    else:
        print('Experiment not defined yet')
        exit()

    # Start plotting
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    for appr_name in ['FT-E', 'Freezing', 'Joint', 'EWC', 'MAS', 'PathInt', 'RWalk', 'LwM', 'DMC',
                      'LwF', 'iCaRL', 'EEIL', 'BiC', 'LUCI', 'IL2M']:
        for appr in approaches:
            if appr.name == appr_name:
                ax.plot(x_axis, appr.data, color=appr.color, label=appr.name, linewidth=1.5,
                        marker=appr.marker, linestyle=appr.linestyle, markersize=appr.markersize)
    # Put ticks
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='major')
    ax.tick_params(axis='x', which='major', length=0)
    ax.set_xticks(x_axis, minor=False)
    # Axis titles
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontfamily='serif')
    ax.set_ylim(-1, 101)
    ax.set_xlabel("Number of classes", fontsize=11, fontfamily='serif')
    ax.set_xlim(-1, num_classes + 1)
    # Legend
    leg = ax.legend(bbox_to_anchor=(0., 1.20, 1., 0.1), loc='upper center', ncol=6, mode="expand", fancybox=True,
                    prop={'family': 'serif', 'size': 11})

    # Format plot and save figure
    plt.tight_layout()
    fig.savefig(os.path.join(args.path, '{}_plot.png'.format(args.experiment)))
    fig.savefig(os.path.join(args.path, '{}_plot.pdf'.format(args.experiment)))
    if args.show_plot:
        pylab.show()


class ApprPlotter:
    """ Class to manage all info to plot curves"""
    def __init__(self, approach_name, exp_file):
        self.lookup_table = {
                             'finetune': {'name': 'FT-E', 'color': '#949494', 'type': 'baseline'},
                             'freezing': {'name': 'Freezing', 'color': '#9edae5', 'type': 'baseline'},
                             'joint': {'name': 'Joint', 'color': '#666666', 'type': 'baseline'},
                             'lwf': {'name': 'LwF', 'color': '#ff7f0e', 'type': 'approach'},
                             'icarl': {'name': 'iCaRL', 'color': '#ff9896', 'type': 'approach'},
                             'eeil': {'name': 'EEIL', 'color': '#e39802', 'type': 'approach'},
                             'bic': {'name': 'BiC', 'color': '#d62728', 'type': 'approach'},
                             'luci': {'name': 'LUCI', 'color': '#cc78bc', 'type': 'approach'},
                             'lwm': {'name': 'LwM', 'color': '#029e73', 'type': 'approach'},
                             'ewc': {'name': 'EWC', 'color': '#8CD17D', 'type': 'approach'},
                             'mas': {'name': 'MAS', 'color': '#56b4e9', 'type': 'approach'},
                             'path': {'name': 'PathInt', 'color': '#fbafe4', 'type': 'approach'},
                             'r': {'name': 'RWalk', 'color': '#8c564b', 'type': 'approach'},
                             'dmc': {'name': 'DMC', 'color': '#ffda66', 'type': 'approach'},
                             'il2m': {'name': 'IL2M', 'color': '#0173b2', 'type': 'approach'}
                             }
        self.name = self.lookup_table[approach_name]['name']
        self.type = self.lookup_table[approach_name]['type']
        self.color = self.lookup_table[approach_name]['color']

        # open file and extract data
        self.file = exp_file
        with open(self.file) as f:
            data = f.read().splitlines()
            self.data = [float(elem)*100.0 for elem in data[0].split()]

        # check if approach is a baseline
        if self.type == 'baseline':
            self.linestyle = '--'
            self.marker = 'D'
            self.markersize = 5
        else:
            self.linestyle = '-'
            self.marker = 'o'
            self.markersize = 6


if __name__ == '__main__':
    main()
