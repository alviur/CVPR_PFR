import os
import pylab
import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def main(argv=None):
    # Arguments
    parser = argparse.ArgumentParser(description='Incremental Learning Framework - Experiment 2 Plotting')
    parser.add_argument('--path', type=str, default='/data/experiments/LLL/exp2a', help='(default=%(default)d)')
    parser.add_argument('--experiment', type=str, default='exp2a', help='(default=%(default)s)')
    parser.add_argument('--approach', default='finetune', type=str, help='(default=%(default)s)')
    parser.add_argument('--num_classes', default=100, type=int, help='(default=%(default)s)')
    parser.add_argument('--show_plot', action='store_true', help='(default=%(default)s)')
    args, extra_args = parser.parse_known_args(argv)
    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))

    font = {'family': 'serif',
            'weight': 'normal',
            'size': 11}
    matplotlib.rc('font', **font)

    # Iterate through folders in the experiment path and create ApprPlotter per approach
    exp_folders = [item for item in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, item))]
    datasets = []
    seeds = {}
    for exp in exp_folders:
        # skip experiments that are not for the defined approach
        if args.approach not in exp:
            continue
        # extra check because of 'icarl' and 'cifar_icarl' being confusing
        if args.approach == 'icarl' and ('lwf' in exp or 'bic' in exp or 'luci' in exp or 'il2m' in exp):
            continue
        # get dataset name - remove last underscore
        dataset = exp.split(args.approach)[0][:-1]
        # extra check because of 'icarl' and 'cifar_icarl' being confusing
        if exp.count('icarl') == 2:
            dataset = 'cifar100_icarl'
        # check if it has results and get the latest version
        exp_path = os.path.join(args.path, exp, 'results')
        exp_file = [elem for elem in os.listdir(exp_path) if 'wavg_accs_tag' in elem]
        exp_file.sort()
        if not exp_file:
            print('Warning: ' + exp + ' has no results file.')
            continue
        else:
            exp_file = exp_file[-1]
        # check if dataset exists
        if dataset in seeds.keys():
            seeds[dataset].append(os.path.join(exp_path, exp_file))
        else:
            seeds[dataset] = [os.path.join(exp_path, exp_file)]

    # create approach plotters
    for k, v in seeds.items():
        datasets.append(DsetPlotter(k, v))

    # Adapt axis to current experiment
    x_axis = [(task + 1)*(args.num_classes / len(datasets[0].data)) for task in range(len(datasets[0].data))]

    # Start plotting
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    for dset_name in ['random', 'iCaRL seed', 'maxConf.', 'minConf.', 'decTaskConf',
                      'eqTaskConf', 'incTaskConf', 'coarse']:
        for dset in datasets:
            if dset.name == dset_name:
                ax.plot(x_axis, dset.data, color=dset.color, label=dset.name, linewidth=1.5,
                        marker=dset.marker, linestyle=dset.linestyle, markersize=dset.markersize)
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
    ax.set_xlim(-1, args.num_classes + 1)
    # Legend
    leg = ax.legend(bbox_to_anchor=(0., 1.20, 1., 0.1), loc='upper center', ncol=4, mode="expand", fancybox=True,
                    prop={'family': 'serif', 'size': 11})

    # Format plot and save figure
    plt.tight_layout()
    fig.savefig(os.path.join(args.path, '{}_{}_plot.png'.format(args.experiment, args.approach)))
    fig.savefig(os.path.join(args.path, '{}_{}_plot.pdf'.format(args.experiment, args.approach)))
    if args.show_plot:
        pylab.show()


class DsetPlotter:
    """ Class to manage all info to plot curves"""
    def __init__(self, dataset_name, exp_file):
        self.lookup_table = {
                             'cifar100': {'name': 'random', 'color': '#666666', 'type': 'baseline'},
                             'cifar100_icarl': {'name': 'iCaRL seed', 'color': '#0173b2', 'type': 'baseline'},
                             'cifar100_corse_grained': {'name': 'coarse', 'color': '#ca9161', 'type': 'dataset'},
                             'cifar100_cm_max': {'name': 'maxConf.', 'color': '#56b4e9', 'type': 'dataset'},
                             'cifar100_cm_min': {'name': 'minConf.', 'color': '#de8f05', 'type': 'dataset'},
                             'cifar100_10tasks_cm_dec': {'name': 'decTaskConf', 'color': '#d55e00', 'type': 'dataset'},
                             'cifar100_10tasks_cm_eq': {'name': 'eqTaskConf', 'color': '#cc78bc', 'type': 'dataset'},
                             'cifar100_10tasks_cm_inc': {'name': 'incTaskConf', 'color': '#029e73', 'type': 'dataset'}
        }
        self.name = self.lookup_table[dataset_name]['name']
        self.type = self.lookup_table[dataset_name]['type']
        self.color = self.lookup_table[dataset_name]['color']

        # open file and extract data
        self.file = exp_file
        # if is a list, the seeds mean needs to be calculated
        if isinstance(self.file, list):
            self.data = None
            counter = 0
            for m in range(len(self.file)):
                with open(self.file[m]) as f:
                    data = f.read().splitlines()
                    # check if the experiment is finished
                    if '0.0' in data[0].split():
                        print('removing a seed')
                        continue
                    counter += 1
                    if self.data is None:
                        self.data = [float(elem) * 100.0 for elem in data[0].split()]
                    else:
                        self.data = [x + float(y) * 100.0 for (x, y) in zip(self.data, data[0].split())]
            self.data = [elem / (1.0 * counter) for elem in self.data]
        else:
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
