import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')


def last_layer_analysis(heads, task, taskcla, y_lim=False, sort_weights=False):
    print('Plotting last layer analysis...')
    num_classes = sum([x for (_, x) in taskcla])
    weights, biases, indexes = [], [], []
    class_id = 0
    with torch.no_grad():
        for t in range(task + 1):
            n_classes_t = taskcla[t][1]
            indexes.append(np.arange(class_id, class_id + n_classes_t))
            if type(heads) == torch.nn.Linear:  # Single head
                biases.append(heads.bias[class_id: class_id + n_classes_t].detach().cpu().numpy())
                weights.append((heads.weight[class_id: class_id + n_classes_t] ** 2).sum(1).sqrt().detach().cpu().numpy())
            else:  # Multi-head
                weights.append((heads[t].weight ** 2).sum(1).sqrt().detach().cpu().numpy())
                if type(heads[t]) == torch.nn.Linear:
                    biases.append(heads[t].bias.detach().cpu().numpy())
                else:
                    biases.append(np.zeros(weights[-1].shape))  # For LUCI
            class_id += n_classes_t

    # Figure weights
    f_weights = plt.figure(dpi=300)
    ax = f_weights.subplots(nrows=1, ncols=1)
    for i, (x, y) in enumerate(zip(indexes, weights), 0):
        if sort_weights:
            ax.bar(x, sorted(y, reverse=True), label="Task {}".format(i))
        else:
            ax.bar(x, y, label="Task {}".format(i))
    ax.set_xlabel("Classes", fontsize=11, fontfamily='serif')
    ax.set_ylabel("L2 norms of the weights", fontsize=11, fontfamily='serif')
    if num_classes is not None:
        ax.set_xlim(0, num_classes)
    if y_lim:
        ax.set_ylim(0, 5)
    ax.legend(loc='upper left', fontsize='11', fontfamily='serif')

    # Figure weights
    f_biases = plt.figure(dpi=300)
    ax = f_biases.subplots(nrows=1, ncols=1)
    for i, (x, y) in enumerate(zip(indexes, biases), 0):
        if sort_weights:
            ax.bar(x, sorted(y, reverse=True), label="Task {}".format(i))
        else:
            ax.bar(x, y, label="Task {}".format(i))
    ax.set_xlabel("Classes", fontsize=11, fontfamily='serif')
    ax.set_ylabel("Values of the biases", fontsize=11, fontfamily='serif')
    if num_classes is not None:
        ax.set_xlim(0, num_classes)
    if y_lim:
        ax.set_ylim(-1.0, 1.0)
    ax.legend(loc='upper left', fontsize='11', fontfamily='serif')

    return f_weights, f_biases
