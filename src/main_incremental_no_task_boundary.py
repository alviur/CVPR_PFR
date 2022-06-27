import os
import time
import torch
import argparse
import importlib
import numpy as np
from functools import reduce
import utils
import approach
from loggers.exp_logger import MultiLogger
from datasets.data_loader import get_loaders
from datasets.dataset_config import dataset_config
from networks import tvmodels, allmodels, set_tvmodel_head_var
from datasets.sampler import ContinuousMultinomialSampler, _create_task_probs
import lightly


def main(argv=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description='Incremental Learning Framework')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--results_path', type=str, default='/data/experiments/LLL/', help='(default=%(default)s)')
    parser.add_argument('--exp_name', default=None, type=str, help='(default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--log', default=['disk', 'tensorboard'], type=str, choices=['disk', 'tensorboard'],
                        help='(default=%(default)s)', nargs='*')
    parser.add_argument('--save_models', action='store_true', help='(default=%(default)s)')
    parser.add_argument('--last_layer_analysis', action='store_true', help='(default=%(default)s)')
    # data args
    parser.add_argument('--datasets', default=['cifar100'], type=str, choices=list(dataset_config.keys()),
                        help='(default=%(default)s)', nargs='+')
    parser.add_argument('--num_workers', default=4, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--pin_memory', default=False, type=bool, required=False, help='(default=%(default)d)')
    parser.add_argument('--num_tasks', default=4, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--nc_first_task', default=None, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--use_valid_only', action='store_true', help='(default=%(default)s)')
    parser.add_argument('--stop_at_task', default=0, type=int, required=False, help='(default=%(default)d)')
    # model args
    parser.add_argument('--network', default='resnet32', type=str, choices=allmodels,
                        help='(default=%(default)s)')
    parser.add_argument('--not_remove_existing_head', action='store_true', help='(default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true', help='(default=%(default)s)')
    # training args
    parser.add_argument('--approach', default='finetune', type=str, choices=approach.__all__,
                        help='(default=%(default)s)')
    parser.add_argument('--nepochs', default=2, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch_size', default=64, type=int, required=False, help='(default=%(default)s)')
    parser.add_argument('--lr', default=0.1, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--lr_min', default=1e-4, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--lr_factor', default=3, type=float, required=False, help='(default=%(default)s)')
    parser.add_argument('--lr_patience', default=5, type=int, required=False, help='(default=%(default)s)')
    parser.add_argument('--clipping', default=10000, type=float, required=False, help='(default=%(default)s)')
    parser.add_argument('--momentum', default=0.0, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--weight_decay', default=0, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--warmup_nepochs', default=0, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--warmup_lr_factor', default=1.0, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--multi_softmax', action='store_true', help='(default=%(default)s)')
    parser.add_argument('--fix_bn', action='store_true', help='(default=%(default)s)')
    parser.add_argument('--no_cudnn_deterministic', action='store_true', help='(default=%(default)s)')
    parser.add_argument('--eval_on_train', action='store_true', help='(default=%(default)s)')

    # self-supervised args
    parser.add_argument('--eval_omni_head', default=True, action='store_true', help='(default=%(default)s)')
    parser.add_argument('--eval_fresh_head', default=False, action='store_true', help='(default=%(default)s)')
    parser.add_argument('--save_activations', default=False, action='store_true', help='(default=%(default)s)')
    parser.add_argument('--no_task_boundary_beta', default=4, type=int, help='(default=%(default)s)')
    parser.add_argument("--task1_nepochs", type=int, default=1500)

    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args(argv)
    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                       lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                       wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train)

    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    utils.seed_everything(seed=args.seed)
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)

    # Args -- CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'
    # Multiple gpus
    # if torch.cuda.device_count() > 1:
    #     self.C = torch.nn.DataParallel(C)
    #     self.C.to(self.device)
    ####################################################################################################################

    # Args -- Network
    # from networks.network import LLL_Net
    # if args.network in tvmodels:  # torchvision models
    #     tvnet = getattr(importlib.import_module(name='torchvision.models'), args.network)
    #     init_model = tvnet(pretrained=args.pretrained)
    #     set_tvmodel_head_var(init_model)
    # else:  # other models declared in networks package's init
    #     net = getattr(importlib.import_module(name='networks'), args.network)
    #     # WARNING: fixed to pretrained False for other model (non-torchvision)
    #     init_model = net(pretrained=False)

    # Args -- Continual Learning Approach
    from approach.learning_approach import Learning_Appr
    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
    assert issubclass(Appr, Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # Args -- Exemplars Management
    from datasets.exemplars_dataset import ExemplarsDataset
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
        print('Exemplars dataset arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    # assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))
    ####################################################################################################################

    # Log all arguments
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    full_exp_name += '_' + args.approach
    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name
    logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models)
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__))

    # Loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(args.datasets, args.num_tasks, args.nc_first_task,
                                                              args.batch_size, num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory)
    # Apply arguments for loaders
    if args.use_valid_only:
        tst_loader = val_loader
    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

    # Network and Approach instances
    # utils.seed_everything(seed=args.seed)
    # net = LLL_Net(init_model, remove_existing_head=not args.not_remove_existing_head)
    net = None
    utils.seed_everything(seed=args.seed)
    # taking transformations and class indices from first train dataset
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
    appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
    if Appr_ExemplarsDataset:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(transform, class_indices,
                                                                 **appr_exemplars_dataset_args.__dict__)
    utils.seed_everything(seed=args.seed)
    appr = Appr(net, device, **appr_kwargs)

    # the below code for no task boundary is taken from EBM method:
    # https://github.com/ShuangLI59/ebm-continual-learning/blob/main/data_loader/data_loader_online.py#L205-L256
    print('Prepare dataset without bounduaries....')
    labels_per_task = []
    sidx = 0
    for tid, nc in taskcla:
        eidx = sidx + nc
        labels_per_task.append(class_indices[sidx:eidx])
        sidx = eidx

    print('Labels per task:')
    print(labels_per_task)

    epc_per_virtual_task = args.nepochs
    iterations_per_virtual_epc = int(len(trn_loader[0].dataset.labels) / args.batch_size) + 1
    print('Epoch per virtual task: ', epc_per_virtual_task)
    print('Iteration per virtual task: ', iterations_per_virtual_epc)
    total_num_epochs = epc_per_virtual_task * len(labels_per_task)
    total_iters = total_num_epochs * iterations_per_virtual_epc
    overall_tasks_probs_over_iterations = [
        _create_task_probs(total_iters, len(labels_per_task), task_id, beta=args.no_task_boundary_beta) for task_id in
        range(len(labels_per_task))]

    normalize_probs = torch.zeros_like(overall_tasks_probs_over_iterations[0])
    for probs in overall_tasks_probs_over_iterations:
        normalize_probs.add_(probs)
    for probs in overall_tasks_probs_over_iterations:
        probs.div_(normalize_probs)
    tasks_probs_over_iterations = torch.cat(overall_tasks_probs_over_iterations).view(-1,
                                                                                      overall_tasks_probs_over_iterations[
                                                                                          0].shape[0])
    tasks_probs_over_iterations_lst = []
    for col in range(tasks_probs_over_iterations.shape[1]):
        tasks_probs_over_iterations_lst.append(tasks_probs_over_iterations[:, col])
    tasks_probs_over_iterations = tasks_probs_over_iterations_lst

    train_datasets = [dl.dataset for dl in trn_loader]
    tasks_samples_indices = []
    total_len = 0
    for train_dataset in train_datasets:
        tasks_samples_indices.append(torch.tensor(range(total_len, total_len + len(train_dataset)), dtype=torch.int32))
        total_len += len(train_dataset)

    all_datasets = torch.utils.data.ConcatDataset(train_datasets)
    all_val_datasets = torch.utils.data.ConcatDataset([dl.dataset for dl in val_loader])

    # prepare CIFAR-train
    cifar_normalize = {'mean': [0.5071, 0.4866, 0.4409], 'std': [0.2009, 0.1984, 0.2023]}
    collate_fn = lightly.data.SimCLRCollateFunction(input_size=32, gaussian_blur=0., normalize=cifar_normalize,
                                                    cj_strength=0.5)
    trainD = lightly.data.LightlyDataset.from_torch_dataset(all_datasets)
    valD = lightly.data.LightlyDataset.from_torch_dataset(all_val_datasets)

    train_sampler = ContinuousMultinomialSampler(
        data_source=trainD,
        samples_in_batch=args.batch_size,
        tasks_samples_indices=tasks_samples_indices,
        tasks_probs_over_iterations=tasks_probs_over_iterations,
        num_of_batches=len(tasks_probs_over_iterations)
    )

    dataloader_train_ssl = torch.utils.data.DataLoader(
        trainD,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        # drop_last=True,
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=args.pin_memory,
    )

    dataloader_val = torch.utils.data.DataLoader(
        valD,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    # train single head on all classes
    omni_trn_loader, omni_val_loader, omni_tst_loader, omni_taskcla = get_loaders(
        args.datasets, 1, None, args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory
    )

    appr.train(0, dataloader_train_ssl, dataloader_val, len(class_indices), omni_trn_loader, omni_val_loader,
               omni_tst_loader)


if __name__ == '__main__':
    # import debug, sys, json
    #
    # print('ARGV:', json.dumps(sys.argv))
    # debug.hook_debugger_to_signal()
    main()