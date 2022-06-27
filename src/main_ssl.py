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
from last_layer_analysis import last_layer_analysis
from networks import tvmodels, allmodels, set_tvmodel_head_var


def main(argv=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description='Incremental Learning Framework with SSL')

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
    # gridsearch args
    parser.add_argument('--gridsearch_tasks', default=-1, type=int, help='(default=%(default)d)')
    # self-supervised args
    parser.add_argument('--eval_omni_head', action='store_true', help='(default=%(default)s)')
    parser.add_argument('--eval_fresh_head', action='store_true', help='(default=%(default)s)')

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
    from networks.network import LLL_Net
    if args.network in tvmodels:  # torchvision models
        tvnet = getattr(importlib.import_module(name='torchvision.models'), args.network)
        init_model = tvnet(pretrained=args.pretrained)
        set_tvmodel_head_var(init_model)
    else:  # other models declared in networks package's init
        net = getattr(importlib.import_module(name='networks'), args.network)
        # WARNING: fixed to pretrained False for other model (non-torchvision)
        init_model = net(pretrained=False)

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

    # Args -- GridSearch
    if args.gridsearch_tasks > 0:
        from gridsearch import GridSearch
        gs_args, extra_args = GridSearch.extra_parser(extra_args)
        Appr_finetune = getattr(importlib.import_module(name='approach.finetune'), 'Appr')
        assert issubclass(Appr_finetune, Learning_Appr)
        GridSearch_ExemplarsDataset = Appr.exemplars_dataset_class()
        print('GridSearch arguments =')
        for arg in np.sort(list(vars(gs_args).keys())):
            print('\t' + arg + ':', getattr(gs_args, arg))
        print('=' * 108)

    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))
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
    utils.seed_everything(seed=args.seed)
    net = LLL_Net(init_model, remove_existing_head=not args.not_remove_existing_head)
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

    # GridSearch
    if args.gridsearch_tasks > 0:
        ft_kwargs = {**base_kwargs, **dict(logger=logger,
                                           exemplars_dataset=GridSearch_ExemplarsDataset(transform, class_indices))}
        appr_ft = Appr_finetune(net, device, **ft_kwargs)
        gridsearch = GridSearch(appr_ft, args.seed, gs_args.gridsearch_config, gs_args.gridsearch_acc_drop_thr,
                                gs_args.gridsearch_hparam_decay, gs_args.gridsearch_max_num_searches)

    # Loop tasks
    print(taskcla)
    net.task_offset = np.array([0] + [ncls for _, ncls in taskcla]).cumsum()[:-1]
    print('Task offsets: ', net.task_offset)

    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))
    all_splited_tasks = len(trn_loader)
    fresh_head_acc_taw = np.zeros((all_splited_tasks, all_splited_tasks))
    fresh_head_acc_tag = np.zeros((all_splited_tasks, all_splited_tasks))
    omni_head_acc_taw = np.zeros(max_task)
    omni_head_acc_tag = np.zeros(max_task)
    for t, (_, ncla) in enumerate(taskcla):
        # Early stop tasks if flag
        if t >= max_task:
            continue

        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        # Add head for current task
        net.add_head(taskcla[t][1])
        net.to(device)

        # GridSearch
        if t < args.gridsearch_tasks:

            # Search for best finetune learning rate -- Maximal Plasticity Search
            print('LR GridSearch')
            best_ft_acc, best_ft_lr = gridsearch.search_lr(appr.model, t, trn_loader[t], val_loader[t])
            # Apply to approach
            appr.lr = best_ft_lr
            gen_params = gridsearch.gs_config.get_params('general')
            for k, v in gen_params.items():
                if not isinstance(v, list):
                    setattr(appr, k, v)

            # Search for best forgetting/intransigence tradeoff -- Stability Decay
            print('Trade-off GridSearch')
            best_tradeoff, tradeoff_name = gridsearch.search_tradeoff(args.approach, appr,
                                                                      t, trn_loader[t], val_loader[t], best_ft_acc)
            # Apply to approach
            if tradeoff_name is not None:
                setattr(appr, tradeoff_name, best_tradeoff)

            print('-' * 108)

        # Train
        appr.train(t, trn_loader[t], val_loader[t])
        print('-' * 108)
        top1 = appr.eval_knn(t, tst_loader[t], trn_loader[t])
        logger.log_scalar(task=None, iter=t, value=top1, name="top1_knn_current_task")

        # Test
        for u in range(t + 1):
            test_loss, acc_taw[t, u], acc_tag[t, u] = appr.eval(u, tst_loader[u])
            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                  '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                 100 * acc_taw[t, u], 100 * forg_taw[t, u],
                                                                 100 * acc_tag[t, u], 100 * forg_tag[t, u]))

            # logger.tbwriter.add_scalar('overall/test_loss', test_loss, u + 1)
            logger.log_scalar(task=t, iter=u, name='loss', group='test', value=test_loss)
            # logger.tbwriter.add_scalar('overall/test_acc_taw', 100 * acc_taw[t, u], u + 1)
            logger.log_scalar(task=t, iter=u, name='acc_taw', group='test', value=100 * acc_taw[t, u])
            # logger.tbwriter.add_scalar('overall/test_acc_tag', 100 * acc_tag[t, u], u + 1)
            logger.log_scalar(task=t, iter=u, name='acc_tag', group='test', value=100 * acc_tag[t, u])
            # logger.tbwriter.add_scalar('overall/test_forg_taw', 100 * forg_taw[t, u], u + 1)
            logger.log_scalar(task=t, iter=u, name='forg_taw', group='test', value=100 * forg_taw[t, u])
            # logger.tbwriter.add_scalar('overall/test_forg_tag', 100 * forg_tag[t, u], u + 1)
            logger.log_scalar(task=t, iter=u, name='forg_tag', group='test', value=100 * forg_tag[t, u])

        # Save
        print('Save at ' + os.path.join(args.results_path, full_exp_name))
        logger.log_result(acc_taw, "acc_taw", t)
        logger.log_result(acc_tag, "acc_tag", t)
        logger.log_result(forg_taw, "forg_taw", t)
        logger.log_result(forg_tag, "forg_tag", t)
        logger.save_model(net.state_dict(), task=t)
        logger.log_result(acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1), "avg_accs_taw", t)
        logger.log_result(acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1), "avg_accs_tag", t)
        aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0))
        logger.log_result((acc_taw * aux).sum(1) / aux.sum(1), "wavg_accs_taw", t)
        logger.log_result((acc_tag * aux).sum(1) / aux.sum(1), "wavg_accs_tag", t)

        # Last layer analysis
        if args.last_layer_analysis:
            weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True)
            logger.log_figure(name='weights', iter=t, figure=weights)
            logger.log_figure(name='bias', iter=t, figure=biases)

            # Output sorted weights and biases
            weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True, sort_weights=True)
            logger.log_figure(name='weights', iter=t, figure=weights)
            logger.log_figure(name='bias', iter=t, figure=biases)


        # Additional evaluation for SS tasks
        SS_APPROACHES = ['simsiam']
        if args.approach in SS_APPROACHES and (args.eval_omni_head or args.eval_fresh_head):
            if args.eval_fresh_head:
                # train fresh heads
                fresh_heads = []
                for _t, (_, ncla) in enumerate(taskcla):
                    fresh_heads.append(
                        appr.train_downstream_classifier(
                            _t, trn_loader[_t], val_loader[_t], f"task-{t}-fresh-head-{_t}-classifier"
                        )
                    )

                # evaluate overa all tasks
                for _t, (_, ncla) in enumerate(taskcla):
                    test_loss, fresh_head_acc_taw[t, _t], fresh_head_acc_tag[t, _t] = \
                        appr.eval(_t, tst_loader[_t], heads_to_evaluate=fresh_heads)

                    print(
                        '[FRESH HEAD] >>> Task {:2d} -  Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}% '
                        '| TAg acc={:5.1f}% <<<'.format(
                            t, _t, test_loss, 100 * fresh_head_acc_taw[t, _t], 100 * fresh_head_acc_tag[t, _t]
                        )
                    )
                    logger.log_scalar(
                        task=t, iter=_t, name='fresh_head_acc_taw', group='test', value=100 * fresh_head_acc_taw[t, _t]
                    )
                    logger.log_scalar(
                        task=t, iter=_t, name='fresh_head_acc_tag', group='test', value=100 * fresh_head_acc_tag[t, _t]
                    )
                logger.log_result(fresh_head_acc_taw, "fresh_head_acc_taw", t)
                logger.log_result(fresh_head_acc_tag, "fresh_head_acc_tag", t)

            if args.eval_omni_head:
                # train single head on all classes
                omni_trn_loader, omni_val_loader, omni_tst_loader, omni_taskcla = get_loaders(
                    args.datasets, 1, None, args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory
                )

                # train and eval - like task '0', then offset is also '0'
                omni_head = appr.train_downstream_classifier(
                    0, omni_trn_loader[0], omni_val_loader[0], f"task-{t}-omni-head-classifier"
                )
                test_loss, omni_head_acc_taw[t], omni_head_acc_tag[t] = appr.eval(
                    0, omni_tst_loader[0], heads_to_evaluate=[omni_head]
                )
                print(
                    '[OMNI HEAD] >>> Task {:2d} - Test with omni-head: loss={:.3f} | TAw acc={:5.1f}% '
                    '| TAg acc={:5.1f}% <<<'.format(t, test_loss, 100 * omni_head_acc_taw[t], 100 * omni_head_acc_tag[t])
                )
                logger.log_scalar(
                    task=None, iter=t, name='omni_head_acc_tag', group='test', value=100 * omni_head_acc_tag[t]
                )
                logger.log_scalar(
                    task=None, iter=t, name='omni_head_acc_taw', group='test', value=100 * omni_head_acc_taw[t]
                )

    # Print Summary
    utils.print_summary(acc_taw, acc_tag, forg_taw, forg_tag)
    if fresh_head_acc_taw.sum() > 0:  # Self-supervision additional evaluation
        print('FRESH HEADS RESULTS:')
        utils.print_summary(fresh_head_acc_taw, fresh_head_acc_tag, None, None)
        print('OMNI HEADS RESULTS:')
        utils.print_summary(
            omni_head_acc_taw.reshape([1, max_task]), omni_head_acc_tag.reshape([1, max_task]), None, None
        )
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

    return acc_taw, acc_tag, forg_taw, forg_tag, logger.exp_path
    ####################################################################################################################


if __name__ == '__main__':
    main()
