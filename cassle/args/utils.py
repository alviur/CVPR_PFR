import os
from argparse import Namespace

N_CLASSES_PER_DATASET = {
    "cifar10": 10,
    "cifar100": 100,
    "stl10": 10,
    "imagenet": 1000,
    "imagenet100": 100,
    "domainnet": 345,
}


def additional_setup_pretrain(args: Namespace):
    """Provides final setup for pretraining to non-user given parameters by changing args.

    Parsers arguments to extract the number of classes of a dataset, create
    transformations kwargs, correctly parse gpus, identify if a cifar dataset
    is being used and adjust the lr.

    Args:
        args (Namespace): object that needs to contain, at least:
        - dataset: dataset name.
        - brightness, contrast, saturation, hue, min_scale: required augmentations
            settings.
        - multicrop: flag to use multicrop.
        - dali: flag to use dali.
        - optimizer: optimizer name being used.
        - gpus: list of gpus to use.
        - lr: learning rate.

        [optional]
        - gaussian_prob, solarization_prob: optional augmentations settings.
    """

    args.transform_kwargs = {}

    if args.dataset in N_CLASSES_PER_DATASET:
        args.num_classes = N_CLASSES_PER_DATASET[args.dataset]
    else:
        # hack to maintain the current pipeline
        # even if the custom dataset doesn't have any labels
        dir_path = args.data_dir / args.train_dir
        args.num_classes = max(
            1,
            len([entry.name for entry in os.scandir(dir_path) if entry.is_dir]),
        )

    unique_augs = max(
        len(p)
        for p in [
            args.brightness,
            args.contrast,
            args.saturation,
            args.hue,
            args.gaussian_prob,
            args.solarization_prob,
            args.min_scale,
            args.size,
        ]
    )
    assert unique_augs == args.num_crops or unique_augs == 1

    # assert that either all unique augmentation pipelines have a unique
    # parameter or that a single parameter is replicated to all pipelines
    for p in [
        "brightness",
        "contrast",
        "saturation",
        "hue",
        "gaussian_prob",
        "solarization_prob",
        "min_scale",
        "size",
    ]:
        values = getattr(args, p)
        n = len(values)
        assert n == unique_augs or n == 1

        if n == 1:
            setattr(args, p, getattr(args, p) * unique_augs)

    args.unique_augs = unique_augs

    if unique_augs > 1:
        args.transform_kwargs = [
            dict(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                gaussian_prob=gaussian_prob,
                solarization_prob=solarization_prob,
                min_scale=min_scale,
                size=size,
            )
            for (
                brightness,
                contrast,
                saturation,
                hue,
                gaussian_prob,
                solarization_prob,
                min_scale,
                size,
            ) in zip(
                args.brightness,
                args.contrast,
                args.saturation,
                args.hue,
                args.gaussian_prob,
                args.solarization_prob,
                args.min_scale,
                args.size,
            )
        ]

    elif not args.multicrop:
        args.transform_kwargs = dict(
            brightness=args.brightness[0],
            contrast=args.contrast[0],
            saturation=args.saturation[0],
            hue=args.hue[0],
            gaussian_prob=args.gaussian_prob[0],
            solarization_prob=args.solarization_prob[0],
            min_scale=args.min_scale[0],
            size=args.size[0],
        )
    else:
        args.transform_kwargs = dict(
            brightness=args.brightness[0],
            contrast=args.contrast[0],
            saturation=args.saturation[0],
            hue=args.hue[0],
            gaussian_prob=args.gaussian_prob[0],
            solarization_prob=args.solarization_prob[0],
        )

    # add support for custom mean and std
    if args.dataset == "custom":
        if isinstance(args.transform_kwargs, dict):
            args.transform_kwargs["mean"] = args.mean
            args.transform_kwargs["std"] = args.std
        else:
            for kwargs in args.transform_kwargs:
                kwargs["mean"] = args.mean
                kwargs["std"] = args.std

    if args.dataset in ["cifar10", "cifar100", "stl10"]:
        if isinstance(args.transform_kwargs, dict):
            del args.transform_kwargs["size"]
        else:
            for kwargs in args.transform_kwargs:
                del kwargs["size"]

    args.cifar = True if args.dataset in ["cifar10", "cifar100"] else False

    if args.dali:
        assert args.dataset in ["imagenet100", "imagenet", "domainnet", "custom"]

    args.extra_optimizer_args = {}
    if args.optimizer == "sgd":
        args.extra_optimizer_args["momentum"] = 0.9

    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]
    elif isinstance(args.gpus, str):
        args.gpus = [int(gpu) for gpu in args.gpus.split(",") if gpu]

    # adjust lr according to batch size
    args.lr = args.lr * args.batch_size * len(args.gpus) / 256


def additional_setup_linear(args: Namespace):
    """Provides final setup for linear evaluation to non-user given parameters by changing args.

    Parsers arguments to extract the number of classes of a dataset, correctly parse gpus, identify
    if a cifar dataset is being used and adjust the lr.

    Args:
        args: Namespace object that needs to contain, at least:
        - dataset: dataset name.
        - optimizer: optimizer name being used.
        - gpus: list of gpus to use.
        - lr: learning rate.
    """

    assert args.dataset in N_CLASSES_PER_DATASET
    args.num_classes = N_CLASSES_PER_DATASET[args.dataset]

    args.cifar = True if args.dataset in ["cifar10", "cifar100"] else False

    if args.dali:
        assert args.dataset in ["imagenet100", "imagenet", "domainnet"]

    args.extra_optimizer_args = {}
    if args.optimizer == "sgd":
        args.extra_optimizer_args["momentum"] = 0.9

    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]
    elif isinstance(args.gpus, str):
        args.gpus = [int(gpu) for gpu in args.gpus.split(",") if gpu]
