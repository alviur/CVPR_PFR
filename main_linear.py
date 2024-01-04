import os
import types

import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torchvision.models import resnet18, resnet50

from cassle.args.setup import parse_args_linear

try:
    from cassle.methods.dali import ClassificationABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True
from cassle.methods.linear import LinearModel
from cassle.utils.classification_dataloader import prepare_data
from cassle.utils.checkpointer import Checkpointer


def main():
    seed_everything(5)

    args = parse_args_linear()

    # split classes into tasks
    tasks = None
    if args.split_strategy == "class":
        assert args.num_classes % args.num_tasks == 0
        tasks = torch.randperm(args.num_classes).chunk(args.num_tasks)

    if args.encoder == "resnet18":
        backbone = resnet18()
    elif args.encoder == "resnet50":
        backbone = resnet50()
    else:
        raise ValueError("Only [resnet18, resnet50] are currently supported.")

    if args.cifar:
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()

    assert (
        args.pretrained_feature_extractor.endswith(".ckpt")
        or args.pretrained_feature_extractor.endswith(".pth")
        or args.pretrained_feature_extractor.endswith(".pt")
    )
    ckpt_path = args.pretrained_feature_extractor

    state = torch.load(ckpt_path)["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder.", "")] = state[k]
        del state[k]
    backbone.load_state_dict(state, strict=False)

    print(f"Loaded {ckpt_path}")

    if args.dali:
        assert _dali_avaliable, "Dali is not currently avaiable, please install it first."
        MethodClass = types.new_class(
            f"Dali{LinearModel.__name__}", (ClassificationABC, LinearModel)
        )
    else:
        MethodClass = LinearModel

    model = MethodClass(backbone, **args.__dict__, tasks=tasks)

    train_loader, val_loader = prepare_data(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        semi_supervised=args.semi_supervised,
    )

    callbacks = []

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name, project=args.project, entity=args.entity, offline=args.offline
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, "linear"),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=True),
        checkpoint_callback=False,
        terminate_on_nan=True,
        accelerator="ddp",
    )
    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
