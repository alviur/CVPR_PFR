from tests import run_main_and_assert

# FAST_LOCAL_TEST_ARGS = "--gpu 8 --exp_name simsiam --datasets cifar100" \
#                        " --network ss_resnet18 --num_tasks 10 --seed 1 --batch_size 512" \
#                        " --nepochs 1 --lr_factor 10 --momentum 0.9 --weight_decay 1e-6 --" \
#                        " --num_workers 0 --stop_at_task 3 --lr_warmup_epochs 2 --classifier_nepochs 2"

# LONGER RUN
# FAST_LOCAL_TEST_ARGS = "--gpu 8 --exp_name simsiam --datasets cifar100" \
#                        " --network ss_resnet18 --num_tasks 10 --seed 1 --batch_size 512" \
#                        " --nepochs 300 --momentum 0.9 --weight_decay 1e-6" \
#                        " --num_workers 0 --stop_at_task 3 --lr_warmup_epochs 2 --classifier_nepochs 20"

# SINGLE TASK
FAST_LOCAL_TEST_ARGS = "--seed 1234 --gpu 3 --exp_name simsiam --datasets cifar100_noTrans" \
                       " --network LinearBase --num_tasks 1 --seed 1993 --batch_size 512" \
                       " --nepochs 800 --optim_name sgd --lr 0.03 --momentum 0.9 --weight_decay 0.005" \
                       " --num_workers 4 --classifier_nepochs 150 --hidden_mlp 512 --jitter_strength 0.5"


def test_simsiam():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach simsiam"
    run_main_and_assert(args_line, 0, 0)
