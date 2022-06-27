from tests import run_main_and_assert

# 10 TASK
FAST_LOCAL_TEST_ARGS = "--gpu 1 --exp_name simsiam --datasets cifar100_noTrans" \
                       " --network ss_resnet18_lightly --num_tasks 1 --seed 1234 --batch_size 512" \
                       " --nepochs 800 --optim_name sgd --lr 0.03 --momentum 0.9 --weight_decay 0.005" \
                       " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
                       " --lr_patience 20 --lr_min 5e-7"


def test_simsiam():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach simsiam"
    run_main_and_assert(args_line, 0, 0, result_dir='result_simsiam')
