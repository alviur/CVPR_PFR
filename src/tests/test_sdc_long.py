from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp_name sdc_local --datasets cifar100" \
                       " --network resnet32 --num_tasks 10 --stop_at_task 3 --seed 1 --batch_size 256" \
                       " --nepochs 200 --lr 0.0001 --momentum 0.9 --weight_decay 5e-4 --lr_patience 15" \
                       " --lr_factor 10 --lr_min 0.000001" \
                       " --num_workers 0"


def test_no_sdc():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach sdc"
    run_main_and_assert(args_line)


def test_sdc():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach sdc --sdc"
    run_main_and_assert(args_line)
