from tests import run_main

FAST_LOCAL_TEST_ARGS = "--exp_name local_test --datasets mnist" \
                       " --network LeNet --num_tasks 3 --seed 1 --batch_size 32" \
                       " --nepochs 2 --lr_factor 10 --momentum 0.9 --lr_min 1e-7" \
                       " --num_workers 0" \
                       " --approach finetune"


def test_last_layer_analysis():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --last_layer_analysis"
    run_main(args_line)

