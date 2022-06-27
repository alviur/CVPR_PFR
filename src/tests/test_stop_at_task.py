from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp_name local_test --datasets mnist" \
                       " --network LeNet --num_tasks 5 --seed 1 --batch_size 32" \
                       " --nepochs 2 --num_workers 0 --stop_at_task 3"


def test_finetuning_stop_at_task():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetune"
    run_main_and_assert(args_line)
