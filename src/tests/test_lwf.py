from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp_name local_test --datasets mnist" \
                       " --network LeNet --num_tasks 3 --seed 1 --batch_size 32" \
                       " --nepochs 3" \
                       " --num_workers 0" \
                       " --approach lwf"


def test_lwf_without_exemplars():
    run_main_and_assert(FAST_LOCAL_TEST_ARGS)


def test_lwf_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num_exemplars 200"
    run_main_and_assert(args_line)


def test_lwf_with_exemplars_kd_only_new():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num_exemplars_per_class 10 --kd_only_new"
    run_main_and_assert(args_line)


def test_lwf_with_no_exemplars_kd_only_new():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num_exemplars_per_class 0 --kd_only_new"
    run_main_and_assert(args_line)


def test_lwf_with_exemplars_kd_only_new_and_ce_exemplars_task_aware():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num_exemplars_per_class 10 --kd_only_new --ce_exemplars_task_aware"
    run_main_and_assert(args_line)


def test_lwf_with_warmup():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --warmup_nepochs 5"
    args_line += " --warmup_lr_factor 0.5"
    args_line += " --num_exemplars 200"
    run_main_and_assert(args_line)
