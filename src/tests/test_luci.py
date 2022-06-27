from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp_name local_test --datasets mnist" \
                       " --network LeNet --num_tasks 3 --seed 1 --batch_size 32" \
                       " --nepochs 3" \
                       " --num_workers 0" \
                       " --gridsearch_tasks -1" \
                       " --approach luci"


def test_luci_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num_exemplars_per_class 20"
    run_main_and_assert(args_line)


def test_luci_exemplars_with_gridsearch():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num_exemplars_per_class 20"
    args_line = args_line.replace('--gridsearch_tasks -1', '--gridsearch_tasks 3')
    run_main_and_assert(args_line)


def test_luci_exemplars_orig_scheduler():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num_exemplars_per_class 20"
    args_line += " --orig_scheduler"
    run_main_and_assert(args_line)


def test_luci_exemplars_orig_scheduler_remove_less_forget():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num_exemplars_per_class 20"
    args_line += " --orig_scheduler"
    args_line += " --remove_less_forget"
    run_main_and_assert(args_line)


def test_luci_exemplars_orig_scheduler_remove_margin_ranking():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num_exemplars_per_class 20"
    args_line += " --orig_scheduler"
    args_line += " --remove_margin_ranking"
    run_main_and_assert(args_line)


def test_luci_exemplars_orig_scheduler_remove_adapt_lamda():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num_exemplars_per_class 20"
    args_line += " --orig_scheduler"
    args_line += " --remove_adapt_lamda"
    run_main_and_assert(args_line)


def test_luci_exemplars_warmup():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num_exemplars_per_class 20"
    args_line += " --warmup_nepochs 5"
    args_line += " --warmup_lr_factor 0.5"
    run_main_and_assert(args_line)
