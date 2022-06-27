from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp_name local_test --datasets cifar100" \
                       " --network resnet32 --num_tasks 10 --stop_at_task 3 --seed 1 --batch_size 256" \
                       " --nepochs 3" \
                       " --num_workers 0"


def test_metric_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num_exemplars 200"
    args_line += " --approach metric"
    run_main_and_assert(args_line)


def test_sscl_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach sscl"
    args_line += " --num_exemplars 200"
    run_main_and_assert(args_line)


def test_augmix_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach mixup"
    args_line += " --num_exemplars 200"
    args_line += " --augmix"
    run_main_and_assert(args_line)


def test_mixup_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach mixup"
    args_line += " --num_exemplars 200"
    args_line += " --mixup"
    run_main_and_assert(args_line)


def test_sdc():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach sdc"
    run_main_and_assert(args_line)