from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp_name local_test --datasets mnist" \
                       " --network LeNet --num_tasks 3 --seed 1 --batch_size 32" \
                       " --nepochs 3" \
                       " --num_workers 0" \
                       " --approach lwm --gradcam_layer conv2"


def test_lwm_without_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --log_gradcam_samples 16"
    run_main_and_assert(FAST_LOCAL_TEST_ARGS)


def test_lwm_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num_exemplars 200"
    run_main_and_assert(args_line)


def test_lwm_with_warmup():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --warmup_nepochs 5"
    args_line += " --warmup_lr_factor 0.5"
    args_line += " --num_exemplars 200"
    run_main_and_assert(args_line)
