from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp_name local_test --datasets mnist" \
                       " --network LeNet --num_tasks 3 --seed 1 --batch_size 32" \
                       " --nepochs 3 --num_workers 0 --num_exemplars 200" \
                       " --gridsearch_tasks 3 --gridsearch_config gridsearch_config" \
                       " --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5"


def test_gridsearch_finetune():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetune"
    run_main_and_assert(args_line)


def test_gridsearch_freezing():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach freezing"
    run_main_and_assert(args_line)


def test_gridsearch_joint():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach joint"
    run_main_and_assert(args_line)


def test_gridsearch_lwf():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach lwf"
    run_main_and_assert(args_line)


def test_gridsearch_icarl():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach icarl"
    run_main_and_assert(args_line)


def test_gridsearch_eeil():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach eeil --nepochs_finetuning 3"
    run_main_and_assert(args_line)


def test_gridsearch_bic():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach bic --bias_epochs 3"
    run_main_and_assert(args_line)


def test_gridsearch_luci():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach luci"
    run_main_and_assert(args_line)


def test_gridsearch_lwm():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach lwm --gradcam_layer conv2 --log_gradcam_samples 16"
    run_main_and_assert(args_line)


def test_gridsearch_ewc():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach ewc"
    run_main_and_assert(args_line)


def test_gridsearch_mas():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach mas"
    run_main_and_assert(args_line)


def test_gridsearch_pathint():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach path_integral"
    run_main_and_assert(args_line)


def test_gridsearch_rwalk():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach r_walk"
    run_main_and_assert(args_line)


def test_gridsearch_dmc():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach dmc"
    args_line += " --aux_dataset mnist"  # just to test the grid search fast
    run_main_and_assert(args_line)
