from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp_name local_test --datasets mnist" \
                       " --network LeNet --num_tasks 3 --seed 1 --batch_size 32" \
                       " --nepochs 5 --lr_factor 10 --momentum 0.9 --lr_min 1e-7" \
                       " --num_workers 0 --lr 0.01"

# lr = 0.12 * (opt.batch_size / 256)


def test_triplet():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach emb_align"
    args_line += " --metric_loss triplet"
    run_main_and_assert(args_line)


def test_triplet_with_margin_and_mining():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach emb_align"
    args_line += " --metric_loss triplet"
    args_line += " --margin 0.5"
    args_line += " --mining"
    run_main_and_assert(args_line)


def test_align_uniform():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach emb_align"
    args_line += " --metric_loss align_uniform"
    # args_line += " --mining"
    run_main_and_assert(args_line)
