from tests import run_main_and_assert

EPOCHS = 200
EPOCHS = 2
TEST_ARGS = f"--datasets mnist" \
            f" --network LeNet --num_tasks 3 --seed 1 --batch_size 32 --num_workers 0" \
            f" --nepochs {EPOCHS} --lr 0.01 --momentum 0.9 --weight_decay 5e-4 --lr_patience 10" \
            f" --approach lwf_ent"


def test_wlwf():
    args = f"{TEST_ARGS} --exp_name l0_a1_b1 --lamb 0.0 --alpha 1.0 --beta 1.0"
    run_main_and_assert(args)


def test_wlwf_beta_zero():
    args = f"{TEST_ARGS} --exp_name l0_a1_b0 --lamb 0.0 --alpha 1.0 --beta 0.0"
    run_main_and_assert(args)


def test_lwf():
    args = f"{TEST_ARGS} --exp_name l1_a0 --lamb 1.0 --alpha 0.0 --beta 0.0"
    run_main_and_assert(args)


def test_ft():
    args = f"{TEST_ARGS} --exp_name l0_a0 --lamb 0.0 --alpha 0.0 --beta 0.0"
    run_main_and_assert(args)


def test_both():
    args = f"{TEST_ARGS} --exp_name l1_a1_b1 --lamb 1.0 --alpha 1.0 --beta 1.0"
    run_main_and_assert(args)

# def test_lwf_cifar100():
# TEST_ARGS = "--exp_name lwf_ent --datasets cifar100_icarl" \
#                    " --network resnet32 --num_tasks 10 --stop_at_task 3 --seed 1 --batch_size 128" \
#                    " --nepochs 200 --lr 0.0001 --momentum 0.9 --weight_decay 5e-4 --lr_patience 15" \
#                    " --lr_factor 10 --lr_min 0.000001" \
#                    " --num_workers 0" \
#                    " --approach lwf_ent"
# run_main_and_assert(TEST_ARGS)
