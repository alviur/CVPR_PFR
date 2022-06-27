from tests import run_main_and_assert

# EPOCHS = 200
EPOCHS = 5
TEST_ARGS = f"--datasets mnist" \
            f" --network LeNet --num_tasks 3 --seed 1 --batch_size 32 --num_workers 0" \
            f" --nepochs {EPOCHS} --lr 0.01 --momentum 0.9 --weight_decay 5e-4 --lr_patience 10" \
            f" --approach lwf_ent_cos"


def test_wlwf():
    args = f"{TEST_ARGS} --exp_name l0_a1_b1 --lamb 0.0 --alpha 1.0 --beta 1.0"
    run_main_and_assert(args)


def test_wlwf_beta_zero():
    args = f"{TEST_ARGS} --exp_name l0_a1_b0 --lamb 0.0 --alpha 1.0 --beta 0.0"
    run_main_and_assert(args)


def test_lwf():
    args = f"{TEST_ARGS} --exp_name l1_a0 --lamb 1.0 --alpha 0.0 --beta 0.0 --lambda_cos_base 0.0"
    run_main_and_assert(args)


def test_lwf_wit_cos_dist():
    args = f"{TEST_ARGS} --exp_name l1_a0 --lamb 1.0 --alpha 0.0 --beta 0.0 --lambda_cos_base 5.0"
    run_main_and_assert(args)


def test_ft():
    args = f"{TEST_ARGS} --exp_name ft --lamb 0.0 --alpha 0.0 --beta 0.0 --lambda_cos_base 0.0"
    run_main_and_assert(args)


def test_both():
    args = f"{TEST_ARGS} --exp_name l1_a1_b1 --lamb 1.0 --alpha 1.0 --beta 1.0 --lambda_cos_base 0.0"
    run_main_and_assert(args)


def test_lwf_wit_cos_dist_1():
    args = f"{TEST_ARGS} --exp_name all_zero --lamb 0.0 --alpha 0.0 --beta 0.0 --lambda_cos_base 1.0"
    run_main_and_assert(args)


def test_lwf_wit_cos_dist_5():
    args = f"{TEST_ARGS} --exp_name all_zero --lamb 0.0 --alpha 0.0 --beta 0.0 --lambda_cos_base 5.0"
    run_main_and_assert(args)


def test_lwf_wit_cos_dist_50():
    args = f"{TEST_ARGS} --exp_name all_zero --lamb 0.0 --alpha 0.0 --beta 0.0 --lambda_cos_base 50.0"
    run_main_and_assert(args)