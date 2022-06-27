from tests import run_main_and_assert
import pytest

EPOCHS = 200
RESULT_DIR = 'wlwf_mnist_beta'

TEST_ARGS = f"--datasets mnist" \
            f" --network LeNet --num_tasks 3 --seed 1 --batch_size 32 --num_workers 0" \
            f" --nepochs {EPOCHS} --lr 0.01 --momentum 0.9 --weight_decay 5e-4 --lr_patience 10" \
            f" --approach lwf_ent"


@pytest.mark.parametrize("beta", [0.0, 0.1, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
@pytest.mark.parametrize("a", [1.0, 2.0, 0.5])
def test_wlwf_with_beta(beta, a):
    args = f"{TEST_ARGS} --exp_name l0_a{a}_b{beta} --lamb 0.0 --alpha {a} --beta {beta}"
    run_main_and_assert(args, result_dir=RESULT_DIR)
