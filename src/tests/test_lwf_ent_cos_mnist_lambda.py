from tests import run_main_and_assert
import pytest

EPOCHS = 100
TEST_ARGS = f"--datasets mnist" \
            f" --network LeNet --num_tasks 3 --seed 1 --batch_size 32 --num_workers 0" \
            f" --nepochs {EPOCHS} --lr 0.01 --momentum 0.9 --weight_decay 5e-4 --lr_patience 3" \
            f" --approach lwf_ent_cos"


@pytest.mark.parametrize(
    "lc", [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
)
def test_lambda_cos(lc):
    args = f"{TEST_ARGS} --exp_name test_lambda_cos --lamb 0.0 --alpha 0.0 --beta 0.0 --lambda_cos_base {lc}"
    run_main_and_assert(args, result_dir='results_mnist_lambda_cos')