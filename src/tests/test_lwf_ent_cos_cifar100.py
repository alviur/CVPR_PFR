from tests import run_main_and_assert
import pytest

EPOCHS = 200

TEST_ARGS = f"--network resnet32 --seed 1 --batch_size 128" \
            f" --nepochs {EPOCHS} --lr 0.01 --momentum 0.9 --weight_decay 0.0002 --lr_patience 10" \
            f" --lr_factor 3 --lr_min 0.0001" \
            f" --num_workers 0" \
            f" --approach lwf_ent_cos"


@pytest.mark.parametrize(
    "params",
    [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.05), (0.0, 00.0, 0.0, 0.1)]
    # [(0.0, 20.0, 0.5, 0.0), (0.0, 20.0, 0.3, 0.0), (0.0, 0.0, 0.0, 5.0), (0.0, 1.0, 0.0, 5.0), (0.0, 1.0, 1.0, 1.0)]
)
@pytest.mark.parametrize("dataset", ['cifar100_icarl', 'cifar100'])
@pytest.mark.parametrize("seed", [1, 2])
def test_wlwf_0_10x10(params, dataset, seed):
    l, a, b, lc = params
    args = f"{TEST_ARGS} --num_tasks 10 --dataset {dataset} --exp_name l{l}_a{a}_b{b}_lc{lc}_seed{seed} --lamb {l} --lambda_cos_base {lc} --alpha {a} --beta {b} --seed {seed}"
    run_main_and_assert(args, result_dir='results_2_wlwf_cifar100_0_10x10')


@pytest.mark.parametrize(
    "params",
#     [(0.0, 20.0, 0.5, 0.0), (0.0, 20.0, 0.3, 0.0), (0.0, 0.0, 0.0, 5.0), (0.0, 1.0, 0.0, 5.0), (0.0, 1.0, 1.0, 1.0)]
    [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.05), (0.0, 00.0, 0.0, 0.1)]
)
@pytest.mark.parametrize("dataset", ['cifar100_icarl', 'cifar100'])
@pytest.mark.parametrize("seed", [1, 2])
def test_wlwf_with_cos_50_10x5(params, dataset, seed):
    l, a, b, lc = params
    args = f"{TEST_ARGS} --num_tasks 11 --nc_first_task 50 --dataset {dataset} --exp_name l{l}_a{a}_b{b}_lc{lc}_seed{seed} --lamb {l} --lambda_cos_base {lc} --alpha {a} --beta {b} --seed {seed}"
    run_main_and_assert(args, result_dir='results_2_wlwf_cifar100_50_10x5')