import pytest
from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--datasets mnist1d" \
                       " --network ConvBase --num_tasks 3 --seed 1 --batch_size 100" \
                       " --nepochs 100 --lr_factor 10 --momentum 0.9 --lr 1e-2 --lr_min 1e-7" \
                       " --num_workers 0" \
                       " --approach lwf_ft"


def test_lwf_without_exemplars():
    run_main_and_assert("--exp_name lwfft " + FAST_LOCAL_TEST_ARGS)


@pytest.mark.parametrize("n", list(range(0, 60, 10)))
def test_finetuning_with_exemplars_per_class_and_herding(n):
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += f" --exp_name lwfft_exe_{n}"
    args_line += f" --num_exemplars_per_class {n}"
    args_line += f" --exemplar_selection herding"
    run_main_and_assert(args_line)


@pytest.mark.parametrize("n", list(range(0, 60, 10)))
def test_no_finetuning_with_exemplars(n):
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += f" --exp_name lwfft_noft_exe_{n}"
    args_line += f" --num_exemplars_per_class {n}"
    args_line += f" --exemplar_selection herding"
    args_line += f" --nepochs_finetuning 0"
    run_main_and_assert(args_line)


@pytest.mark.parametrize("n", list(range(0, 60, 10)))
def test_only_head(n):
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += f" --exp_name ft_only_heads_exe_{n}"
    args_line += f" --num_exemplars_per_class {n}"
    args_line += f" --exemplar_selection herding --lr_finetuning_factor 0.03"
    args_line += f" --only_head_finetuning"
    run_main_and_assert(args_line)