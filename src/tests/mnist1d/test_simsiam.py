import pytest
import torch

from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--datasets mnist1d" \
                       " --network ConvBase --num_tasks 3 --seed 1 --batch_size 100" \
                       " --nepochs 100 --lr_factor 10 --momentum 0.9 --lr 1e-2 --lr_min 1e-7" \
                       " --num_workers 0 --gpu 8"

# TODO: Data augumentations for Mnist1d 
# have to be solved in order to these
# tests to work.

# def test_simsiam():
#     args_line = FAST_LOCAL_TEST_ARGS
#     args_line += " --exp_name simsiam --approach simsiam"
#     run_main_and_assert(args_line)


# def test_finetuning_with_exemplars():
#     args_line = FAST_LOCAL_TEST_ARGS
#     args_line += " --approach finetune"
#     args_line += " --num_exemplars 200"
#     run_main_and_assert(args_line)



# @pytest.mark.parametrize(
# "n", list(range(0, 60, 10))
# )
# def test_finetuning_with_exemplars_per_class_and_herding(n):
#     args_line = FAST_LOCAL_TEST_ARGS
#     args_line += f" --exp_name ft_exe_{n} --approach finetune"
#     args_line += f" --num_exemplars_per_class {n}"
#     args_line += f" --exemplar_selection herding"
#     run_main_and_assert(args_line)
