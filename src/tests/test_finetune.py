import pytest
import torch

from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp_name local_test --datasets mnist" \
                       " --network LeNet --num_tasks 3 --seed 1 --batch_size 32" \
                       " --nepochs 3 --lr_factor 10 --momentum 0.9 --lr_min 1e-7" \
                       " --num_workers 0"


def test_finetuning():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetune"
    run_main_and_assert(args_line)


def test_finetuning_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetune"
    args_line += " --num_exemplars 200"
    run_main_and_assert(args_line)


@pytest.mark.xfail
def test_finetuning_with_exemplars_per_class_and_herding():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetune"
    args_line += " --num_exemplars_per_class 10"
    args_line += " --exemplar_selection herding"
    run_main_and_assert(args_line)


def test_finetuning_with_exemplars_per_class_and_entropy():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetune"
    args_line += " --num_exemplars_per_class 10"
    args_line += " --exemplar_selection entropy"
    run_main_and_assert(args_line)


def test_finetuning_with_exemplars_per_class_and_distance():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetune"
    args_line += " --num_exemplars_per_class 10"
    args_line += " --exemplar_selection distance"
    run_main_and_assert(args_line)


def test_wrong_args():
    with pytest.raises(SystemExit):  # error of providing both args
        args_line = FAST_LOCAL_TEST_ARGS
        args_line += " --approach finetune"
        args_line += " --num_exemplars_per_class 10"
        args_line += " --num_exemplars 200"
        run_main_and_assert(args_line)


def test_finetuning_with_eval_on_train():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetune"
    args_line += " --num_exemplars_per_class 10"
    args_line += " --exemplar_selection distance"
    args_line += " --eval_on_train"
    run_main_and_assert(args_line)


def test_finetuning_with_no_cudnn_deterministic():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetune"
    args_line += " --num_exemplars_per_class 10"
    args_line += " --exemplar_selection distance"

    run_main_and_assert(args_line)
    assert torch.backends.cudnn.deterministic

    args_line += " --no_cudnn_deterministic"
    run_main_and_assert(args_line)
    assert not torch.backends.cudnn.deterministic