from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp_name local_test --datasets cifar100" \
                       " --network resnet32 --num_tasks 3 --seed 1 --batch_size 32" \
                       " --nepochs 3" \
                       " --num_workers 0" \
                       " --approach dmc" \
                       " --aux_dataset cifar100"


def test_dmc():
    run_main_and_assert(FAST_LOCAL_TEST_ARGS)


