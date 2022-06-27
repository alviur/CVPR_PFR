from tests import run_main_and_assert
import pytest

EPOCHS = 200

              #   PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name exp2b_random_${SEED} \
              #           --datasets imagenet_subset --num_tasks 10 --network resnet18 --seed $SEED \
              #           --nepochs 200 --batch_size 128 --results_path $RESULTS_DIR \
              #           --gridsearch_tasks 3 --gridsearch_config gridsearch_config \
              #           --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
              #           --approach $APPROACH --gpu $2         
              
              # PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp_name subset_imagenet_${NETWORK} \
              #  --datasets imagenet_subset --num_tasks 10 --network $NETWORK \
              #  --nepochs 100 --batch_size 128 --results_path $RESULTS_DIR \
              #  --momentum 0.9 --weight_decay 0.0002 --lr 0.1 --lr_patience 10 \
              #  --gridsearch_tasks 10 --gridsearch_config gridsearch_config_subset \
              #  --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
              #  --approach $1 --gpu $2 \
              #  --num_exemplars_per_class 20 --exemplar_selection herding

TEST_ARGS = f"--datasets imagenet_subset --network resnet18 --seed 1 --batch_size 128 --nepochs {EPOCHS}" \
            f" --approach lwf_ent --num_tasks 10 --seed 1" \
            f" --momentum 0.9 --weight_decay 0.0002 --lr 0.1 --lr_patience 10"


@pytest.mark.parametrize(
    "params",
    [(0.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 10.0, 0.3), (10.0, 10.0, 1.0)]
)
def test_wlwf_0_10x10(params):
    l, a, b = params
    args = f"{TEST_ARGS} --exp_name l{l}_a{a}_b{b} --lamb {l} --alpha {a} --beta {b}"
    run_main_and_assert(args, result_dir='results_wlwf_imagenet_subset')