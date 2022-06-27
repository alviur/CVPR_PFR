from tests import run_main_and_assert

# 10 TASK, STOP AFTER 5
FAST_LOCAL_TEST_ARGS = "--gpu 8 --exp_name simsiam --datasets cifar100" \
                       " --network ss_resnet18 --num_tasks 10 --seed 1234 --batch_size 512" \
                       " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001" \
                       " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
                       " --lr_patience 20 --lr_min 5e-7 --stop_at_task 5 --head_classifier_hidden_mlp 0 --eval_omni_head --eval_fresh_head --init_after_each_task"

def test_simsiam():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach simsiam"
    run_main_and_assert(args_line, 0, 0, result_dir='result_simsiam')
