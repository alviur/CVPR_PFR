from tests import run_main_and_assert


#ft Barlow Twins Para
FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name barlowTwins --datasets tiny_imagenet" \
                        " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
                        " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
                        " --num_workers 16 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
                        " --approach BarlowTwinsTinyPara  --task1_nepochs 1000 " \
                        "  --head_classifier_lr 5e-3 --wandblog --projectorArc 2048_2048_2048"

#Joint Barlow Twins Para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name barlowTwins --datasets tiny_imagenet" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 16 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                         " --approach BarlowTwinsTinyPara  --task1_nepochs 1000  --lambdap2 0.1" \
#                         "  --head_classifier_lr 5e-3 --wandblog --projectorArc 2048_2048_2048" \
#                         " --pathModelT1 /home/agomezvi/simSiamLB2/4_tasks/"

#PFR Barlow Twins Para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name barlowTwins --datasets tiny_imagenet" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 16 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                         " --approach BarlowTwinsTinyPara  --task1_nepochs 500  --lambdap2 0.1" \
#                         "  --head_classifier_lr 5e-3 --wandblog --projectorArc 2048_2048_2048" \
#                         " --pathModelT1 /home/agomezvi/simSiamLB2/4_tasks/"

#L2 Barlow Twins Para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name barlowTwins --datasets tiny_imagenet" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 16 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method p2_f" \
#                         " --approach BarlowTwinsTinyPara  --task1_nepochs 500 --loadTask1 --lambdap2 0.1" \
#                         "  --head_classifier_lr 5e-3 --wandblog --projectorArc 2048_2048_2048" \
#                         " --pathModelT1 /home/agomezvi/simSiamLB2/4_tasks/FT_0.06/"

#EWC Barlow Twins Para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name barlowTwins --datasets tiny_imagenet" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 1 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 16 --classifier_nepochs 1 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method EWC" \
#                         " --approach BarlowTwinsTinyPara  --task1_nepochs 500 --loadTask1 --lambdap2 0.1" \
#                         "  --head_classifier_lr 5e-3 --wandblog --projectorArc 2048_2048_2048" \
#                         " --pathModelT1 /home/agomezvi/simSiamLB2/4_tasks/FT_0.06/"

#ft Barlow Twins Para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name barlowTwins --datasets tiny_imagenet" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 16 --classifier_nepochs 1 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_fresh_head  " \
#                         " --approach recencyBiasTiny  --task1_nepochs 500 --loadTask1 --lambdap2 20" \
#                         "  --head_classifier_lr 5e-3 --wandblog --projectorArc 2048_2048_2048"



#
def test_simsiam():
    args_line = FAST_LOCAL_TEST_ARGS
    run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi')#112
    #run_main_and_assert(args_line, 0, 0, result_dir='/data3/users/agomezvi')  # 103
