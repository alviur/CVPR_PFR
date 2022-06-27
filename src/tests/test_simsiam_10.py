from tests import run_main_and_assert

# Continual joint
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft" \
#                        " --approach simsiam --chansge_lr_scheduler --joint --task1_nepochs 1500 --wandblog   --loadTask1"

# Continual joint - Para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft" \
#                        " --approach simsiamTinyPara --change_lr_scheduler --joint --task1_nepochs 1500 --wandblog --head_classifier_lr 5e-2"


# Joint (no CL)
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 1 --seed 667 --batch_size 512 "\
#                         " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft" \
#                         " --approach simsiam --change_lr_scheduler --task1_nepochs 1500   --head_classifier_lr 5e-2"



# Joint (no CL) - Para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 1 --seed 667 --batch_size 512 "\
#                         " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft" \
#                         " --approach simsiamTinyPara --change_lr_scheduler --task1_nepochs 1500   --head_classifier_lr 5e-2 --wandblog"

# # FT
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets cifar100_noTrans" \
#                       " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                       " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                       " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                       " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft --wandblog" \
#                       " --approach simsiam --change_lr_scheduler --task1_nepochs 1500 --head_classifier_lr 5e-2 "

# FT para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiamP2 --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                         " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7  --eval_fresh_head --eval_omni_head  --kd_method ft" \
#                         " --approach simsiamTinyPara  --task1_nepochs 1500  " \
#                         "  --head_classifier_lr 5e-2 --wandblog"


# P2
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method p2" \
#                        " --approach simsiam --change_lr_scheduler --task1_nepochs 1500  --lambdap2 3 " \
#                        "  --head_classifier_lr 5e-2 --wandblog"

# supervised LwF
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ce" \
#                        " --approach simsiam --change_lr_scheduler --task1_nepochs 1500  --lambdap2 1 " \
#                        "  --head_classifier_lr 5e-2 --wandblog"

# supervised PFR
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method cePFR" \
#                        " --approach simsiam --change_lr_scheduler --task1_nepochs 1500  --lambdap2 1 " \
#                        "  --head_classifier_lr 5e-2 --wandblog"


# FD
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method L2" \
#                        " --approach simsiam --change_lr_scheduler --task1_nepochs 1500  --lamb 0.1 " \
#                        " --loadTask1 --head_classifier_lr 5e-2 --wandblog"


# FD - Para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method L2" \
#                        " --approach simsiamTinyPara --change_lr_scheduler --task1_nepochs 1500  --lamb 0.1 " \
#                        "  --head_classifier_lr 5e-2 --wandblog"

# EWC
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method EWC" \
#                        " --approach simsiam  --task1_nepochs 800  --lamb 70000 " \
#                        "  --head_classifier_lr 5e-2 --wandblog"

# EWC max CM
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_cm_min" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method EWC" \
#                        " --approach simsiam  --task1_nepochs 800  --lamb 70000 " \
#                        "   --head_classifier_lr 5e-2 --wandblog"

# P2_f
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method p2_f" \
#                        " --approach simsiam --change_lr_scheduler --task1_nepochs 1500  --lambdap2 7 " \
#                        " --loadTask1 --head_classifier_lr 5e-2  --wandblog"

# P2_f para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method p2_f" \
#                        " --approach simsiamTinyPara --change_lr_scheduler --task1_nepochs 1500  --lambdap2 6 " \
#                        "  --head_classifier_lr 5e-2  --wandblog"



# # P2_f max CM
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_cm_min" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method p2_f" \
#                        " --approach simsiam --change_lr_scheduler --task1_nepochs 1500  --lambdap2 6 " \
#                        "  --head_classifier_lr 5e-2  --wandblog"

# EWC-P2
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method EWC_p2" \
#                        " --approach simsiam --change_lr_scheduler --task1_nepochs 1500  --lamb 50000 " \
#                        " --loadTask1 --head_classifier_lr 5e-2 --wandblog"


# Lwf-v3
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method lwf_v3" \
#                        " --approach simsiam --change_lr_scheduler --task1_nepochs 1500  --lambdap2 1 " \
#                        " --loadTask1 --head_classifier_lr 5e-2  --wandblog"



# Eval CIFAR100
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                         " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft" \
#                         " --approach evalSimsiam  --task1_nepochs 1700 --head_classifier_lr 5e-2 "

# Eval SVHN
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets svhn_noTrans" \
#                        " --network resnet18 --num_tasks 1 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                        " --approach evalSimsiam  --task1_nepochs 1700 --head_classifier_lr 5e-2 "

# kNN CIFAR100
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                         " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 1 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                         " --approach evalSimsiamkNN  --task1_nepochs 1700 --head_classifier_lr 5e-2 "

# kNN SVHN
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets svhn_noTrans" \
#                         " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                         " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 1 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                         " --approach evalSimsiamkNN  --task1_nepochs 1700 --head_classifier_lr 5e-2 "


####################### Barlow Twins ###############################################

# Joint task Barlow Twins
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name barlowTwins --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft" \
#                         " --approach barlowTwinsFB --change_lr_scheduler --task1_nepochs 500 " \
#                         "  --head_classifier_lr 5e-2 --wandblog --projectorArc 2048_2048_2048 --loadTask1 --joint"

# single task Barlow Twins
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name barlowTwins --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 1 --seed 667 --batch_size 512 "\
#                         " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft" \
#                         " --approach barlowTwinsFB --change_lr_scheduler --task1_nepochs 1500 " \
#                         "  --head_classifier_lr 5e-2 --wandblog --projectorArc 2048_2048_2048"

# ft Barlow Twins
FAST_LOCAL_TEST_ARGS = "--gpu 0  --exp_name barlowTwins_FT_8192_singleGPU --datasets cifar100_noTrans" \
                        " --network resnet18 --num_tasks 50 --seed 667 --batch_size 256  "\
                        " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft  --joint" \
                        " --approach barlowTwinsFB  --task1_nepochs 500 -lambdap2 0.05" \
                        "  --head_classifier_lr 5e-2 --wandblog --projectorArc 2048_2048_2048 --dataset2 cifar100"\


# p2 Barlow Twins
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method p2_f" \
#                         " --approach barlowTwinsFB  --task1_nepochs 500  --lambdap2 60 " \
#                         "  --head_classifier_lr 5e-2 --wandblog --projectorArc 2048_2048_2048 --loadTask1"

# L2 Barlow Twins
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method L2" \
#                         " --approach barlowTwinsFB  --task1_nepochs 500  --lambdap2 10 " \
#                         "  --head_classifier_lr 5e-2 --wandblog --projectorArc 2048_2048_2048 --loadTask1 "

# EWC
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method EWC" \
#                         " --approach barlowTwinsFB  --task1_nepochs 500  --lamb 1 --lambdap2 1" \
#                         "  --head_classifier_lr 5e-2 --wandblog --projectorArc 8192_8192_8192 "


# Eval CIFAR
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                        " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                        " --approach evalBarlowTwins  --task1_nepochs 500 --head_classifier_lr 5e-1 --projectorArc 2048_2048_2048 "

# Eval SVHN
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets svhn_noTrans" \
#                        " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                        " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                        " --approach evalBarlowTwins  --task1_nepochs 500 --head_classifier_lr 5e-2 --projectorArc 2048_2048_2048 "


# kNN CIFAR100
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 1 --seed 667 --batch_size 512 "\
#                         " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 1 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                         " --approach evalBarlowTwinskNN  --task1_nepochs 1700 --head_classifier_lr 5e-2 --projectorArc 2048_2048_2048"

# joint Barlow Twins tiny
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets tiny_imagenet" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft" \
#                         " --approach barlowTwinsFB  --task1_nepochs 500  " \
#                         "  --head_classifier_lr 5e-1 --wandblog --joint --loadTask1"

# ft Barlow Twins tiny Para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name BarlowTwinsMultiGPU_FT --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 2 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                         " --approach BarlowTwinsTinyPara  --task1_nepochs 500 --lambdap2 0 " \
#                         "  --head_classifier_lr 5e-3 --wandblog --projectorArc 2048_2048_2048"

# ft Barlow Twins tiny Single
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 2 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                         " --approach BarlowTwinsTinyPara  --task1_nepochs 500 --lambdap2 0 " \
#                         "  --head_classifier_lr 5e-3 --wandblog --projectorArc 2048_2048_2048 "

# PFR Barlow Twins tiny Single
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 2 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                         " --approach barlowTwinsFB  --task1_nepochs 500 --lambdap2 50 " \
#                         "  --head_classifier_lr 5e-3 --wandblog --projectorArc 2048_2048_2048 --dataset2 cifar100 --loadTask1" \
#                         " --pathModelT1 /data3/users/agomezvi/cifar100_noTrans_BarlowTwinsTinyPara_simsiam-02162022_130504_891160/ "

# EWC Barlow Twins tiny Single
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 50 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 2 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                         " --approach barlowTwinsFB  --task1_nepochs 50 --lambdap2 1  --lamb 1" \
#                         "  --head_classifier_lr 5e-3 --wandblog --projectorArc 2048_2048_2048 --dataset2 cifar100"

# # P2_f Barlow Twins tiny Para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets tiny_imagenet" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 2 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                         " --approach BarlowTwinsTinyPara  --task1_nepochs 500 --lambdap2 1 " \
#                         "  --head_classifier_lr 5e-3 --wandblog  --projectorArc 2048_2048_2048"

# # P2_f Barlow Twins tiny Para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name BarlowTwins --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 2 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                         " --approach BarlowTwinsTinyPara  --task1_nepochs 500 --lambdap2 0 " \
#                         "  --head_classifier_lr 5e-3 --wandblog  --projectorArc 2048_2048_2048"

# # P2_f Barlow Twins tiny single
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name BarlowTwins --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 10 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head  --kd_method L2" \
#                         " --approach barlowTwinsFB  --task1_nepochs 500 --lambdap2 0.05 --loadTask1" \
#                         "  --head_classifier_lr 5e-2 --wandblog  --projectorArc 2048_2048_2048 " \
#                         "  --pathModelT1 /home/agomezvi/simSiamLB2/temp/  --dataset2 cifar100"
# # L2 Barlow Twins tony
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets tiny_imagenet" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method L2" \
#                         " --approach barlowTwinsFB  --task1_nepochs 500 --lambdap2 1 " \
#                         "  --head_classifier_lr 5e-1 --wandblog --loadTask1 "

# # EWC Barlow Twins tony
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets tiny_imagenet" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method EWC" \
#                         " --approach barlowTwinsFB  --task1_nepochs 500 --lambdap2 1  --lamb 1  " \
#                         "  --head_classifier_lr 5e-1 --wandblog --loadTask1 "

# Joint Barlow Twins tiny
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets tiny_imagenet" \
#                         " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                         " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft" \
#                         " --approach barlowTwins --change_lr_scheduler --joint --task1_nepochs 1500  " \
#                         "  --head_classifier_lr 5e-1 --wandblog"


# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets tiny_imagenet" \
#                         " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                         " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method p2_f" \
#                         " --approach barlowTwins --change_lr_scheduler --task1_nepochs 1500  --lambdap2 1500" \
#                         "  --head_classifier_lr 5e-1 --wandblog"

# ft Barlow Twins Para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name barlowTwins --datasets tiny_imagenet" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 16 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft" \
#                         " --approach BarlowTwinsTinyPara  --task1_nepochs 500 --lambdap2 30" \
#                         "  --head_classifier_lr 5e-1 --wandblog --projectorArc 2048_2048_2048"

# Eval Tiny
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets tiny_imagenet" \
#                        " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                        " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                        " --approach evalBarlowTwinsTiny  --task1_nepochs 500 --head_classifier_lr 5e-3 --projectorArc 2048_2048_2048 "


# Eval aircraft
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets aircraft" \
#                        " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                        " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                        " --approach evalBarlowTwinsTiny  --task1_nepochs 500 --head_classifier_lr 5e-1 --projectorArc 8192_8192_8192 "

# Eval cars
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets cars" \
#                        " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                        " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                        " --approach evalBarlowTwinsTiny  --task1_nepochs 500 --head_classifier_lr 5e-1 --projectorArc 8192_8192_8192 "

# kNN Tiny
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets tiny_imagenet" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 1 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                         " --approach evalBarlowTwinskNNTiny  --task1_nepochs 500 --head_classifier_lr 5e-2 --projectorArc 2048_2048_2048"

####################### Tiny Imagenet ##################################################

# Eval
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets imagenet_subset_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 1 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                        " --approach evalSimsiamTiny  --task1_nepochs 1700 --head_classifier_lr 5e-1 --wandblog"

# Continual joint
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_Continual_joint --datasets tiny_imagenet" \
#                         " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                         " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --head_classifier_lr 5e-1 --kd_method ft" \
#                         " --approach simsiamTinyPara --chansge_lr_scheduler --joint --task1_nepochs 1500 --wandblog   --loadTask1"

# Joint (no CL)
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_Joint --datasets tiny_imagenet" \
#                        " --network resnet18 --num_tasks 1 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft" \
#                        " --approach simsiamTinyPara --change_lr_scheduler --task1_nepochs 1700 --head_classifier_lr 5e-1"


# Single
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_single --datasets tiny_imagenet" \
#                        " --network resnet18 --num_tasks 1 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                        " --approach simsiamTinyPara  --task1_nepochs 1700 --head_classifier_lr 5e-1 --wandblog"

#FT
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_FT --datasets tiny_imagenet" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 5 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 1 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft" \
#                        " --approach simsiamTinyPara  --task1_nepochs 5 --head_classifier_lr 5e-1 --wandblog"

# P2
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiamP2 --datasets tiny_imagenet" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method p2" \
#                        " --approach simsiamTiny --change_lr_scheduler --task1_nepochs 1500  --lambdap2 1 " \
#                        " --loadTask1 --head_classifier_lr 5e-1 --wandblog"

# P2 para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiamP2 --datasets tiny_imagenet" \
#                         " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                         " --nepochs 1000 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7  --eval_fresh_head --eval_omni_head  --kd_method p2" \
#                         " --approach simsiamTinyPara  --task1_nepochs 1700  --lambdap2 1 " \
#                         "  --head_classifier_lr 5e-1 --wandblog"

# P2_f para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiamP2 --datasets tiny_imagenet" \
#                       " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                       " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                       " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                       " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --kd_method p2_f" \
#                       " --approach simsiamTinyPara --change_lr_scheduler --task1_nepochs 1500  --lambdap2 0.2" \
#                       " --loadTask1 --head_classifier_lr 5e-1 --wandblog"


# FD
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiamFD --datasets tiny_imagenet" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method L2" \
#                        " --approach simsiamTiny --change_lr_scheduler --task1_nepochs 1500  --lamb 0.001 " \
#                        " --loadTask1 --head_classifier_lr 5e-1 --wandblog"


# EWC
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam --datasets tiny_imagenet" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method EWC" \
#                        " --approach simsiamTiny  --task1_nepochs 1500  --lamb 5000 " \
#                        "  --loadTask1 --head_classifier_lr 5e-2 --wandblog"

# EWC para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiamP2 --datasets tiny_imagenet" \
#                         " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                         " --nepochs 1 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 1 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7  --kd_method EWC" \
#                         " --approach simsiamTinyPara --change_lr_scheduler --task1_nepochs 1  --lambdap2 10000 " \
#                         "  --head_classifier_lr 5e-1 --wandblog"

# #  evaluation
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name Cars_evaluation --datasets tiny_imagenet" \
#                         " --network resnet18 --num_tasks 4 --seed 667 --batch_size 256 "\
#                         " --nepochs 1 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7  --eval_omni_head  --kd_method ft --wandblog" \
#                         " --approach evalBarlowTwinsTiny --change_lr_scheduler --head_classifier_lr 5e-3 --projectorArc 2048_2048_2048"

# # Cars evaluation
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name Cars_evaluation --datasets cars" \
#                         " --network resnet18 --nuTwins_tasks 1 --seed 667 --batch_size 512 "\
#                         " --nepochs 1 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft --wandblog" \
#                         " --approach evalSimsiamTiny --change_lr_scheduler --head_classifier_lr 5e-1"

# Birds evaluation
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name Birds_evaluation --datasets birds" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 1 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft" \
#                        " --approach evalSimsiamTiny --change_lr_scheduler --task1_nepochs 1 --head_classifier_lr 5e-1"



# # # aircraft evaluation
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name aircraft_evaluation --datasets aircraft" \
#                        " --network resnet18 --num_tasks 1 --seed 667 --batch_size 512 "\
#                        " --nepochs 1 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft  --wandblog" \
#                        " --approach evalSimsiamTiny --change_lr_scheduler --task1_nepochs 1 --head_classifier_lr 5e-1"


############################################# Imagenet_subset ##################################################

#FT
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_FT_ImagenetSubset --datasets imagenet_subset_noTrans" \
#                        " --network resnet18 --num_tasks 5 --seed 667 --batch_size 256 "\
#                        " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft" \
#                        " --approach BarlowTwinsTinyPara  --task1_nepochs 500 --head_classifier_lr 5e-2 --wandblog --projectorArc 2048_2048_2048"

#Single
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_FT_ImagenetSubset --datasets imagenet_subset_noTrans" \
#                        " --network resnet18 --num_tasks 1 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft" \
#                        " --approach simsiamTinyPara  --task1_nepochs 1500 --head_classifier_lr 5e-1 --wandblog"

# P2_f para
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiamP2 --datasets imagenet_subset_noTrans" \
#                       " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                       " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                       " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                       " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --kd_method p2_f" \
#                       " --approach simsiamTinyPara  --task1_nepochs 1500  --lambdap2 7" \
#                       "  --head_classifier_lr 5e-2 --wandblog --loadTask1"

############################################# Birds ##################################################

#FT
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_FT_ImagenetSubset --datasets birds" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 1000 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft" \
#                        " --approach simsiamTinyPara  --task1_nepochs 2500 --head_classifier_lr 5e-2 --wandblog"

#P2_f
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiamP2 --datasets birds" \
#                       " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                       " --nepochs 1000 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                       " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                       " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --kd_method p2_f" \
#                       " --approach simsiamTinyPara  --task1_nepochs 2000  --lambdap2 1" \
#                       "  --head_classifier_lr 5e-2 --wandblog "

# Birds evaluation
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name Birds_evaluation --datasets birds" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 1 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head  --kd_method ft --wandblog"\
#                        " --approach evalSimsiamTiny --change_lr_scheduler --task1_nepochs 1 --head_classifier_lr 5e-1"



################################################### No Task boundaries #########################################
# joint
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_ft_new --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 100 --seed 667 --batch_size 512 "\
#                        " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --kd_method ft" \
#                        " --approach simsiam_online --change_lr_scheduler --task1_nepochs 500  --lambdap2 1 " \
#                        "  --head_classifier_lr 5e-2 --no_task_boundary_beta 4 --updateBackbone 1 --joint"

# FT
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_ft_new --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft" \
#                        " --approach simsiam_online --change_lr_scheduler --task1_nepochs 500  --lambdap2 1 " \
#                        "  --head_classifier_lr 5e-2 --no_task_boundary_beta 4 --updateBackbone 1"

# P2_f
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_ft_new --datasets cifar100_noTrans" \
#                       " --network resnet18 --num_tasks 5 --seed 667 --batch_size 512 "\
#                       " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                       " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                       " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method p2_f" \
#                       " --approach simsiam_online --change_lr_scheduler --task1_nepochs 500  --lambdap2 0.5 " \
#                       "  --head_classifier_lr 5e-2 --updateBackbone 1 --no_task_boundary_beta 4"

# P2
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_ft_new --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method p2" \
#                        " --approach simsiam_online --change_lr_scheduler --task1_nepochs 1500  --lambdap2 3 " \
#                        "  --head_classifier_lr 5e-2 --updateBackbone 1 --no_task_boundary_beta 4"

# # L2
#FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_ft_new --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method L2" \
#                        " --approach simsiam_online --change_lr_scheduler --task1_nepochs 1500  --lamb 0.001 " \
#                        "  --head_classifier_lr 5e-2 --updateBackbone 1 --no_task_boundary_beta 4"


# Lwf v3
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_ft_new --datasets cifar100_noTrans" \
#                         " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                         " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                         " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                         " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method lwf_v3" \
#                         " --approach simsiam_online --change_lr_scheduler --task1_nepochs 1500  --lambdap2 3 " \
#                         "  --head_classifier_lr 5e-2 --updateBackbone 1 --no_task_boundary_beta 4"

# FT Barlow Twins
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_ft_new --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft" \
#                        " --approach simsiam_onlineBarlowTwins --change_lr_scheduler --task1_nepochs 500  --lambdap2 1 " \
#                        "  --head_classifier_lr 5e-2 --no_task_boundary_beta 4 --updateBackbone 1"

# P2
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_ft_new --datasets cifar100_noTrans" \
#                        " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                        " --nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                        " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                        " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method p2" \
#                        " --approach simsiam_onlineBarlowTwins --change_lr_scheduler --task1_nepochs 500  --lambdap2 1 " \
#                        "  --head_classifier_lr 5e-2 --no_task_boundary_beta 4 --updateBackbone 1"


#########################################Datasets

# FT Tiny
# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets imagenet_subset_noTrans" \
#                       " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                       " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                       " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                       " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft --wandblog" \
#                       " --approach simsiam --change_lr_scheduler --task1_nepochs 1500 --head_classifier_lr 5e-2 "

# FAST_LOCAL_TEST_ARGS = "--gpu 0 --exp_name simsiam_p2 --datasets imagenet_subset_noTrans" \
#                       " --network resnet18 --num_tasks 10 --seed 667 --batch_size 512 "\
#                       " --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 "\
#                       " --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5" \
#                       " --lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --kd_method ft --wandblog" \
#                       " --approach simsiam  --task1_nepochs 1500 --head_classifier_lr 5e-2 "



def test_simsiam():
    args_line = FAST_LOCAL_TEST_ARGS
    #run_main_and_assert(args_line, 0, 0, result_dir='/data/users/agomezvi')#112
    run_main_and_assert(args_line, 0, 0, result_dir='/data3/users/agomezvi')  # 103
