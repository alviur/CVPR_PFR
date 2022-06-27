RESULT_DIR=/home/agomezvi/simSiamLB2/result_simsiam/
mkdir -p $RESULT_DIR

PYTHONPATH='.' python main_incremental.py \
--gpu 0 --exp_name simsiam_ft --datasets cifar100_noTrans \
--network resnet18 --num_tasks 10 --seed 667 --batch_size 512 \
--nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 \
--num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5 \
--lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head \
--kd_method ft --approach simsiamTinyPara  --task1_nepochs 1700 \
--results_path $RESULT_DIR  --head_classifier_lr 5e-1