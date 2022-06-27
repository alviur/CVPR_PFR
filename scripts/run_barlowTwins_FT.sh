
echo "GPU: $1"
echo "FT run: $2"
echo "Port: $3"

RESULT_DIR="/data2/users/btwardow/4_tasks_BarlowTwins/FT-$2"
mkdir -p $RESULT_DIR

CUDA_VISIBLE_DEVICES=$1 PYTHONPATH='.' python main_incremental.py \
--gpu 0 --exp_name barlowTwins_FT-$2 --datasets cifar100_noTrans \
--network resnet18 --num_tasks 4 --seed 667 --batch_size 256 \
--nepochs 500 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 \
--num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5 \
--lr_patience 20 --lr_min 5e-7 --eval_omni_head --eval_fresh_head --loadTask1 \
--kd_method ft --approach barlowTwinsFB  --task1_nepochs 500 --dataset2 cifar100 \
--results_path $RESULT_DIR  --head_classifier_lr 5e-3 --wandblog --projectorArc 2048_2048_2048 --port $3 \
--pathModelT1  /data2/users/btwardow/feb_barlowtiwns/FT-cifar100-4-tasks/cifar100_noTrans_barlowTwinsFB_barlowTwins-01112022_225056_903837/
