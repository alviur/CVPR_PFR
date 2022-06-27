# PYTHONPATH='.' python main_incremental.py --gpu 5 --exp_name simsiam --datasets cifar100_noTrans --network ss_resnet18 --num_tasks 10 --seed 1234 --batch_size 512 --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7 --stop_at_task 5 --eval_omni_head --eval_fresh_head --kd_method ft --approach simsiam --results_path /home/btwardowski/cvc-class-il/result_simsiam

PYTHONPATH='.' python main_incremental.py \
--gpu 7 --exp_name simsiam --datasets cifar100_noTrans \
--network ss_resnet18 --num_tasks 10 --seed 1234 --batch_size 512 \
--nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 \
--num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5 \
--lr_patience 20 --lr_min 5e-7 --eval_omni_head \
--eval_fresh_head --kd_method ft --approach simsiam \
--results_path /data/users/btwardow/cvc-class-il-results/simsiam_p2_analysis/