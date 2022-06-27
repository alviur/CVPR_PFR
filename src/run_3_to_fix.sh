# PYTHONPATH='.' python main_incremental.py --gpu 5 --exp_name simsiam --datasets cifar100_noTrans --network ss_resnet18 --num_tasks 10 --seed 1234 --batch_size 512 --nepochs 800 --optim_name sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 --num_workers 4 --classifier_nepochs 200 --hidden_mlp 512 --jitter_strength 0.5 --lr_patience 20 --lr_min 5e-7 --stop_at_task 5 --eval_omni_head --eval_fresh_head --kd_method ft --approach simsiam --results_path /home/btwardowski/cvc-class-il/result_simsiam

PYTHONPATH='.' python main_incremental.py \
       --exp_name cifar100_10tasks \
       --datasets cifar100 --num_tasks 10 --network resnet18_and_proj \
       --nepochs 200 --batch_size 128 \
       --gridsearch_tasks 10 --gridsearch_config gridsearch_config \
       --gridsearch_acc_drop_thr 0.2 --gridsearch_hparam_decay 0.5 \
       --approach finetune --gpu $1 \
       --results_path /home/btwardowski/cvc-class-il/result_simsiam_p2/
       