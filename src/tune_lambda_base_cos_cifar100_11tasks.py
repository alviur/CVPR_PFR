import os
import argparse

import ray
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from main_incremental import main as main_incremental

EPOCHS = 200
TEST_ARGS = f"--network resnet32 --seed 1 --batch_size 128 --num_tasks 11 --nc_first_task 50 --dataset cifar100_icarl" \
            f" --nepochs {EPOCHS} --lr 0.01 --momentum 0.9 --weight_decay 0.0002 --lr_patience 10" \
            f" --lr_factor 3 --lr_min 0.0001" \
            f" --num_workers 0" \
            f" --approach lwf_ent_cos"

MAIN_METRIC_KEY = 'avg_inc_acc'


def run_main(args_line, result_dir):
    assert "--results_path" not in args_line
    TEST_RESULTS_PATH = os.getcwd() + '/../' + result_dir
    os.makedirs(TEST_RESULTS_PATH, exist_ok=True)
    args_line += " --results_path {}".format(TEST_RESULTS_PATH)
    print('ARGS:', args_line)
    return main_incremental(args_line.split(' '))


def evaluate_test(config):
    print('>> config: ', config)
    print('>> CUDA_VISIBLE_DEVICES: ', os.environ['CUDA_VISIBLE_DEVICES'])
    l = 0.0
    a = 0.0
    b = 0.0
    seed = 1
    lc = config['lambda_cos_base']
    args = f"{TEST_ARGS} --lamb {l} --lambda_cos_base {lc} --alpha {a} --beta {b} --seed {seed}"
    acc_taw, acc_tag, forg_taw, forg_tag, exp_dir = run_main(args, 'results_test')

    acc = acc_tag[-1].mean()
    tune.report(**{MAIN_METRIC_KEY: acc})


CONFIG = {'lambda_cos_base': tune.uniform(0.01, 30)}
OUTPUT_NAME = 'ray_cifar100_11tasks_lambda_cos_base'


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    ray.init(configure_logging=False)

    current_best_params = [{
        "lambda_cos_base": 5.0,
    }]

    algo = HyperOptSearch(points_to_evaluate=current_best_params)
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    scheduler = AsyncHyperBandScheduler()
    result = tune.run(
        evaluate_test,
        search_alg=algo,
        scheduler=scheduler,
        metric=MAIN_METRIC_KEY,
        mode="max",
        resources_per_trial={
            'cpu': 1,
            'gpu': 1
        },
        local_dir=f"~/ray_results/{OUTPUT_NAME}/",
        config=CONFIG,
        num_samples=8 if args.smoke_test else 20,
    )

    print("Best config: ", result.get_best_config(metric=MAIN_METRIC_KEY, mode="max"))

    # Get a dataframe for analyzing trial results.
    df = result.dataframe()
    df.to_pickle(f'~/ray_results/{OUTPUT_NAME}.pkl')


if __name__ == "__main__":
    main()