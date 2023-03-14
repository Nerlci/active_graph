import subprocess, os
import tqdm
my_env = os.environ.copy()
# my_env['CUDA_VISIBLE_DEVICES'] = '2'

#
ori_args = 'python active_graph.py --lr 0.01 --epoch 200 --rand_rounds 10 --uniform_random'.split()

def run(args, my_env):
    print(args)
    p = subprocess.Popen(args, env=my_env)
    (output, err) = p.communicate()
    #This makes the wait possible
    p_status = p.wait()


models = ['GCN']
datasets = ['Actor', 'Cornell', 'Wisconsin', 'Texas', 'Chameleon', 'Squirrel']
label = {'Actor': 120, 'Cornell': 80, 'Wisconsin': 80,
         'Texas': 80, 'Chameleon': 120, 'Squirrel': 120}
seeds = ['123', '998244353', '19260817']
rewire_parameters = {'Actor': ['--growing_threshold', '0.49', '--pruning_threshold', '0.070', '--mask_threshold', '0.2', '--added_edges', '0.0054'],
         'Cornell': [],
         'Wisconsin': ['--growing_threshold', '0.36', '--pruning_threshold', '0.19', '--mask_threshold', '0.14', '--added_edges', '0.020'],
         'Texas': ['--growing_threshold', '0.36', '--pruning_threshold', '0.19', '--mask_threshold', '0.14', '--added_edges', '0.020'],
         'Chameleon': ['--growing_threshold', '0.40', '--pruning_threshold', '-0.09', '--mask_threshold', '0.43', '--added_edges', '0.009'],
         'Squirrel': ['--growing_threshold', '0.45', '--pruning_threshold', '-0.027', '--mask_threshold', '0.155', '--added_edges', '0.0036']}

for seed in seeds:
    for model in models:
        for dataset in datasets:
            model_args = ['--model', model]
            label_list = ['10'] + [str(i) for i in range(20, label[dataset] + 1, 20)]
            dataset_args = ['--dataset', dataset, '--label_list'] + label_list
            print(dataset_args)
            # for dropout in ['0.5', '0.']:
            for dropout in ['0.5']:
                extra_args = model_args + dataset_args + ['--dropout', dropout] + ['--seed', seed]

                # random
                args = ori_args + ['--method', 'random'] + extra_args
                run(args, my_env)

                args = ori_args + ['--method', 'random'] + extra_args + ['--rewire'] + rewire_parameters[dataset]
                run(args, my_env)

                # degree
                args = ori_args + ['--method', 'degree'] + extra_args
                run(args, my_env)

                args = ori_args + ['--method', 'degree'] + extra_args + ['--rewire'] + rewire_parameters[dataset]
                run(args, my_env)

                # kmeans
                for cluster_method in ['kmeans']:
                # for cluster_method in ['kmeans']:
                    # for kmeans_num_layer in ['0', '2', '5']:
                    for kmeans_num_layer in ['2']:
                        # for self_loop_coeff in ['0.', '1.']:
                        for self_loop_coeff in ['1.']:
                            kmeans_args = ['--method', 'kmeans', '--cluster_method',
                            cluster_method, '--kmeans_num_layer', kmeans_num_layer,
                            '--self_loop_coeff', self_loop_coeff]
                            # run
                            args = ori_args + kmeans_args + extra_args
                            run(args, my_env)

                            args = ori_args + kmeans_args + extra_args + ['--rewire'] + rewire_parameters[dataset]
                            run(args, my_env)

                # age
                args = ori_args + ['--method', 'age', '--uncertain_score', 'entropy'] + extra_args
                run(args, my_env)

                args = ori_args + ['--method', 'age', '--uncertain_score', 'entropy'] + extra_args + ['--rewire'] + rewire_parameters[dataset]
                run(args, my_env)

                # anrmab
                args = ori_args + ['--method', 'anrmab', '--uncertain_score', 'entropy'] + extra_args
                run(args, my_env)

                args = ori_args + ['--method', 'anrmab', '--uncertain_score', 'entropy'] + extra_args + ['--rewire'] + rewire_parameters[dataset]
                run(args, my_env)