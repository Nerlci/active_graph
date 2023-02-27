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


models = ['MatrixGCN', 'SGC', 'GCN', 'H2GCN']
datasets = ['Actor', 'Cornell', 'Wisconsin', 'Texas', 'Chameleon', 'Squirrel', 'Cora', 'Citeseer', 'PubMed']
label = {'Cora': 140, 'Citeseer': 120, 'PubMed': 60, 'Actor': 120,
         'Cornell': 80, 'Wisconsin': 80, 'Texas': 80, 'Chameleon': 120, 'Squirrel': 120}

for model in models:
    for dataset in datasets:
        model_args = ['--model', model]
        label_list = ['10'] + [str(i) for i in range(20, label[dataset]+1, 20)]
        dataset_args = ['--dataset', dataset, '--label_list'] + label_list
        print(dataset_args)
        # for dropout in ['0.5', '0.']:
        for dropout in ['0.5']:
            extra_args = model_args + dataset_args + ['--dropout', dropout]

            # random
            args = ori_args + ['--method', 'random'] + extra_args
            run(args, my_env)

            args = ori_args + ['--method', 'random'] + extra_args + ['--rewire']
            run(args, my_env)

            # degree
            args = ori_args + ['--method', 'degree'] + extra_args
            run(args, my_env)

            args = ori_args + ['--method', 'degree'] + extra_args + ['--rewire']
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

                        args = ori_args + kmeans_args + extra_args + ['--rewire']
                        run(args, my_env)

            # age
            extra_args = model_args + dataset_args + ['--dropout', dropout]

            args = ori_args + ['--method', 'age', '--uncertain_score', 'entropy'] + extra_args
            run(args, my_env)

            args = ori_args + ['--method', 'age', '--uncertain_score', 'entropy'] + extra_args + ['--rewire']
            run(args, my_env)

            # anrmab
            extra_args = model_args + dataset_args + ['--dropout', dropout]

            args = ori_args + ['--method', 'anrmab', '--uncertain_score', 'entropy'] + extra_args
            run(args, my_env)

            args = ori_args + ['--method', 'anrmab', '--uncertain_score', 'entropy'] + extra_args + ['--rewire']
            run(args, my_env)