import json
import os
import subprocess

from openbox import space as sp
from openbox import Optimizer

def sample_condition(config):
    if config['pruning_threshold'] > config['growing_threshold']:
        return False
    return True

def get_configspace():
    space = sp.ConditionedSpace()
    added_edges = sp.Int("added_edges", 1, 5, default_value=1)
    growing_threshold = sp.Real("growing_threshold", 0.1, 0.5, default_value=0.3)
    pruning_threshold = sp.Real("pruning_threshold", -0.1, 0.3, default_value=0)
    rewire_batch_size = sp.Int("rewire_batch_size", 20, 100, default_value=50, q=5)
    mask_threshold = sp.Real("mask_threshold", 0.1, 0.5, default_value=0.3)
    space.add_variables([added_edges, growing_threshold, pruning_threshold, rewire_batch_size, mask_threshold])
    space.set_sample_condition(sample_condition)
    return space

def run(args, my_env):
    p = subprocess.Popen(args, env=my_env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    (output, err) = p.communicate()
    #This makes the wait possible
    p_status = p.wait()

def objective_function(config: sp.Configuration):
    params = config.get_dictionary()

    args = 'python active_graph.py --label_list 10 20 40 60 80 --lr 0.01 --epoch 200 --uniform_random --rewire --method kmeans --cluster_method kmeans --kmeans_num_layer 2 --self_loop_coeff 1.'.split()

    for key in params.keys():
        args.append('--' + key)
        args.append(str(params[key]))

    print(params)

    objectives = []

    for dataset in ['Cornell', 'Wisconsin', 'Texas', 'Chameleon', 'Squirrel', 'Actor']:
        run(args + ['--dataset ', dataset], os.environ.copy())
        parsed = json.load(open('optim.json', 'r'))
        objectives.append(1 - parsed['avg'][4][0])

    return dict(objectives=objectives)

opt = Optimizer(
    objective_function,
    get_configspace(),
    num_objectives=1,
    num_constraints=0,
    max_runs=100,
    surrogate_type='prf',
    time_limit_per_trial=1800,
    task_id='so_hpo',
)
history = opt.run()
print(history)
