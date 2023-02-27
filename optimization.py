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
    added_edges = sp.Int("added_edges", 1, 3, default_value=1)
    growing_threshold = sp.Real("growing_threshold", 0.1, 0.5, default_value=0.2)
    pruning_threshold = sp.Real("pruning_threshold", -0.1, 0.3, default_value=0.1)
    rewire_batch_size = sp.Int("rewire_batch_size", 20, 100, default_value=50, q=5)
    rewire_epoch = sp.Int("rewire_epoch", 100, 1000, default_value=200, q=100)
    mask_threshold = sp.Real("mask_threshold", 0.1, 0.5, default_value=0.2)
    space.add_variables([added_edges, growing_threshold, pruning_threshold, rewire_batch_size, rewire_epoch,
                      mask_threshold])
    space.set_sample_condition(sample_condition)
    return space

def run(args, my_env):
    p = subprocess.Popen(args, env=my_env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    (output, err) = p.communicate()
    #This makes the wait possible
    p_status = p.wait()

def objective_function(config: sp.Configuration):
    params = config.get_dictionary()

    args = 'python active_graph.py --methon anrmab --label_list 10 20 40 60 80 --lr 0.01 --epoch 200 --uniform_random --dataset Cornell --rewire'.split()

    for key in params.keys():
        args.append('--' + key)
        args.append(str(params[key]))

    print(params)

    run(args, os.environ.copy())
    parsed = json.load(open('optim.json', 'r'))

    return dict(objectives=[parsed['avg'][4][0]])

opt = Optimizer(
    objective_function,
    get_configspace(),
    num_objectives=1,
    num_constraints=0,
    max_runs=100,
    surrogate_type='prf',
    time_limit_per_trial=180,
    task_id='so_hpo',
)
history = opt.run()
print(history)