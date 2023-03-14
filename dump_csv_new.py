import numpy as np
import json
import os
import glob
import csv

# dataset = 'Cora'
# # dataset = 'Citeseer'
# model = 'MatrixGCN'
# # model = 'SGC'

# with open('computers.csv', 'w') as csvfile:
with open('main.csv', 'w') as csvfile:
    header = ['model', 'dataset', 'seed', 'method', 'cluster_method',
    'kmeans_num_layer', 'uncertain_score', 
    'self_loop_coeff', 'dropout', 'time',
    'rewire']
    writer = csv.writer(csvfile)
    writer.writerow(header + [10, 20, 40, 60, 80] + [10, 20, 40, 60, 80] + ['metric_names', 'filename'])
    # writer.writerow(header + ['10', 'std'] + [str(i) if i / 10 % 2 == 0 else 'std' for i in range(20, 131, 10)] + ['10', 'std'] + [str(i) if i / 10 % 2 == 0 else 'std' for i in range(20, 131, 10)] + ['metric_names', 'filename'])
    # for model in ['MatrixGCN', 'SGC', 'GCN']:
    for model in ['MatrixGCN', 'SGC', 'GCN', 'H2GCN']:
        for dataset in ['Cora', 'Citeseer', 'PubMed', 'CoraFull', 'Photo', 'Actor',
                        'Cornell', 'Wisconsin', 'Texas', 'Chameleon', 'Squirrel']:
            filenames = glob.glob('./{}/{}/*/*'.format(model, dataset))
            for filename in filenames:
                parsed = json.load(open(filename, 'r'))

                args = [model, dataset, parsed['args']['seed'], parsed['args']['method'],
                parsed['args']['cluster_method'], 
                parsed['args']['kmeans_num_layer'], parsed['args']['uncertain_score'], 
                parsed['args']['self_loop_coeff'], parsed['args']['dropout'], parsed['time'],
                parsed['args']['rewire']]
                avg = parsed['avg']
                std = parsed['std']
                # avgstd = []
                # for r in range(2):
                #     for i, v in enumerate(avg[:, r]):
                #         avgstd.append(v)
                #         avgstd.append(std[i, r])
                #     while len(avgstd) != 14 * (r + 1):
                #         avgstd.append('')
                row = args + avg +std + [parsed['metric_names']] + [filename]
                writer.writerow(row)