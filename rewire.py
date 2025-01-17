import random
import time

import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import scipy
import torch
import math

from torch import optim
from torch.nn.parameter import Parameter
from tqdm import tqdm
import torch_geometric.utils as tgu

from utils import convert_edge2adj, normalize_adj

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

accuracy_MSE = torch.nn.MSELoss()

class MLP(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()
        self.lr1 = torch.nn.Linear(nfeat, nhid)
        self.lr2 = torch.nn.Linear(nhid, nclass)

        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lr1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lr2(x)
        return x

def train_mask(args, model, optimizer, data, mask):
    mx = data.x
    adj = tgu.to_scipy_sparse_matrix(data.edge_index)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = torch.FloatTensor(np.array(normalize_adj(adj + sp.eye(adj.shape[0])).todense()))

    loss_history = []
    val_acc_history = []
    t = time.time()
    model.train()

    print("Training rewire similarity matrix")
    t_total = time.time()
    for epoch in tqdm(range(args.rewire_epoch)):
        v1 = random.sample(mask.tolist(), args.rewire_batch_size)
        # v2 = random.sample(mask, args.rewire_batch_size)
        # todo: 2-order neighbour
        # compute similarity matrix
        optimizer.zero_grad()
        output = model(mx)
        output_batch = output[v1]
        features_batch = mx[v1]
        S = calculate_similarity_mx(output_batch)
        S_X = calculate_similarity_mx(features_batch)
        loss_train = F.l1_loss(S, S_X)
        acc_train = accuracy_MSE(S, S_X)
        loss_train.backward()
        optimizer.step()
        loss_val = F.l1_loss(S, S_X)
        acc_val = accuracy_MSE(S, S_X)
        loss_history.append(loss_val.item())
        val_acc_history.append(acc_val.item())

        if args.verbose:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))

    if args.verbose:
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    return loss_history, val_acc_history


def calculate_similarity_mx(mx):
    mx = mx / torch.norm(mx, dim=-1, keepdim=True)
    return torch.mm(mx, mx.T)

def rewire(args, data, sim_mx):
    node_num = sim_mx.size(0)
    adj = convert_edge2adj(data.edge_index).to(device)

    _, to_grow = torch.topk(sim_mx * torch.gt(sim_mx, args.growing_threshold) * (1 - adj), k=int(args.added_edges * node_num))
    to_grow = (torch.arange(node_num).repeat_interleave(int(args.added_edges * node_num)).to(device), to_grow.view(-1))
    adj[to_grow] = 1

    # TODO: exclude rows with all zeros

    to_prune = torch.nonzero(torch.lt(sim_mx, args.pruning_threshold)).T
    to_prune = (to_prune[0], to_prune[1])
    adj[to_prune] = 0

    adj = adj.to_dense()
    return adj

def train_and_rewire(args, data, mask):
    model = MLP(nfeat=args.num_features,
                        nhid=args.num_features,
                        nclass=args.num_features,
                        dropout=args.dropout)
    model.to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    loss, val_acc = train_mask(args, model, optimizer, data, mask)

    weight1 = model.lr1.weight.data
    weight2 = model.lr2.weight.data
    sim_mx = (calculate_similarity_mx(torch.mm(data.x, weight1)) +
              calculate_similarity_mx(torch.mm(data.x, weight2))) / 2

    adj = tgu.to_torch_coo_tensor(data.edge_index)
    adj = normalize_adj(sp.coo_matrix(rewire(args, data, sim_mx).cpu()) + sp.eye(adj.shape[0]))
    data.edge_index = tgu.from_scipy_sparse_matrix(adj)[0].to(device)

    return data