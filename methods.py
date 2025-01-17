import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.utils as tgu
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances

from sklearn.cluster import KMeans
# from cuml import KMeans
from sklearn.metrics import pairwise_distances
from utils import convert_edge2adj, normalize, normalize_adj, convert_edge2adj_sparse, normalize_row
from utils import kcenter_choose, kmeans_choose, kmedoids_choose, combine_new_old

import time

# Factory class:
class ActiveFactory:
    def __init__(self, args, model, data, prev_index, train_mask):
        #
        self.args = args
        self.model = model
        self.data = data
        self.prev_index = prev_index
        self.train_mask = train_mask

    def get_learner(self):
        if self.args.method == 'random':
            self.learner = RandomLearner
        elif self.args.method == 'kmeans':
            self.learner = KmeansLearner
        elif self.args.method == 'degree':
            self.learner = DegreeLearner
        elif self.args.method == 'nonoverlapdegree':
            self.learner = NonOverlapDegreeLearner
        elif self.args.method == 'coreset':
            self.learner = CoresetLearner
        elif self.args.method == 'uncertain':
            self.learner = UncertaintyLearner
        elif self.args.method == 'anrmab':
            self.learner = AnrmabLearner
        elif self.args.method == 'age':
            self.learner = AgeLearner
        elif self.args.method == 'grain':
            self.learner = GrainLearner
        elif self.args.method == 'combined':
            self.learner = CombinedLearner
        return self.learner(self.args, self.model, self.data, self.prev_index, self.train_mask)

# Base class
class ActiveLearner:
    def __init__(self, args, model, data, prev_index, train_mask):
        self.model = model
        self.data = data
        self.n = data.num_nodes
        self.args = args
        self.prev_index = prev_index
        self.train_mask = train_mask

        if prev_index is None:
            self.prev_index_list = []
        else:
            self.prev_index_list = np.where(self.prev_index.cpu().numpy())[0]


    def choose(self, num_points):
        raise NotImplementedError

    def pretrain_choose(self, num_points):
        raise NotImplementedError

class CombinedLearner(ActiveLearner):
    def __init__(self, args, model, data, prev_index, train_mask):
        super(CombinedLearner, self).__init__(args, model, data, prev_index, train_mask)

    def pretrain_choose(self, num_points):
        # first choose half nodes from uncertain
        prev_index_len = len(self.prev_index_list)
        if prev_index_len == 0:
            return KmeansLearner(self.args, self.model, self.data, self.prev_index).pretrain_choose(num_points)

        ul = UncertaintyLearner(self.args, self.model, self.data, self.prev_index)
        new_len = num_points - prev_index_len
        ul_mask = ul.pretrain_choose(prev_index_len + new_len // 2)
        ul_mask_list = np.where(ul_mask.cpu().numpy())[0]

        kl = KmeansLearner(self.args, self.model, self.data, self.prev_index)
        kl_mask = ul.pretrain_choose(num_points)
        kl_mask_list = np.where(kl_mask.cpu().numpy())[0]
        return combine_new_old(kl_mask_list, ul_mask_list, num_points, self.n)

# reimplementation of graph
def centralissimo(G):
    centralities = []
    centralities.append(nx.pagerank(G))                #print 'page rank: check.'
    L = len(centralities[0])
    Nc = len(centralities)
    cenarray = np.zeros((Nc,L))
    for i in range(Nc):
        cenarray[i][list(centralities[i].keys())]=list(centralities[i].values())
    normcen = (cenarray.astype(float)-np.min(cenarray,axis=1)[:,None])/(np.max(cenarray,axis=1)-np.min(cenarray,axis=1))[:,None]
    return normcen

#calculate the percentage of elements smaller than the k-th element
def perc(input,k):
    return sum([1 if i else 0 for i in input<input[k]])/float(len(input))

#calculate the percentage of elements larger than the k-th element
def percd(input,k):
    return sum([1 if i else 0 for i in input>input[k]])/float(len(input))

# quick reimplementation
def perc_full_np(input):
    l = len(input)
    indices = np.argsort(input)
    loc = np.zeros(l, dtype=np.float)
    for i in range(l):
        loc[indices[i]] = i
    return loc / l

class AgeLearner(ActiveLearner):
    def __init__(self, args, model, data, prev_index, train_mask):
        # start_time = time.time()
        super(AgeLearner, self).__init__(args, model, data, prev_index, train_mask)
        self.device = data.x.get_device()

        self.G = tgu.to_networkx(data)
        self.normcen = centralissimo(self.G).flatten()
        self.cenperc = np.asarray([perc(self.normcen,i) for i in range(len(self.normcen))])
        self.NCL = len(np.unique(data.y.cpu().numpy()))
        self.basef = 0.995
        if args.dataset == 'Citeseer':
            self.basef = 0.9
        # print('Age init time', time.time() - start_time)

    def pretrain_choose(self, num_points):
        # start_time = time.time()
        self.model.eval()
        (features, prev_out, no_softmax), out = self.model(self.data)

        if self.args.uncertain_score == 'entropy':
            scores = torch.sum(-F.softmax(prev_out, dim=1) * F.log_softmax(prev_out, dim=1), dim=1)
        elif self.args.uncertain_score == 'margin':
            pred = F.softmax(prev_out, dim=1)
            top_pred, _ = torch.topk(pred, k=2, dim=1)
            # use negative values, since the largest values will be chosen as labeled data
            scores =  (-top_pred[:,0] + top_pred[:,1]).view(-1)
        else:
            raise NotImplementedError

        epoch = len(self.prev_index_list)
        gamma = np.random.beta(1, 1.005-self.basef**epoch)
        alpha = beta = (1-gamma)/2

        softmax_out = F.softmax(prev_out, dim=1).cpu().detach().numpy()
        # print('Age pretrain softmax_out time', time.time() - start_time)
        # start_time = time.time()
        # entrperc = np.asarray([perc(scores,i) for i in range(len(scores))])
        entrperc = perc_full_np(scores.detach().cpu().numpy())
        # print('Age pretrain entrperc time', time.time() - start_time)
        # start_time = time.time()
        kmeans = KMeans(n_clusters=self.NCL, random_state=0, n_init=10).fit(softmax_out)
        # print('Age pretrain kmeans time', time.time() - start_time)
        # start_time = time.time()
        ed=euclidean_distances(softmax_out,kmeans.cluster_centers_)
        # print('Age pretrain eucidean distance time', time.time() - start_time)
        # start_time = time.time()
        ed_score = np.min(ed,axis=1)	#the larger ed_score is, the far that node is away from cluster centers, the less representativeness the node is
        # edprec = np.asarray([percd(ed_score,i) for i in range(len(ed_score))])
        edprec = 1. - perc_full_np(ed_score)
        finalweight = alpha*entrperc + beta*edprec + gamma*self.cenperc
        full_new_index_list = np.argsort(finalweight * self.train_mask.numpy())[::-1][:num_points]
        # print('Age pretrain_choose time', time.time() - start_time)

        return combine_new_old(full_new_index_list, self.prev_index_list, num_points, self.n, in_order=True)

class AnrmabLearner(ActiveLearner):
    def __init__(self, args, model, data, prev_index, train_mask):
        # start_time = time.time()
        super(AnrmabLearner, self).__init__(args, model, data, prev_index, train_mask)
        self.device = data.x.get_device()

        self.y = data.y.detach().cpu().numpy()
        self.NCL = len(np.unique(data.y.cpu().numpy()))

        self.G = tgu.to_networkx(data)
        self.normcen = centralissimo(self.G).flatten()
        self.w = np.array([1., 1., 1.]) # ie, nc, id
        # print('AnrmabLearner init time', time.time() - start_time)

    def pretrain_choose(self, num_points):
        # here we adopt a slightly different strategy which does not exclude sampled points in previous rounds to keep consistency with other methods
        self.model.eval()
        (features, prev_out, no_softmax), out = self.model(self.data)

        if self.args.uncertain_score == 'entropy':
            scores = torch.sum(-F.softmax(prev_out, dim=1) * F.log_softmax(prev_out, dim=1), dim=1)
        elif self.args.uncertain_score == 'margin':
            pred = F.softmax(prev_out, dim=1)
            top_pred, _ = torch.topk(pred, k=2, dim=1)
            # use negative values, since the largest values will be chosen as labeled data
            scores =  (-top_pred[:,0] + top_pred[:,1]).view(-1)
        else:
            raise NotImplementedError

        epoch = len(self.prev_index_list)

        softmax_out = F.softmax(prev_out, dim=1).cpu().detach().numpy()
        kmeans = KMeans(n_clusters=self.NCL, random_state=0, n_init=10).fit(softmax_out)
        ed=euclidean_distances(softmax_out,kmeans.cluster_centers_)
        ed_score = np.min(ed,axis=1)	#the larger ed_score is, the far that node is away from cluster centers, the less representativeness the node is

        q_ie = scores.detach().cpu().numpy()
        q_nc = self.normcen
        q_id = 1. / (1. + ed_score)
        q_mat = np.vstack([q_ie, q_nc, q_id])  # 3 x n
        q_sum = q_mat.sum(axis=1, keepdims=True)
        q_mat = q_mat / q_sum

        w_len = self.w.shape[0]
        p_min = np.sqrt(np.log(w_len) / w_len / num_points)
        p_mat = (1 - w_len*p_min) * self.w / self.w.sum() + p_min # 3

        phi = p_mat[:, np.newaxis] * q_mat # 3 x n
        phi = phi.sum(axis=0) # n

        # sample new points according to phi
        # TODO: change to the sampling method
        if self.args.anrmab_argmax:
            full_new_index_list = np.argsort(phi * self.train_mask.numpy())[::-1][:num_points] # argmax
        else:
            full_new_index_list = np.random.choice(len(phi), num_points, p=(phi * self.train_mask.numpy())/np.sum(phi * self.train_mask.numpy()), replace=False)

        mask = combine_new_old(full_new_index_list, self.prev_index_list, num_points, self.n, in_order=True)
        mask_list = np.where(mask)[0]
        diff_list = np.asarray(list(set(mask_list).difference(set(self.prev_index_list))))

        pred = torch.argmax(out, dim=1).detach().cpu().numpy()
        reward = 1. / num_points / (self.n - num_points) * np.sum((pred[mask_list] == self.y[mask_list]).astype(np.float64) / phi[mask_list]) # scalar
        reward_hat = reward * np.sum(q_mat[:, diff_list] / phi[np.newaxis, diff_list], axis=1)
        # update self.w
        # get current node label epoch
        epoch = self.args.label_list.index(num_points) + 1
        p_const = np.sqrt(np.log(self.n * 10. / 3. / epoch))
        self.w = self.w * np.exp(p_min / 2 * (reward_hat + 1. / p_mat * p_const))

        # import ipdb; ipdb.set_trace()
        # print('Age pretrain_choose time', time.time() - start_time)

        return mask


class UncertaintyLearner(ActiveLearner):
    def __init__(self, args, model, data, prev_index, train_mask):
        super(UncertaintyLearner, self).__init__(args, model, data, prev_index, train_mask)
        self.device = data.x.get_device()

    def pretrain_choose(self, num_points):
        self.model.eval()
        (features, prev_out, no_softmax), out = self.model(self.data)

        if self.args.uncertain_score == 'entropy':
            scores = torch.sum(-F.softmax(prev_out, dim=1) * F.log_softmax(prev_out, dim=1), dim=1)
        elif self.args.uncertain_score == 'margin':
            pred = F.softmax(prev_out, dim=1)
            top_pred, _ = torch.topk(pred, k=2, dim=1)
            # use negative values, since the largest values will be chosen as labeled data
            scores =  (-top_pred[:,0] + top_pred[:,1]).view(-1)
        else:
            raise NotImplementedError


        vals, full_new_index_list = torch.topk(scores, k=num_points)
        full_new_index_list = full_new_index_list.cpu().numpy()

        '''

        # excluding existing indices
        add_index_list = []
        exist_num = 0
        for cur_index in new_index_list:
            if cur_index not in self.prev_index_list:
                exist_num += 1
                add_index_list.append(cur_index)
            if exist_num == num_points - len(self.prev_index_list):
                break

        indices = torch.LongTensor( np.concatenate((self.prev_index_list, add_index_list)) )
        ret_tensor = torch.zeros((self.n), dtype=torch.bool)
        ret_tensor[indices] = 1
        return ret_tensor'''
        return combine_new_old(full_new_index_list, self.prev_index_list, num_points, self.n, in_order=True)

class CoresetLearner(ActiveLearner):
    def __init__(self, args, model, data, prev_index):
        super(CoresetLearner, self).__init__(args, model, data, prev_index)
        self.device = data.x.get_device()

    def pretrain_choose(self, num_points):
        # random selection if the model is untrained
        if self.prev_index is None:
            indices = torch.multinomial(torch.masked_select(torch.ones(self.n), self.train_mask), num_samples=num_points, replacement=False)
            ret_tensor = torch.zeros((self.n), dtype=torch.bool)
            ret_tensor[indices] = 1
            return ret_tensor

        self.model.eval()
        (features, prev_out, no_softmax), out = self.model(self.data)

        features = features.cpu().detach().numpy()
        '''
        # TODO: should be modified to K-center method
        kmeans = KMeans(n_clusters=num_points).fit(features)
        center_dist = pairwise_distances(kmeans.cluster_centers_, features) # k x n

        new_index_list = np.argmin(center_dist, axis=1)
        prev_index_len = len(self.prev_index_list)
        diff_list = np.asarray(list(set(new_index_list).difference(set(self.prev_index_list))))
        indices = torch.LongTensor( np.concatenate((self.prev_index_list, diff_list[:-prev_index_len + num_points])) )
        ret_tensor = torch.zeros((self.n), dtype=torch.bool)
        ret_tensor[indices] = 1
        '''
        if self.args.cluster_method == 'kmeans':
            return kmeans_choose(features, num_points, prev_index_list=self.prev_index_list, n=self.n)
        elif self.args.cluster_method == 'kcenter':
            return kcenter_choose(features, num_points, prev_index_list=self.prev_index_list, n=self.n)
        else:
            raise NotImplementedError

class KmeansLearner(ActiveLearner):
    def __init__(self, args, model, data, prev_index, train_mask):
        super(KmeansLearner, self).__init__(args, model, data, prev_index, train_mask)
        start = time.time()
        self.adj_full = convert_edge2adj(data.edge_index)
        print('Time cost: {}'.format(time.time() - start))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.norm_adj = normalize(self.adj_full + torch.eye(self.n).to(self.device) * self.args.self_loop_coeff)

    def pretrain_choose(self, num_points):
        features = self.data.x
        for k in range(self.args.kmeans_num_layer):
            features = self.norm_adj.matmul(features)
        features = features.cpu().numpy()

        # Note all prev_index_list's are empty since features of KmeansLearner do not rely on previous results, and have not clue of the intermediate model status (it trains from scratch)
        if self.args.cluster_method == 'kmeans':
            return kmeans_choose(features, num_points, prev_index_list=[], n=self.n)
        elif self.args.cluster_method == 'kcenter':
            return kcenter_choose(features, num_points, prev_index_list=[], n=self.n)
        elif self.args.cluster_method == 'kmedoids':
            return kmedoids_choose(features, num_points, prev_index_list=[], n=self.n)
        else:
            raise NotImplementedError
        '''
        kmeans = KMeans(n_clusters=num_points).fit(features)
        center_dist = pairwise_distances(kmeans.cluster_centers_, features) # k x n
        indices = torch.LongTensor(np.argmin(center_dist, axis=1))
        ret_tensor = torch.zeros((self.n), dtype=torch.bool)
        ret_tensor[indices] = 1
        return ret_tensor
        '''

class RandomLearner(ActiveLearner):
    def __init__(self, args, model, data, prev_index, train_mask):
        super(RandomLearner, self).__init__(args, model, data, prev_index, train_mask)
    def pretrain_choose(self, num_points):
        indices = torch.multinomial(self.train_mask.to(torch.float64), num_samples=num_points, replacement=False)
        ret_tensor = torch.zeros((self.n), dtype=torch.bool)
        ret_tensor[indices] = 1
        return ret_tensor

class DegreeLearner(ActiveLearner):
    def __init__(self, args, model, data, prev_index, train_mask):
        super(DegreeLearner, self).__init__(args, model, data, prev_index, train_mask)
        start = time.time()
        self.adj_full = convert_edge2adj(data.edge_index)
        print('Time cost: {}'.format(time.time() - start))
    def pretrain_choose(self, num_points):
        ret_tensor = torch.zeros((self.n), dtype=torch.bool)
        degree_full = self.adj_full.sum(dim=1)
        vals, indices = torch.topk(degree_full * self.train_mask, k=num_points)
        ret_tensor[indices] = 1
        return ret_tensor

# impose all category constraint
# no direct linkage
class NonOverlapDegreeLearner(ActiveLearner):
    def __init__(self, args, model, data, prev_index, train_mask):
        super(NonOverlapDegreeLearner, self).__init__(args, model, data, prev_index, train_mask)
        start = time.time()
        self.adj_full = convert_edge2adj(data.edge_index)
        print('Time cost: {}'.format(time.time() - start))
    def pretrain_choose(self, num_points):
        # select by degree
        ret_tensor = torch.zeros((self.n), dtype=torch.bool)
        degree_full = self.adj_full.sum(dim=1)
        vals, indices = torch.sort(degree_full, descending=True)

        index_list = []

        num = 0
        for i in indices:
            edge_flag = False
            for j in index_list:
                if self.adj_full[i, j] != 0:
                    edge_flag = True
                    break
            if not edge_flag:
                index_list.append(i)
                num += 1
            if num == num_points:
                break

        ret_tensor[torch.LongTensor(index_list)] = 1
        return ret_tensor

def get_current_neighbors_dense(cur_nodes, adj2):
    if len(cur_nodes) == 0:
        return np.ones(adj2.shape[0])
    neighbors = (adj2[list(cur_nodes)].sum(axis = 0) != 0) + 0
    return neighbors if len(cur_nodes) != 0 else np.ones(adj2.shape[0])

class GrainLearner(ActiveLearner):
    def __init__(self, args, model, data, prev_index, train_mask):
        super(GrainLearner, self).__init__(args, model, data, prev_index, train_mask)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        adj = normalize_adj(convert_edge2adj_sparse(data.edge_index)).todense()
        adj_matrix = torch.FloatTensor(adj).to(self.device)
        adj_matrix2 = torch.mm(adj_matrix, adj_matrix).to(self.device)
        features = torch.from_numpy(normalize_row(data.x.cpu())).to(self.device)
        features_aax = torch.mm(adj_matrix2, features)
        num_nodes = data.num_nodes

        g = torch.mm(features_aax, features_aax.T)
        h = torch.diag(g).repeat(features_aax.shape[0], 1)
        distance_aax = torch.sqrt(h + h.T - 2 * g)
        distance_aax = (distance_aax - torch.min(distance_aax)) / (torch.max(distance_aax) - torch.min(distance_aax))

        self.adj = adj
        self.adj2 = adj_matrix2
        self.distance_aax = distance_aax
        self.radium = args.radium


    def pretrain_choose(self, num_points):
        num_nodes = self.data.num_nodes
        balls_dict = dict()
        covered_balls = set()
        balls = self.distance_aax <= self.radium

        available = list(torch.squeeze(torch.nonzero(self.train_mask), dim=1).cpu().numpy())
        dot_results = torch.matmul((self.adj2 != 0).to(torch.float64),
                                   balls.to(torch.float64))
        for node in available:
            # neighbors_tmp = torch.unsqueeze((self.adj2 != 0)[node], dim=1)
            # dot_result = np.matmul(balls, neighbors_tmp).T
            # balls_dict[node] = set(np.nonzero(dot_result[0])[0])
            balls_dict[node] = set(torch.squeeze(torch.nonzero(dot_results[node]), dim=1).cpu().numpy())

        # choose the node
        ret_tensor = torch.zeros(num_nodes, dtype=torch.bool)
        while torch.sum(ret_tensor) < num_points:
            node_max = max(available, key=lambda n: len(covered_balls | balls_dict[n]), default=None)
            ret_tensor[node_max] = 1
            available.remove(node_max)
            covered_balls = covered_balls.union(balls_dict[node_max])
            res_ball_num = num_nodes - len(covered_balls)
            if res_ball_num == 0:
                break

        return ret_tensor