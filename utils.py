import torch
import numpy as np
import time
import torch_geometric.utils as tgu
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import scipy.sparse as sp

# Transformation utils
# construct adj matrix from edge_index
# TODO: should consider GPU/CPU convertion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_edge2adj(edge_index):
    # float type
    mat = tgu.to_scipy_sparse_matrix(edge_index)
    mat = mat + mat.T.multiply(mat.T > mat) - mat.multiply(mat.T > mat)
    mat = torch.from_numpy(np.array(mat.todense())).to(device)
    return mat

def convert_edge2adj_sparse(edge_index):
    # float type
    mat = tgu.to_scipy_sparse_matrix(edge_index)
    mat = mat + mat.T.multiply(mat.T > mat) - mat.multiply(mat.T > mat)
    mat = mat + sp.eye(mat.shape[0])
    return mat

def normalize(adj):
    inv_sqrt_degree = 1. / torch.sqrt(adj.sum(dim=1, keepdim=False))
    inv_sqrt_degree[inv_sqrt_degree == float("Inf")] = 0
    return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]

def normalize_row(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()

# Clustering utils
# Note: code modified from https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py
def kcenter_choose(features, num_points, prev_index_list, n):
    prev_index_len = len(prev_index_list)
    # print('DEBUG n: {}'.format(n))
    if prev_index_len == 0:
        # one point initialization
        prev_index_list = [np.random.randint(n)]
        prev_index_len = len(prev_index_list)

    # kCenter
    # print('DEBUG min_distances shape', pairwise_distances(features, features[prev_index_list]).shape)
    min_distances = np.min(pairwise_distances(features, features[prev_index_list]), axis=1)
    # select num_points new indices
    new_index_list = []
    # import ipdb; ipdb.set_trace()
    for _ in range(num_points - prev_index_len):
        ind = np.argmax(min_distances)
        # print('DEBUG ind', ind)
        assert ind not in prev_index_list
        new_index_list.append(ind)
        # update distances
        new_distances = pairwise_distances(features, features[ind].reshape(1, -1)).reshape(-1)
        min_distances = np.minimum(min_distances, new_distances)

    indices = torch.LongTensor( np.concatenate((prev_index_list, new_index_list)) )
    ret_tensor = torch.zeros((n), dtype=torch.uint8)
    ret_tensor[indices] = 1
    return ret_tensor

def kmeans_choose(features, num_points, prev_index_list, n):
    kmeans = KMeans(n_clusters=num_points).fit(features)
    center_dist = pairwise_distances(kmeans.cluster_centers_, features) # k x n
    full_new_index_list = np.argmin(center_dist, axis=1)
    # TODO: difference of in_order when implementing coreset
    ret_tensor = combine_new_old(full_new_index_list, prev_index_list, 
                                 num_points, n, in_order=True)   
    return ret_tensor


def kmedoids_choose(features, num_points, prev_index_list, n):
    from pyclustering.cluster.kmedoids import kmedoids
    from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

    start_time = time.time()
    # Prepare initial centers using K-Means++ method.
    initial_centers = kmeans_plusplus_initializer(features, num_points).initialize() # num_points x feature_dim
    distances = pairwise_distances(features, initial_centers, n_jobs=-1) # parallel computing, n x num_points
    initial_medoids = np.argmin(distances, axis=0)
    print('Medoids number', len(initial_medoids))
    # Create instance of K-Medoids algorithm.
    kmedoids_instance = kmedoids(features, initial_medoids)
    # Run cluster analysis and obtain results.
    kmedoids_instance.process()
    print('K-Medoids clustering time', time.time() - start_time)
    full_new_index_list = kmedoids_instance.get_medoids()
    # TODO: difference of in_order when implementing coreset
    ret_tensor = combine_new_old(full_new_index_list, prev_index_list, 
                                 num_points, n, in_order=True)   
    return ret_tensor


def combine_new_old(full_new_index_list, prev_index_list, num_points, n, in_order=True):
    prev_index_len = len(prev_index_list)
    if in_order:
        # in-order difference
        new_index_list = []
        exist_num = 0
        for ind in full_new_index_list:
            if ind not in prev_index_list:
                exist_num += 1
                new_index_list.append(ind)
            if exist_num == num_points - prev_index_len:
                break
                # return new_index_list
    else:
        # random difference
        diff_list = np.asarray(list(set(full_new_index_list).difference(set(prev_index_list))))
        new_index_list = diff_list[:-prev_index_len + num_points]
        # return new_index_list
    indices = torch.LongTensor( np.concatenate((prev_index_list, new_index_list)) )
    ret_tensor = torch.zeros((n), dtype=torch.bool)
    ret_tensor[indices] = 1
    return ret_tensor

    