import torch
import torch.nn.functional as F
import time

from torch_geometric.utils import to_torch_coo_tensor
import torch_sparse
from torch.nn import Parameter, Linear
from torch.nn.init import xavier_uniform_
from torch_geometric.nn import GCNConv

from utils import convert_edge2adj, normalize

def get_model(model_name):
    return eval(model_name)

class MatrixGCN(torch.nn.Module):
    def __init__(self, args, data):
        super(MatrixGCN, self).__init__()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.mat = normalize(convert_edge2adj(data.edge_index, data.num_nodes) + torch.eye(data.num_nodes)).to(device)

        start = time.time()

        self.mat = Parameter(normalize(convert_edge2adj(data.edge_index, data.num_nodes) + torch.eye(data.num_nodes)), requires_grad=False)
        self.linear1 = Parameter(torch.Tensor(args.num_features, args.hid_dim))
        self.linear2 = Parameter(torch.Tensor(args.hid_dim, args.num_classes))

        xavier_uniform_(self.linear1)
        xavier_uniform_(self.linear2)

        self.args = args
        print('MatrixGCN init time cost: {}'.format(time.time() - start))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.mat.matmul(x.matmul(self.linear1))
        hid_x = F.relu(x)
        drop_x = F.dropout(hid_x, self.args.dropout, training=self.training)
        bef_linear2 = self.mat.matmul(drop_x)
        fin_x = bef_linear2.matmul(self.linear2)
        if self.args.multilabel:
            out = fin_x # without sigmoid
        else:
            out = F.log_softmax(fin_x, dim=1)


        return (hid_x, bef_linear2, fin_x), out

class MatrixGCN_layer3(torch.nn.Module):
    def __init__(self, args, data):
        super(MatrixGCN_layer3, self).__init__()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.mat = normalize(convert_edge2adj(data.edge_index, data.num_nodes) + torch.eye(data.num_nodes)).to(device)

        start = time.time()

        self.mat = Parameter(normalize(convert_edge2adj(data.edge_index, data.num_nodes) + torch.eye(data.num_nodes)), requires_grad=False)
        self.linear1 = Parameter(torch.Tensor(args.num_features, args.hid_dim))
        self.linear2 = Parameter(torch.Tensor(args.hid_dim, args.hid_dim))
        self.linear3 = Parameter(torch.Tensor(args.hid_dim, args.num_classes))

        xavier_uniform_(self.linear1)
        xavier_uniform_(self.linear2)
        xavier_uniform_(self.linear3)

        self.args = args
        print('MatrixGCN init time cost: {}'.format(time.time() - start))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.mat.matmul(x.matmul(self.linear1))
        hid_x = F.relu(x)
        drop_x = F.dropout(hid_x, self.args.dropout, training=self.training)
        bef_linear2 = self.mat.matmul(drop_x)
        af_linear2 = bef_linear2.matmul(self.linear2)
        af_linear2 = F.relu(x)
        af_linear2 = F.dropout(af_linear2, self.args.dropout, training=self.training)
        bf_linear3 = self.mat.matmul(af_linear2)
        fin_x = bf_linear3.matmul(self.linear3)
        if self.args.multilabel:
            out = fin_x # without sigmoid
        else:
            out = F.log_softmax(fin_x, dim=1)


        return (hid_x, bf_linear3, fin_x), out
# Network definition, could be refactored
class GCN(torch.nn.Module):
    def __init__(self, args, data):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.hid_dim)
        self.conv2 = GCNConv(args.hid_dim, args.num_classes)
        self.args = args

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        hid_x = F.relu(x)
        x = F.dropout(hid_x, self.args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        if self.args.multilabel:
            out = x # without sigmoid
        else:
            out = F.log_softmax(x, dim=1)

        # TODO: the final element in the triple is added for compatability
        return (hid_x, x, x), out
    
class SGC(torch.nn.Module):
    def __init__(self, args, data):
        super(SGC, self).__init__()
        self.mat = Parameter(normalize(convert_edge2adj(data.edge_index, data.num_nodes) + torch.eye(data.num_nodes)), requires_grad=False)
        self.linear1 = Parameter(torch.Tensor(args.num_features, args.hid_dim))
        self.linear2 = Parameter(torch.Tensor(args.hid_dim, args.num_classes))

        xavier_uniform_(self.linear1)
        xavier_uniform_(self.linear2)

        self.args = args

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        hid_x = self.mat.matmul(x.matmul(self.linear1))
        drop_x = F.dropout(hid_x, self.args.dropout, training=self.training)
        bef_linear2 = self.mat.matmul(drop_x)
        fin_x = bef_linear2.matmul(self.linear2)
        if self.args.multilabel:
            out = fin_x # without sigmoid
        else:
            out = F.log_softmax(fin_x, dim=1)

        return (hid_x, bef_linear2, fin_x), out

class H2GCN(torch.nn.Module):
    def __init__(self, args, data):
        super(H2GCN, self).__init__()
        self.dropout = args.dropout
        self.dropout = args.dropout
        self.k = 2
        self.act = F.relu
        self.use_relu = True
        self.w_embed = torch.nn.Parameter(
            torch.zeros(size=(args.num_features, args.hid_dim)),
            requires_grad=True
        )
        self.w_classify = torch.nn.Parameter(
            torch.zeros(size=((2 ** (self.k + 1) - 1) * args.hid_dim, args.num_classes)),
            requires_grad=True
        )
        self.params = [self.w_embed, self.w_classify]
        self.initialized = False
        self.a1 = None
        self.a2 = None
        self.reset_parameter()
        self.args = args

    def reset_parameter(self):
        torch.nn.init.xavier_uniform_(self.w_embed)
        torch.nn.init.xavier_uniform_(self.w_classify)

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1, 0),
            size=csp.size(),
            dtype=torch.float
        )

    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        )

    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n)
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    def _prepare_prop(self, adj):
        adj = torch.tensor(adj)
        adj_size = adj.size(0)
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(adj_size)), list(range(adj_size))],
            values=[1.0] * adj_size,
            size=(adj_size, adj_size),
            dtype=torch.float
        )
        # initialize A1, A2
        a1 = self._indicator(adj - sp_eye)
        a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)
        # norm A1 A2
        self.a1 = self._adj_norm(a1)
        self.a2 = self._adj_norm(a2)

    def forward(self, data):
        x = data.x
        adj = to_torch_coo_tensor(data.edge_index)

        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation

        rs = [self.act(torch.mm(torch.tensor(x), self.w_embed))]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(self.act(torch.cat([r1, r2], dim=1)))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)

        fin_x = torch.mm(r_final, self.w_classify)

        return (x, x, fin_x), torch.softmax(torch.mm(r_final, self.w_classify), dim=1)


class MLP(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()
        self.lr1 = torch.nn.Linear(nfeat, nhid)
        self.lr2 = torch.nn.Linear(nhid, nclass)

        self.dropout = dropout

    def forward(self, x):
        x = torch.from_numpy(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lr1(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.lr2(x)
        return x