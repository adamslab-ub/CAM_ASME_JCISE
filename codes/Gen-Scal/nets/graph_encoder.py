import torch
import numpy as np
from torch import nn
import math

class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        h = self.layers(h)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

class CCN3(nn.Module):

    def __init__(
            self,
            node_dim = 3,
            embed_dim = 128,
            n_layers = 2,
    ):
        super(CCN3, self).__init__()
        self.init_embed = nn.Linear(node_dim, embed_dim)
        self.init_neighbour_embed = nn.Linear(node_dim, embed_dim)
        self.neighbour_encode = nn.Linear(node_dim, embed_dim)
        self.init_embed_depot = nn.Linear(2, embed_dim)
        self.final_embedding = nn.Linear(embed_dim, embed_dim)

        self.init_embed_L2 = nn.Linear(embed_dim, embed_dim)
        self.neighbour_encode_L2 = nn.Linear(embed_dim, embed_dim)
        self.final_embedding_L2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, X, mask=None):
        x = torch.cat((X['loc'], X['deadline'][:, :, None]), 2)
        x2 = x[:, :, 0:2]
        activ = nn.LeakyReLU()
        # F0_embedding_2d = self.init_embed_2d(x2)
        F0_embedding_3d = self.init_embed(x)
        # F0_embedding.reshape([1])

        dist_mat = (x2[:, None] - x2[:, :, None]).norm(dim=-1, p=2)  ## device to cuda to be added
        neighbors = dist_mat.sort().indices[:, :, :8]  # for 6 neighbours
        neighbour = x[:, neighbors][0]
        neighbour_delta = neighbour - x[:, :, None, :]
        neighbour_delta_embedding = self.neighbour_encode(neighbour_delta)
        concat = torch.cat((F0_embedding_3d[:, :, None, :], neighbour_delta_embedding), 2)

        F_embed_final = self.final_embedding(concat).sum(dim=2)

        xl2 = self.init_embed_L2(F_embed_final)
        neighbour_delta_L2 = F_embed_final[:, neighbors][0] - F_embed_final[:, :, None, :]
        neighbour_delta_embedding_L2 = self.neighbour_encode_L2(neighbour_delta_L2)
        concat_L2 = torch.cat((xl2[:, :, None, :], neighbour_delta_embedding_L2), 2)
        F_embed_final_L2 = self.final_embedding_L2(concat_L2).sum(dim=2)

        init_depot_embed = self.init_embed_depot(X['depot'])
        h = torch.cat((init_depot_embed, F_embed_final), -2)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )




class CCN_New(nn.Module):

    def __init__(
            self,
            node_dim = 3,
            embed_dim = 128,
            n_layers = 2,
    ):
        super(CCN_New, self).__init__()
        self.init_embed = nn.Linear(node_dim, embed_dim)

        self.neighbour_encode = nn.Linear(embed_dim, embed_dim)
        self.L1_embed = nn.Linear(2*embed_dim, embed_dim)


        self.neighbour_encode_L2 = nn.Linear(embed_dim, embed_dim)
        self.L2_embed = nn.Linear(2 * embed_dim, embed_dim)

        self.init_embed_depot = nn.Linear(2, embed_dim)

    def forward(self, X, mask=None):
        x = torch.cat((X['loc'], X['deadline'][:, :, None]), 2)
        x2 = x[:, :, 0:2]
        # F0_embedding_2d = self.init_embed_2d(x2)
        F0_embedding_3d = self.init_embed(x)
        # F0_embedding.reshape([1])

        dist_mat = (x2[:, None] - x2[:, :, None]).norm(dim=-1, p=2)  ## device to cuda to be added
        neighbors = dist_mat.sort().indices[:, :, :10]  # for 6 neighbours
        # neighbour = x[:, neighbors][0]
        indices = torch.tensor([i for i in range(x.shape[0])])
        # node_indices = torch.tensor([i for i in range(x.shape[1])])
        omega_0_v = torch.tensor([i for i in range(x.shape[1])])
        omega_1_v =[omega_0_v[neighbors[:,:,i]] for i in range(neighbors.shape[2])]
        omega_1_v = torch.stack(omega_1_v, dim=2)
        omega_2_v = [omega_1_v[indices[:,None], neighbors[:,:,i]] for i in range(neighbors.shape[2])]
        omega_2_v = torch.stack(omega_2_v, dim=-1).view(x.shape[0], x.shape[1], -1)

        dt1 = ((omega_0_v.unsqueeze(1).expand(x.shape[0], x.shape[1], 1)).unsqueeze(2).expand(x.shape[0], x.shape[1],
                                                                                        x.shape[1], 1)).unsqueeze(
            3).expand(x.shape[0], x.shape[1], x.shape[1], neighbors.shape[2], 1)
        dt2 = (omega_1_v.unsqueeze(3).expand(x.shape[0], x.shape[1], omega_1_v.shape[2], 1)).unsqueeze(2).expand(x.shape[0], x.shape[1],x.shape[1], omega_1_v.shape[2], 1)
        X_1 = (dt1 == dt2).to(torch.float32)
        at = torch.stack([F0_embedding_3d[indices[:, None],omega_1_v[:,:,i]] for i in range(omega_1_v.shape[2])], dim=-2)
        # at = at - F0_embedding_3d[:,:,None]
        # phi_2 = torch.tensordot(X_1[:,:,:,:,None], (at.unsqueeze(dim=-2)).expand(x.shape[0], x.shape[1],x.shape[1],omega_1_v.shape[2],omega_1_v.shape[2],F0_embedding_3d.shape[2]))

        phi_1 = X_1*(at.unsqueeze(dim=2)).expand(x.shape[0], x.shape[1],x.shape[1],omega_1_v.shape[2],F0_embedding_3d.shape[2])
        phi_1_red = (phi_1.sum(dim=2)).sum(dim=2) - F0_embedding_3d
        f_1_N = self.neighbour_encode(phi_1_red)

        F_1 = self.L1_embed(torch.cat((F0_embedding_3d, f_1_N), dim=-1))

        dt1 = (omega_1_v.unsqueeze(2).expand(x.shape[0], x.shape[1], omega_2_v.shape[2], omega_1_v.shape[2])).unsqueeze(2).expand(x.shape[0], x.shape[1],x.shape[1], omega_2_v.shape[2], omega_1_v.shape[2])
        dt2 = (omega_2_v.unsqueeze(3).expand(x.shape[0], x.shape[1], omega_2_v.shape[2], omega_1_v.shape[2])).unsqueeze(2).expand(x.shape[0], x.shape[1],x.shape[1], omega_2_v.shape[2], omega_1_v.shape[2])
        X_2 = (dt1 == dt2).to(torch.float32)
        at_2 = torch.stack([F_1[indices[:, None], omega_2_v[:, :, i]] for i in range(omega_2_v.shape[2])],
                         dim=-2)

        phi_2 = torch.matmul(X_2.transpose(dim0=-1, dim1=-2),(at_2.unsqueeze(dim=2)).expand(x.shape[0], x.shape[1], x.shape[1],omega_2_v.shape[2],F0_embedding_3d.shape[2]))

        phi_2_red = (phi_2.sum(dim=2)).sum(dim=2) - F_1

        f_2_N = self.neighbour_encode_L2(phi_2_red)

        F_2 = self.L1_embed(torch.cat((F_1, f_2_N), dim=-1))

        return (
            F_2,  # (batch_size, graph_size, embed_dim)
            F_2.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

class SimpleNN(nn.Module):

    def __init__(self, node_dim = 3,
                 embed_dim = 128):
        super(SimpleNN, self).__init__()
        self.init_embed = nn.Linear(node_dim, embed_dim)
        self.init_embed_depot = nn.Linear(2, embed_dim)
        self.final_embedding = nn.Linear(embed_dim, embed_dim)

    def forward(self, X):
        x = torch.cat((X['loc'], X['deadline'][:, :, None]), 2)
        F0_embedding_3d = self.init_embed(x)
        F_embed_final = self.final_embedding(F0_embedding_3d)

        init_depot_embed = self.init_embed_depot(X['depot'])
        h = torch.cat((init_depot_embed, F_embed_final), -2)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )




class GCAPCN(nn.Module):

    def __init__(self,
                 n_layers = 3,
                 n_hops = 5,
                 n_dim = 128,
                 n_p = 3,
                 node_dim = 3
                 ):
        super(GCAPCN, self).__init__()
        self.n_layers = n_layers
        self.n_hops = n_hops
        self.n_dim = n_dim
        self.n_p = n_p
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim)
        self.init_embed_depot = nn.Linear(2, n_dim)
        self.W1 = nn.Linear(n_dim*n_p, n_dim)
        self.W2 = nn.Linear(n_dim * n_p, n_dim)
        self.activ = nn.LeakyReLU()

    def forward(self, data, mask=None):
        X = torch.cat((data['loc'], data['deadline'][:, :, None]), 2)
        X_loc = X[:, :, 0:2]
        distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
        A = (distance_matrix < .3).to(torch.int8)
        num_samples, num_locations, _ = X.size()
        D = torch.mul(torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
                      (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))
        L = D - A
        F0 = self.init_embed(X)

        g1 = torch.cat((F0[:,:,:,None], torch.matmul(L,F0)[:,:,:,None], torch.matmul(torch.matmul(L,L), F0)[:,:,:,None]), -1).reshape((num_samples, num_locations, -1))

        F1 =  self.activ(self.W1(g1))
        g2 = torch.cat((F1[:,:,:,None], torch.matmul(L,F1)[:,:,:,None], torch.matmul(torch.matmul(L,L), F1)[:,:,:,None]), -1).reshape((num_samples, num_locations, -1))
        init_depot_embed = self.init_embed_depot(data['depot'])
        h =  self.activ(self.W2(g2))
        h = torch.cat((init_depot_embed, h), -2)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

