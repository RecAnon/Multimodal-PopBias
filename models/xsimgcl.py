# coding: utf-8
# @email: enoche.chow@gmail.com


import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.loss_SELFRec import bpr_loss, l2_reg_loss, InfoNCE


class XSimGCL(GeneralRecommender):
    def __init__(self, config, dataset):
        super(XSimGCL, self).__init__(config, dataset)

        self.embedding_dim = config["embedding_size"]
        self.cl_rate = config["lambda"]
        self.eps = float(config["eps"])
        self.temp = float(config["tau"])
        self.n_layers, self.layer_cl = [int(x) for x in config["n_layer_l_star"]]
        self.reg = float(config["reg.lambda"])
        self.n_nodes = self.n_users + self.n_items
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        print("Getting adj mat")
        # self.norm_adj = self.get_adj_mat()
        # self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)

        self.embedding_dict = self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict(
            {
                "user_emb": nn.Parameter(
                    initializer(torch.empty(self.n_users, self.embedding_dim))
                ),
                "item_emb": nn.Parameter(
                    initializer(torch.empty(self.n_items, self.embedding_dim))
                ),
            }
        )
        return embedding_dict

    def pre_epoch_processing(self):
        pass

    def get_norm_adj_mat(self):
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(
            i, data, torch.Size((self.n_nodes, self.n_nodes))
        )

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat(
            [self.embedding_dict["user_emb"], self.embedding_dict["item_emb"]], 0
        )
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += (
                    torch.sign(ego_embeddings)
                    * F.normalize(random_noise, dim=-1)
                    * self.eps
                )
            all_embeddings.append(ego_embeddings)
            if k == self.layer_cl - 1:
                all_embeddings_cl = ego_embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(
            final_embeddings, [self.n_users, self.n_items]
        )
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(
            all_embeddings_cl, [self.n_users, self.n_items]
        )
        if perturbed:
            return (
                user_all_embeddings,
                item_all_embeddings,
                user_all_embeddings_cl,
                item_all_embeddings_cl,
            )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        user_idx, pos_idx, neg_idx = interaction
        rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb = self.forward(True)
        user_emb, pos_item_emb, neg_item_emb = (
            rec_user_emb[user_idx],
            rec_item_emb[pos_idx],
            rec_item_emb[neg_idx],
        )
        rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
        cl_loss = self.cl_rate * self.cal_cl_loss(
            [user_idx, pos_idx], rec_user_emb, cl_user_emb, rec_item_emb, cl_item_emb
        )
        batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
        return batch_loss

    def cal_cl_loss(self, idx, user_view1, user_view2, item_view1, item_view2):
        u_idx = torch.unique(idx[0])
        i_idx = torch.unique(idx[1])
        user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(False)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
