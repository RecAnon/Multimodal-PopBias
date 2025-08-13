import torch
import torch.nn as nn
from utils.data_utils import OODDataBuilder
from utils.encoding_utils import (
    next_batch_pairwise_ood,
    InfoNCE,
    sparse_mx_to_torch_sparse_tensor,
)
import torch.nn.functional as F
import numpy as np


class Encoder(object):
    def __init__(
        self,
        args,
        training_data,
        in_dist_valid_data,
        ood_valid_data,
        all_valid_data,
        in_dist_test_data,
        ood_test_data,
        all_test_data,
        user_num,
        item_num,
        user_idx,
        item_idx,
        device,
        user_content=None,
        item_content=None,
    ):
        self.data = OODDataBuilder(
            training_data,
            in_dist_valid_data,
            ood_valid_data,
            all_valid_data,
            in_dist_test_data,
            ood_test_data,
            all_test_data,
            user_num,
            item_num,
            user_idx,
            item_idx,
            user_content,
            item_content,
        )
        top = args.topN.split(",")
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        self.model_name = args.model
        self.dataset_name = args.dataset
        self.emb_size = args.emb_size
        self.batch_size = args.bs
        self.device = device
        self.temp = args.tau
        self.decay_lr, self.maxEpoch = args.decay_lr_epoch
        self.weight_decay = args.reg
        self.lRate = args.lr
        self.t_weight = args.t_weight

        self.model = Encoder_Learner(self.data, self.emb_size, self.t_weight)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lRate,
            weight_decay=self.weight_decay,
        )
        if self.decay_lr:
            steps_per_epoch = len(self.data.training_data) // (self.batch_size)
            t_max = int(steps_per_epoch * self.maxEpoch)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=0.0
            )

        for epoch in range(self.maxEpoch):
            losses = []
            for n, batch in enumerate(
                next_batch_pairwise_ood(self.data, self.batch_size)
            ):
                user_idx, item_idx, _ = batch
                user_vecs, feats = model(True, user_idx, item_idx)
                cl_loss = InfoNCE(user_vecs, feats, self.temp)
                # Backward and optimize
                optimizer.zero_grad()
                cl_loss.backward()
                optimizer.step()
                losses.append(cl_loss.item())
                if self.decay_lr:
                    scheduler.step()
            avg_loss = np.mean(losses)

        with torch.no_grad():
            self.user_emb, self.item_emb = self.model.eval()(perturbed=False)

        with open("./data/%s/outputs/Encoded_KNN.npy" % self.args.dataset, "wb") as f:
            np.save(f, self.item_emb.cpu().numpy()[self.data.mapped_item_idx])

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()


class Encoder_Learner(nn.Module):
    def __init__(self, data, emb_size, t_weight):
        super(Encoder_Learner, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.t_weight = t_weight
        self.v_weight = 1 - t_weight
        self.feat_tensor = torch.from_numpy(data.mapped_item_content).cuda()
        self.v_feats = F.normalize(self.feat_tensor[:, :4096])
        self.t_feats = F.normalize(self.feat_tensor[:, 4096:])
        self.ui_adj_tensor = (
            sparse_mx_to_torch_sparse_tensor(data.interaction_mat)
            .to_sparse_csr()
            .cuda()
        )
        self.total_item_num = self.ui_adj_tensor.shape[1]

        self.feat_shape = self.feat_tensor.shape[1]
        self.v_layer = nn.Sequential(
            nn.Linear(4096, 192),
            nn.BatchNorm1d(192, track_running_stats=False),
            nn.ReLU(),
        )

        self.t_layer = nn.Sequential(
            nn.Linear(384, 192),
            nn.BatchNorm1d(192, track_running_stats=False),
            nn.ReLU(),
        )
        self.final_layer = nn.Linear(192, self.emb_size)

    def forward(self, perturbed=False, user_idx=None, item_idx=None):
        if perturbed:
            interaction_tensor_batch = sparse_mx_to_torch_sparse_tensor(
                self.data.interaction_mat[user_idx]
            ).cuda()
            feats_out = self.final_layer(
                self.v_weight * self.v_layer(self.v_feats)
                + self.t_weight * self.t_layer(self.t_feats)
            )

            user_vecs = torch.sparse.mm(interaction_tensor_batch, feats_out)
            feats_out = feats_out[item_idx]
        else:
            feats_out = self.final_layer(
                self.v_weight * self.v_layer(self.v_feats)
                + self.t_weight * self.t_layer(self.t_feats)
            )
            user_vecs = torch.sparse.mm(self.ui_adj_tensor, feats_out)

        return F.normalize(user_vecs), F.normalize(feats_out)
