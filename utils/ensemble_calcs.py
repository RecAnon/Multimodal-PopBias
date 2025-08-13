import torch
import numpy as np
import torch.nn.functional as F
import pickle
from utils.encoding_utils import (
    sparse_mx_to_torch_tensor,
    sparse_mx_to_torch_sparse_tensor,
)
from utils.eval import get_rec_list_full, get_aplt, ranking_evaluation
from utils.data_utils import get_ood_data
import warnings
from itertools import product

warnings.filterwarnings("ignore")

DATASETS_PATH = "./data"


def get_metrics(top_inds, top_values, item_idx, k, test=True):
    """
    Get OOD and ID metrics plus APLT.
    """
    holdout_set = data.overall_test_set if test else data.overall_valid_set
    ood_holdout_set = data.ood_test_set if test else data.ood_valid_set
    id_holdout_set = data.in_dist_test_set if test else data.in_dist_valid_set
    rec_list = get_rec_list_full(data, holdout_set, top_inds, top_values, item_idx)
    aplt = get_aplt(rec_list, long_tail)
    ood_out = ranking_evaluation(
        ood_holdout_set,
        {u: v for u, v in rec_list.items() if u in ood_holdout_set},
        [k],
    )[0]
    id_out = ranking_evaluation(
        id_holdout_set, {u: v for u, v in rec_list.items() if u in id_holdout_set}, [k]
    )[0]
    return id_out, ood_out, aplt


def get_knn_scores(
    item_emb, interaction_tensor, inter_matrix_batch_zero, knn_k=10, num_batches=4
):
    """
    Given item embeddings, calculates user preference scores from user-item
    interaction matrix.
    """
    vals, inds = [], []
    batch_size = int(item_emb.shape[0] // num_batches) + 1
    for batch_num in range(num_batches):
        item_vec_sim = (
            item_emb[batch_num * batch_size : (batch_num + 1) * batch_size] @ item_emb.T
        )
        # Omit self loops in neighbor calculations
        knn_val, knn_ind = torch.topk(item_vec_sim, knn_k+1, dim=-1)
        vals.append(knn_val[:,1:])
        inds.append(knn_ind[:,1:])
        del item_vec_sim
        torch.cuda.empty_cache()
    knn_val = torch.cat(vals).cuda()
    knn_ind = torch.cat(inds).cuda()
    knn_sim = (
        torch.zeros((item_emb.shape[0], item_emb.shape[0]))
        .cuda()
        .scatter_(-1, knn_ind[:, :knn_k], knn_val[:, :knn_k])
        .to_sparse_coo()
    )
    final_sim = torch.sparse_coo_tensor(
        knn_sim.indices(), knn_sim.values(), size=(item_emb.shape[0], item_emb.shape[0])
    ).to_dense()
    mm_out = torch.sparse.mm(interaction_tensor.cuda(), final_sim.T)
    mm_out_masked = mm_out.cpu() * inter_matrix_batch_zero
    torch.cuda.empty_cache()
    return mm_out_masked


def get_warm_model_pred(dataset, inter_matrix_batch_zero, model="XSimGCL"):
    """
    Get masked predictions from multimodal model (e.g. XSimGCL, SOIL)
    """
    model_user, model_item = torch.load(
        "%s/model_files/%s_ood_%s.pt" % (DATASETS_PATH, dataset, model)
    )
    model_pred = model_user[[int(k) for k in data.user]] @ model_item.T
    return model_pred * inter_matrix_batch_zero


def get_results(scores, k=20):
    top_values, top_inds = torch.topk(scores, k=k)
    rec_id, rec_ood, aplt = get_metrics(top_inds, top_values, data.mapped_item_idx, k)
    return rec_id + rec_ood + [aplt]


def get_raw_scores(
    item_content, interaction_tensor, inter_matrix_batch_zero, v_feat_size=4096
):
    """
    Calculates Raw-KNN scores for input multimodal features.
    """
    v_norm = F.normalize(item_content[:, :v_feat_size]).float()
    t_norm = F.normalize(item_content[:, v_feat_size:]).float()
    t_weight = 0.9

    raw_v_scores = F.normalize(
        get_knn_scores(v_norm, interaction_tensor, inter_matrix_batch_zero, knn_k=10)
    )
    raw_t_scores = F.normalize(
        get_knn_scores(t_norm, interaction_tensor, inter_matrix_batch_zero, knn_k=10)
    )
    raw_scores = t_weight * raw_t_scores + (1 - t_weight) * raw_v_scores
    return raw_scores


def get_pc_output(tst_scores,dataset,model):
    """
    Calculates popularity compensation post-processing scores (reference [75] in paper).
    """
    out = {}
    for alpha in np.arange(0.1,1.6,0.1):
        for beta in np.arange(0.1,1.1,0.1):
            C = inv_degs_arr * (tst_scores * beta + 1 - beta)
            pc_scores = tst_scores + alpha * C * (
                torch.norm(tst_scores, dim=1, keepdim=True)
                / torch.norm(C, dim=1, keepdim=True)
            )
            res = get_results(pc_scores, k=20)
            out[(alpha, beta)] = res
    with open('%s/%s/outputs/%s_PC.pkl'%(DATASETS_PATH,dataset,model),'wb') as f:
                pickle.dump(out,f)
    return out

def run_edge(item_emb,lmbda,alpha,beta,tau,popularity,head):
    norm_orig = torch.norm(item_emb, dim=1) + 1e-12
    item_emb_unit = item_emb / norm_orig[:,None]

    if lmbda > 0:
        sim = torch.mm(item_emb_unit, item_emb_unit[head].t())
        sim = torch.softmax(sim / max(tau, 0.01), dim=1)

        item_emb_att = torch.mm(sim, item_emb_unit[head])
        item_emb_att = item_emb_att / (torch.norm(item_emb_att, dim=1) + 1e-12)[:,None]

        item_emb_adj = item_emb_unit + lmbda * item_emb_att
        item_emb_adj = item_emb_adj / (torch.norm(item_emb_adj, dim=1) + 1e-12)[:,None]

    else:
        item_emb_adj = item_emb_unit
        
    norm = ((popularity + 1) ** beta) * (norm_orig ** (1 - alpha))
    return item_emb_adj * norm[:,None]

def edge_full(model,dataset,combos,inter_matrix_batch_mask,data,popularity,long_tail,head,
              betas = [0.0,0.1,0.2],
        alphas = [0.0,0.2,0.4,0.6,0.8,1.0],
        taus = [0.05,0.1,0.2],
        lmbdas = [0.0, 0.2, 0.4,0.6, 0.8,1.0, 1.2,1.4, 1.6,1.8, 2.0]):
    
    model_user, model_item = torch.load(
        "%s/model_files/%s_ood_%s.pt" % (DATASETS_PATH, dataset, model)
    )
    
    combos = list(product(*(betas,alphas,taus,lmbdas)))
    for i,c in enumerate(combos):
            c = list(c)
            if c[-1] == 0:
                c[-2] = 0.05
            combos[i] = tuple(c)
    combos = sorted(set(combos))
    
    results = {}
    for i,(beta,alpha,tau,lmbda) in enumerate(combos):
        edge_item_emb = run_edge(model_item,lmbda,alpha,beta,tau,popularity,head)
        model_pred = model_user[[int(k) for k in data.user]] @ edge_item_emb.T
        scores = model_pred+inter_matrix_batch_mask
        res = get_results(scores,data,long_tail)
        results[(beta,alpha,tau,lmbda)] = res
        if (i+1)%20 == 0:
            print(model,i+1,"%.2f, %.2f, %.2f, %.2f" % (beta,alpha,tau,lmbda),res)
            with open('%s/%s/outputs/%s_EDGE.pkl'%(DATASETS_PATH,dataset,model),'wb') as f:
                pickle.dump(results,f)

    return results

def get_ensemble(
    m_scores, cb_scores, alphas
):
    """
    Calculates performance of ensemble for varying values of alpha
    """
    out_validation = {}
    out_test = {}
    for alpha in alphas:
        ensemble = alpha * cb_scores + m_scores
        out_test[alpha] = get_results(ensemble, test=True)
        out_validation[alpha] = get_results(ensemble, test=False)
    return out_test, out_validation


def get_item_popularity_info(data):
    item_deg_tups = [
        (i, len(data.training_set_i[i])) for i in sorted(data.training_set_i)
    ]
    item_degs = [p[1] for p in item_deg_tups]
    items_sorted_by_deg = sorted(item_deg_tups, key=lambda x: x[1], reverse=True)
    inv_degs_arr = torch.from_numpy(np.expand_dims(1 / np.array(item_degs), axis=0))

    short_head_size = int(len(items_sorted_by_deg) // 5)
    long_tail = set([p[0] for p in items_sorted_by_deg[short_head_size:]])
    return long_tail, inv_degs_arr


def run_ensemble(
    dataset
):
    global data, long_tail, inv_degs_arr

    item_content = torch.from_numpy(np.load("./%s/feats_concat.npy" % (dataset)))

    data = get_ood_data(dataset, item_content)

    interaction_tensor = sparse_mx_to_torch_sparse_tensor(
        data.interaction_mat[:, data.mapped_item_idx]
    )
    inter_matrix_input = data.interaction_mat[:, data.mapped_item_idx]
    inter_matrix_batch = sparse_mx_to_torch_tensor(inter_matrix_input)
    inter_matrix_batch_zero = 1 + 1e-10 - inter_matrix_batch
    inter_matrix_batch_mask = -1e10 * inter_matrix_batch


    long_tail, inv_degs_arr = get_item_popularity_info(data)

    # Load encoded item embeddings
    item_encoder_out = F.normalize(
        torch.from_numpy(
            np.load("%s/%s/outputs/Encoded_KNN.npy" % (DATASETS_PATH, dataset))
        )
    )
    encoded_scores = get_knn_scores(
        item_encoder_out, interaction_tensor, inter_matrix_batch_zero, knn_k=20
    )
    encoded_scores_norm = F.normalize(encoded_scores)

    # Raw scores are already normalized
    raw_scores = get_raw_scores(
        item_content, interaction_tensor, inter_matrix_batch_zero
    )
    cb_scores_dict = {"Raw-KNN": raw_scores, "Encoded-KNN": encoded_scores_norm}

    models = ["LightGCN", "XSimGCL", "FREEDOM", "MGCN", "SOIL", "TMLP", "GUME"]

    out_test = {k: {} for k in cb_scores_dict}
    out_validation = {k: {} for k in cb_scores_dict}
    for m in models:
        m_scores = F.normalize(get_warm_model_pred(dataset, inter_matrix_batch_zero, m))+inter_matrix_batch_mask
        for cb_model, cb_scores in cb_scores_dict.items():
            alphas = np.arange(0.02,0.32,0.02) if cb_model == 'Encoded-KNN' else np.arange(0.04,0.64,0.04)
            m_ensemble_test, m_ensemble_validation = get_ensemble(
                m_scores,
                cb_scores,
                alphas=alphas,
            )
            out_test[cb_model][m] = m_ensemble_test
            out_validation[cb_model][m] = m_ensemble_validation
            print(dataset, m, cb_model)

    pickle.dump(
        (out_test, out_validation),
        open(
            "%s/outputs/model_ensemble_scores_%s.pkl" % (DATASETS_PATH, dataset),
            "wb",
        ),
    )
