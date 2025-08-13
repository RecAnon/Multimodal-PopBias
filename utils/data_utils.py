import os.path
from os import remove
from re import split
import numpy as np
from collections import defaultdict
import scipy.sparse as sp

class DataLoader(object):
    def __init__(self):
        pass

    @staticmethod
    def write_file(dir, file, content, op="w"):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir + file, op) as f:
            f.writelines(content)

    @staticmethod
    def delete_file(file_path):
        if os.path.exists(file_path):
            remove(file_path)

    @staticmethod
    def load_data_set(file):
        data = []
        with open(file) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                items = split(",", line.strip())
                user_id = items[0]
                item_id = items[1]
                # weight = items[2]
                data.append([int(user_id), int(item_id), 1.0])
        return data

    @staticmethod
    def load_user_list(file):
        user_list = []
        print("loading user List...")
        with open(file) as f:
            for line in f:
                user_list.append(line.strip().split()[0])
        return user_list

    @staticmethod
    def load_social_data(file):
        social_data = []
        print("loading social data...")
        with open(file) as f:
            for line in f:
                items = split(" ", line.strip())
                user1 = items[0]
                user2 = items[1]
                if len(items) < 3:
                    weight = 1
                else:
                    weight = float(items[2])
                social_data.append([user1, user2, weight])
        return social_data


class OODDataBuilder(object):
    def __init__(
        self,
        training_data,
        in_dist_valid_data,
        ood_valid_data,
        overall_valid_data,
        in_dist_test_data,
        ood_test_data,
        overall_test_data,
        user_num,
        item_num,
        user_idx,
        item_idx,
        user_content=None,
        item_content=None,
    ):
        super(OODDataBuilder, self).__init__()
        self.training_data = training_data
        self.in_dist_valid_data = in_dist_valid_data
        self.in_dist_test_data = in_dist_test_data
        self.ood_valid_data = ood_valid_data
        self.ood_test_data = ood_test_data
        self.overall_valid_data = overall_valid_data
        self.overall_test_data = overall_test_data

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.in_dist_valid_set = defaultdict(dict)
        self.in_dist_valid_set_item = set()
        self.ood_valid_set = defaultdict(dict)
        self.ood_valid_set_item = set()
        self.overall_valid_set = defaultdict(dict)
        self.overall_valid_set_item = set()
        self.in_dist_test_set = defaultdict(dict)
        self.in_dist_test_set_item = set()
        self.ood_test_set = defaultdict(dict)
        self.ood_test_set_item = set()
        self.overall_test_set = defaultdict(dict)
        self.overall_test_set_item = set()
        self.source_user_content = None
        self.mapped_user_content = None
        self.source_item_content = None
        self.mapped_item_content = None
        if user_content is not None:
            self.source_user_content = user_content
            self.mapped_user_content = np.empty(
                (user_content.shape[0], user_content.shape[1])
            )
            self.user_content_dim = user_content.shape[-1]
        if item_content is not None:
            self.source_item_content = item_content
            self.mapped_item_content = np.empty(
                (item_content.shape[0], item_content.shape[1]), dtype=np.float32
            )
            self.item_content_dim = item_content.shape[-1]

        self.generate_set()

        self.user_num = user_num
        self.item_num = item_num
        # print(self.item_num, len(self.item.keys()))
        # raise Exception("debugging...")
        # PLEASE NOTE: the original and mapped index are different!
        self.source_user_idx = user_idx
        self.source_item_idx = item_idx
        self.mapped_user_idx = self.get_user_id_list(self.source_user_idx)
        self.mapped_item_idx = self.get_item_id_list(self.source_item_idx)
        # raise Exception("debugging...")
        self.ui_adj = self.create_sparse_complete_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.create_sparse_interaction_matrix()

    def generate_set(self):
        # training set building
        for entry in self.training_data:
            user, item, rating = entry
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = (
                        self.source_user_content[user]
                    )
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = (
                        self.source_item_content[item]
                    )
                # userList.append
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating

        # in_dist validation set building
        for entry in self.in_dist_valid_data:
            user, item, rating = entry
            if user not in self.user:
                # print(f"user {user} not in current id table (in_dist validation set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = (
                        self.source_user_content[user]
                    )
            if item not in self.item:
                # print(f"item {item} not in current id table (in_dist validation set)")
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = (
                        self.source_item_content[item]
                    )
            self.in_dist_valid_set[user][item] = rating
            self.in_dist_valid_set_item.add(item)

        # in_dist testing set building
        for entry in self.in_dist_test_data:
            user, item, rating = entry
            if user not in self.user:
                # print(f"user {user} not in current id table (in_dist test set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = (
                        self.source_user_content[user]
                    )
            if item not in self.item:
                # print(f"item {item} not in current id table (in_dist test set)")
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = (
                        self.source_item_content[item]
                    )
            self.in_dist_test_set[user][item] = rating
            self.in_dist_test_set_item.add(item)

        for entry in self.ood_valid_data:
            user, item, rating = entry
            if user not in self.user:
                # print(f"user {user} not in current id table (ood validation set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = (
                        self.source_user_content[user]
                    )
            if item not in self.item:
                # print(f"item {item} not in current id table (ood valid set)")
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = (
                        self.source_item_content[item]
                    )
            self.ood_valid_set[user][item] = rating
            self.ood_valid_set_item.add(item)

        for entry in self.ood_test_data:
            user, item, rating = entry
            if user not in self.user:
                # print(f"user {user} not in current id table (ood test set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = (
                        self.source_user_content[user]
                    )
            if item not in self.item:
                # print(f"item {item} not in current id table (ood test set)")
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = (
                        self.source_item_content[item]
                    )
            self.ood_test_set[user][item] = rating
            self.ood_test_set_item.add(item)

        for entry in self.overall_valid_data:
            user, item, rating = entry
            if user not in self.user:
                # print(f"user {user} not in current id table (overall valid set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = (
                        self.source_user_content[user]
                    )
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = (
                        self.source_item_content[item]
                    )
            self.overall_valid_set[user][item] = rating
            self.overall_valid_set_item.add(item)

        for entry in self.overall_test_data:
            user, item, rating = entry
            if user not in self.user:
                # print(f"user {user} not in current id table (overall test set)")
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
                if self.source_user_content is not None:
                    self.mapped_user_content[self.user[user]] = (
                        self.source_user_content[user]
                    )
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                if self.source_item_content is not None:
                    self.mapped_item_content[self.item[item]] = (
                        self.source_item_content[item]
                    )
            self.overall_test_set[user][item] = rating
            self.overall_test_set_item.add(item)

        # raise Exception("now debugging...")

    def create_sparse_complete_bipartite_adjacency(self, self_connection=False):
        """
        return a sparse adjacency matrix with the shape (|u| + |i|, |u| + |i|)
        """
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix(
            (ratings, (user_np, item_np + self.user_num)),
            shape=(n_nodes, n_nodes),
            dtype=np.float32,
        )
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def normalize_graph_mat(self, adj_mat):
        """
        :param adj_mat: the sparse adjacency matrix
        :return: normalized adjacency matrix
        """
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0] + adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix(
            (ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),
            shape=(n_nodes, n_nodes),
            dtype=np.float32,
        )
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def create_sparse_interaction_matrix(self):
        """
        return a sparse adjacency matrix with the shape (user number, item number)
        """
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix(
            (entries, (row, col)),
            shape=(self.user_num, self.item_num),
            dtype=np.float32,
        )
        return interaction_mat

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]
        else:
            raise Exception(f"user {u} not in current id table")

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]
        else:
            raise Exception(f"item {i} not in current id table")

    def get_user_id_list(self, u_list):
        mapped_list = []
        for u in u_list:
            if u in self.user:
                mapped_list.append(self.user[u])
            else:
                raise Exception(f"user {u} not in current id table")
        return np.array(mapped_list)

    def get_item_id_list(self, i_list):
        mapped_list = []
        for i in i_list:
            if i in self.item:
                mapped_list.append(self.item[i])
            else:
                raise Exception(f"item {i} not in current id table")
        return np.array(mapped_list)

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def in_dist_valid_size(self):
        return (
            len(self.in_dist_valid_set),
            len(self.in_dist_valid_set_item),
            len(self.in_dist_valid_data),
        )

    def in_dist_test_size(self):
        return (
            len(self.in_dist_test_set),
            len(self.in_dist_test_set_item),
            len(self.in_dist_test_data),
        )

    def ood_valid_size(self):
        return (
            len(self.ood_valid_set),
            len(self.ood_valid_set_item),
            len(self.ood_valid_data),
        )

    def ood_test_size(self):
        return (
            len(self.ood_test_set),
            len(self.ood_test_set_item),
            len(self.ood_test_data),
        )

    def overall_valid_size(self):
        return (
            len(self.overall_valid_set),
            len(self.overall_valid_set_item),
            len(self.overall_valid_data),
        )

    def overall_test_size(self):
        return (
            len(self.overall_test_set),
            len(self.overall_test_set_item),
            len(self.overall_test_data),
        )

    def contain(self, u, i):
        "whether user u rated item i"
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):
        "whether user is in training set"
        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):
        """whether item is in training set"""
        if i in self.item:
            return True
        else:
            return False

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(
            self.training_set_u[u].values()
        )

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(
            self.training_set_i[i].values()
        )

    def row(self, u):
        u = self.id2user[u]
        k, v = self.user_rated(u)
        vec = np.zeros(len(self.item))
        # print vec
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        i = self.id2item[i]
        k, v = self.item_rated(i)
        vec = np.zeros(len(self.user))
        # print vec
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.user_rated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m


DATASETS_PATH = "./data"


def get_ood_data(dataset, item_content):
    training_data = DataLoader.load_data_set(
        f"{DATASETS_PATH}/{dataset}/ood/warm_train.csv"
    )
    all_valid_data = DataLoader.load_data_set(
        f"{DATASETS_PATH}/{dataset}/ood/overall_val.csv"
    )
    in_dist_valid_data = DataLoader.load_data_set(
        f"{DATASETS_PATH}/{dataset}/ood/in_dist_val.csv"
    )
    ood_valid_data = DataLoader.load_data_set(
        f"{DATASETS_PATH}/{dataset}/ood/ood_val.csv"
    )
    all_test_data = DataLoader.load_data_set(
        f"{DATASETS_PATH}/{dataset}/ood/overall_test.csv"
    )
    in_dist_test_data = DataLoader.load_data_set(
        f"{DATASETS_PATH}/{dataset}/ood/in_dist_test.csv"
    )
    ood_test_data = DataLoader.load_data_set(
        f"{DATASETS_PATH}/{dataset}/ood/ood_test.csv"
    )
    # dataset information
    user_idx = sorted(set([p[0] for p in training_data]))
    item_idx = sorted(set([p[1] for p in training_data]))
    user_num = len(user_idx)
    item_num = len(item_idx)
    data = OODDataBuilder(
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
        None,
        item_content,
    )
    return data
