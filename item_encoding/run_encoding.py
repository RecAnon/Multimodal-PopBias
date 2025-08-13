import argparse
import torch
from utils.data_utils import DataLoader
from .encoder_model import Encoder
import yaml
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="clothing")

config = yaml.safe_load(open("./encoding_params.yaml", "r"))
args, _ = parser.parse_known_args()
for k, v in config.items():
    vars(args)[k] = v
device = torch.device("cuda")

# data loader
training_data = DataLoader.load_data_set(f"./data/{args.dataset}/ood/warm_train.csv")
all_valid_data = DataLoader.load_data_set(f"./data/{args.dataset}/ood/overall_val.csv")
in_dist_valid_data = DataLoader.load_data_set(
    f"./data/{args.dataset}/ood/in_dist_val.csv"
)
ood_valid_data = DataLoader.load_data_set(f"./data/{args.dataset}/ood/ood_val.csv")
all_test_data = DataLoader.load_data_set(f"./data/{args.dataset}/ood/overall_test.csv")
in_dist_test_data = DataLoader.load_data_set(
    f"./data/{args.dataset}/ood/in_dist_test.csv"
)
ood_test_data = DataLoader.load_data_set(f"./data/{args.dataset}/ood/ood_test.csv")

user_idx = sorted(set([p[0] for p in training_data]))
item_idx = sorted(set([p[1] for p in training_data]))
user_num = len(user_idx)
item_num = len(item_idx)

item_content = np.load("./data/%s/feats_concat.npy" % args.dataset).astype(
    np.float32
)

model = Encoder(
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
    item_content=item_content,
)
model.train()
