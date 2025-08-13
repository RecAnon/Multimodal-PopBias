import pandas as pd
import random
import os
import numpy as np

DATASETS_PATH = "./data"

def get_train_core(df, core_size):
    """
    Extract core of size core_size from interaction data to ensure
    all users and items have sufficient interactions
    """
    item_users = df.groupby("item")["user"].apply(list).to_dict()
    train_core = []
    remaining = []
    for i, i_users in item_users.items():
        random.shuffle(i_users)
        train_core += [(u, i) for u in i_users[:core_size]]
        remaining += [(u, i) for u in i_users[core_size:]]

    train_core_df = pd.DataFrame(train_core, columns=["user", "item"])
    train_core_user_items = train_core_df.groupby("user")["item"].apply(list).to_dict()

    remaining_df = pd.DataFrame(remaining, columns=["user", "item"])
    remaining_user_items = remaining_df.groupby("user")["item"].apply(list).to_dict()
    all_users = set(df["user"].values)

    final_core = []
    for u in all_users:
        if u in train_core_user_items:
            u_core_items = train_core_user_items[u]
            if len(u_core_items) < core_size:
                u_core_items += random.sample(
                    remaining_user_items[u], core_size - len(u_core_items)
                )
        else:
            u_core_items = random.sample(remaining_user_items[u], core_size)
        final_core += [(u, i) for i in u_core_items]
    final_core_set = set(final_core)
    final_remaining = set(remaining).difference(final_core_set)
    return pd.DataFrame(final_core_set, columns=["user", "item"]), pd.DataFrame(
        final_remaining, columns=["user", "item"]
    )


def get_final_splits(
    df,
    core_size=2,
    test_ood_size=0.1,
    test_id_size=0.1,
    val_ood_size=0.1,
    val_id_size=0.1,
):
    orig_df_size = df.shape[0]
    train_core, remaining = get_train_core(df, core_size)

    ood_total = test_ood_size + val_ood_size
    remaining_item_probs = (1 / remaining["item"].value_counts()).to_dict()
    remaining["item_prob"] = remaining["item"].apply(lambda x: remaining_item_probs[x])
    ood_df = remaining.sample(
        n=int(ood_total * orig_df_size), weights="item_prob"
    ).drop("item_prob", axis=1)
    id_df = remaining.drop(ood_df.index, axis=0).drop("item_prob", axis=1)

    ood_val = ood_df.sample(frac=val_ood_size / ood_total)
    ood_test = ood_df.drop(ood_val.index, axis=0)

    id_val = id_df.sample(n=int(val_id_size * orig_df_size))
    id_test = id_df.drop(id_val.index, axis=0).sample(
        n=int(test_id_size * orig_df_size)
    )
    id_train = id_df.drop(pd.concat([id_val, id_test]).index, axis=0)
    final_train = pd.concat([train_core, id_train])
    return final_train, ood_val, id_val, ood_test, id_test


def split(
    dataset,
    core_size=2,
    test_ood_size=0.1,
    test_id_size=0.1,
    val_ood_size=0.1,
    val_id_size=0.1,
):
    """
    Perform splitting and save output files per ID/OOD split.
    """

    df_full = pd.read_csv(
        "%s/%s/%s.inter" % (DATASETS_PATH, dataset, dataset), sep="\t"
    )
    df = df_full.iloc[:, :2]
    df.columns = ["user", "item"]
    final_train, ood_val, id_val, ood_test, id_test = get_final_splits(
        df, core_size, test_ood_size, test_id_size, val_ood_size, val_id_size
    )

    pth = "%s/%s/ood" % (DATASETS_PATH, dataset)
    if not os.path.exists(pth):
        os.makedirs(pth)
    final_train.to_csv(pth + "/warm_train.csv", index=False)
    ood_val.to_csv(pth + "/ood_val.csv", index=False)
    id_val.to_csv(pth + "/in_dist_val.csv", index=False)
    ood_test.to_csv(pth + "/ood_test.csv", index=False)
    id_test.to_csv(pth + "/in_dist_test.csv", index=False)
    pd.concat([ood_test, id_test]).to_csv(pth + "/overall_test.csv", index=False)
    pd.concat([ood_val, id_val]).to_csv(pth + "/overall_val.csv", index=False)


def create_mmrec_out(dataset):
    """
    Convert output files to MMRec format for MMRec model training
    """

    train = pd.read_csv("%s/%s/ood/warm_train.csv" % (DATASETS_PATH, dataset))
    train["x_label"] = 0
    val = pd.read_csv("%s/%s/ood/overall_val.csv" % (DATASETS_PATH, dataset))
    val["x_label"] = 1

    test = pd.read_csv("%s/%s/ood/overall_test.csv" % (DATASETS_PATH, dataset))
    test["x_label"] = 2
    warm = pd.concat([train, val, test])
    warm.columns = ["userID", "itemID", "x_label"]
    warm["timestamp"] = None
    warm["rating"] = None
    warm = warm[["userID", "itemID", "rating", "timestamp", "x_label"]]
    warm.to_csv(
        "%s/%s/ood/%s_ood.inter" % (DATASETS_PATH, dataset, dataset),
        sep="\t",
        index=False,
    )


def concat_feats(dataset):
    image_feats = np.load("%s/%s/image_feat.npy" % (DATASETS_PATH, dataset))
    text_feats = np.load("%s/%s/text_feat.npy" % (DATASETS_PATH, dataset))
    concat_feats = np.concatenate([image_feats, text_feats], axis=1)
    with open("%s/%s/feats_concat.npy" % (DATASETS_PATH, dataset), "wb") as f:
        np.save(f, concat_feats)


for dataset in ["baby", "clothing", "sports"]:
    concat_feats(dataset)
    split(dataset)
    create_mmrec_out(dataset)
