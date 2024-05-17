import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def load_and_prepare_data(train_path="data/basic_dataset.csv", split_size=0.1):
    """
    loads train_path and does some data prep for a xgboost model
    returns object of type DMatrix for XGB
    :param train_path: path to load
    :param split_size: test split size, so 0.1 means validation size of 10%
    :return: dtrain, dval
    """
    # load training dataset
    chunk = pd.read_csv(train_path, chunksize=1000000)
    df = pd.concat(chunk)

    # split data based on search ids
    train_groups, val_groups = train_test_split(df[['srch_id']].drop_duplicates(), test_size=split_size,
                                                random_state=420)

    # split the actual data on the search ids
    train_data = df[df['srch_id'].isin(train_groups['srch_id'])]
    val_data = df[df['srch_id'].isin(val_groups['srch_id'])]

    # prepare X and y for train/val set
    X_train = train_data.drop(columns=['click_bool', 'booking_bool', 'relevance', 'position', 'gross_bookings_usd'])
    y_train = train_data['relevance']

    X_val = val_data.drop(columns=['click_bool', 'booking_bool', 'relevance', 'position', 'gross_bookings_usd'])
    y_val = val_data['relevance']

    # create groups (for DMatrix)
    train_group = train_data.groupby('srch_id').size().to_list()
    val_group = val_data.groupby('srch_id').size().to_list()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(train_group)

    dval = xgb.DMatrix(X_val, label=y_val)
    dval.set_group(val_group)

    return dtrain, dval


def get_default_params():
    params = {
        'objective': 'rank:pairwise',
        'eta': 0.1,  # learning rate
        'gamma': 0.5,  # minimum loss reduction required to further partition on a leaf node
        'min_child_weight': 0.1,  # larger min_child_weight: more conservative
        'max_depth': 10,  # max depth of the tree
        'eval_metric': 'ndcg'  # 'map' for mean average precision; 'ndcg' for ndcg
    }
    return params


def train():
    pass
