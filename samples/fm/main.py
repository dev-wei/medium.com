from workspace.medium_com.samples.dataset import ml_100k
from workspace.medium_com.samples.utils import FeatureDictionary, DataParser
from workspace.medium_com.samples.fm.fm import FM
from sklearn.model_selection import StratifiedKFold
import numpy as np


def load_data():
    df_train, df_test = ml_100k.get_data("a")

    included_cols = ["user_id", "item_id", "timestamp"]
    target_col = ["rating"]

    X_train = df_train[included_cols].values
    y_train = df_train[target_col].values

    X_test = df_test[included_cols].values
    y_test = df_test[target_col].values

    return df_train, df_test, X_train, y_train, X_test, y_test


NUM_SPLITS = 5
RANDOM_SEED = 1234

dfm_params = {
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "epoch": 30,
    "batch_size": 512,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "random_seed": RANDOM_SEED,
}

df_train, df_test, X_train, y_train, X_test, y_test = load_data()

folds = list(
    StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=RANDOM_SEED).split(
        X_train, y_train
    )
)

fd = FeatureDictionary(
    df_train=df_train,
    df_test=df_test,
    numeric_cols=["timestamp"],
    ignore_cols=[
        "title",
        "release_date",
        "genres",
        "age",
        "gender",
        "occupation",
        "zip",
        "rating",
    ],
)

data_parser = DataParser(feat_dict=fd)

Xi_train, Xv_train = data_parser.parse(df=df_train)
Xi_test, Xv_test = data_parser.parse(df=df_test)

print(df_train.dtypes)

dfm_params["feature_size"] = fd.feat_dim
dfm_params["field_size"] = len(Xi_train[0])

y_train_meta = np.zeros((df_train.shape[0], 1), dtype=float)
y_test_meta = np.zeros((df_train.shape[0], 1), dtype=float)


def get(x, l):
    return [x[i] for i in l]


for i, (train_idx, valid_idx) in enumerate(folds):
    Xi_train_, Xv_train_, y_train_ = (
        get(Xi_train, train_idx),
        get(Xv_train, train_idx),
        get(y_train, train_idx),
    )
    Xi_valid_, Xv_valid_, y_valid_ = (
        get(Xi_train, valid_idx),
        get(Xv_train, valid_idx),
        get(y_train, valid_idx),
    )

    dfm = FM(**dfm_params)
