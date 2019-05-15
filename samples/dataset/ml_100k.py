# %%

import pandas as pd


def get_data(file_set="a"):
    df_users = pd.read_csv(
        "./workspace/data/ml-100k/u.user",
        delimiter="|",
        header=None,
        encoding="latin-1",
        names=["user_id", "age", "gender", "occupation", "zip"],
    )

    genres = [
        "unknown",
        "action",
        "adventure",
        "animation",
        "children",
        "comedy",
        "crime",
        "documentary",
        "drama",
        "fantasy",
        "film-noir",
        "horror",
        "musical",
        "mystery",
        "romance",
        "sci-fi",
        "thriller",
        "war",
        "western",
    ]
    df_items = pd.read_csv(
        "./workspace/data/ml-100k/u.item",
        delimiter="|",
        header=None,
        encoding="latin-1",
        names=["item_id", "title", "release_date", "video_release_date", "imdb_url"]
        + genres,
    )

    def concat(x):
        str_genres = []
        for f in genres:
            if x[f] == 1.0:
                str_genres.append(f)

        return "|".join(str_genres)

    df_items.loc[:, "genres"] = df_items.apply(concat, axis=1)

    df_train_interactions = pd.read_csv(
        f"./workspace/data/ml-100k/u{file_set}.base",
        delimiter="\t",
        header=None,
        encoding="latin-1",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    df_test_interactions = pd.read_csv(
        f"./workspace/data/ml-100k/u{file_set}.test",
        delimiter="\t",
        header=None,
        encoding="latin-1",
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    df_train_all = pd.merge(
        df_train_interactions, df_users, left_on="user_id", right_on="user_id"
    )
    df_train_all = pd.merge(
        df_train_all,
        df_items[["item_id", "title", "release_date", "genres"]],
        left_on="item_id",
        right_on="item_id",
    )

    df_test_all = pd.merge(
        df_test_interactions, df_users, left_on="user_id", right_on="user_id"
    )
    df_test_all = pd.merge(
        df_test_all,
        df_items[["item_id", "title", "release_date", "genres"]],
        left_on="item_id",
        right_on="item_id",
    )

    return df_train_all, df_test_all
