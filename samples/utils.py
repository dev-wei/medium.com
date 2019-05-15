import pandas as pd


class FeatureDictionary:
    def __init__(self, df_train=None, df_test=None, numeric_cols=(), ignore_cols=()):
        assert df_train is not None, "dfTrain can not be null"
        assert df_test is not None, "dfTest can not be null"

        self.df_train = df_train
        self.df_test = df_test
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols

        self.feat_dict = None
        self.feat_dim = None
        self.gen_feat_dict()

    def gen_feat_dict(self):
        df = pd.concat([self.df_train, self.df_test])

        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                self.feat_dict[col] = tc
                tc += 1

            else:
                us = df[col].unique()
                print(us)
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)

        self.feat_dim = tc


class DataParser:
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, df, has_label=False):
        dfi = df.copy()
        dfv = df.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.0

        xi = dfi.values.tolist()
        xv = dfv.values.tolist()

        return xi, xv
