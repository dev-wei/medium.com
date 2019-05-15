import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score


class FM(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature_size,
        field_size,
        embedding_size=8,
        dropout_fm=(1.0, 1.0),
        epoch=10,
        batch_size=256,
        learning_rate=0.001,
        optimizer="adam",
        batch_norm=0,
        batch_norm_decay=0.995,
        verbose=False,
        random_seed=1234,
        use_fm=True,
        use_deep=True,
        loss_type="logloss",
        eval_metric=roc_auc_score,
        l2_reg=0.0,
        greater_is_better=True,
    ):
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        self.dropout_fm = dropout_fm
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better

        self.train_result, self.valid_result = [], []
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            # input
            self.feat_index = tf.placeholder(
                tf.int32, shape=[None, self.field_size], name="feat_index"
            )
            self.feat_value = tf.placeholder(
                tf.float32, shape=[None, self.field_size], name="feat_value"
            )
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")
            self.dropout_keep_fm = tf.placeholder(
                tf.float32, shape=[None], name="dropout_keep_fm"
            )  # ?

            # weights
            self.weights = self._initialize_weights()

            # model
            # find the mapped embedding from [feat_size * 8] based on feat_index
            self.embeddings = tf.nn.embedding_lookup(
                self.weights["feature_embeddings"], self.feat_index
            )  # N * F * K (None, 3, 8)
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)

            # first order term
            self.y_first_order = tf.nn.embedding_lookup(
                self.weights["feature_bias"], self.feat_index
            )
            self.y_first_order = tf.reduce_sum(
                tf.multiply(self.y_first_order, feat_value), axis=2
            )
            self.y_first_order = tf.nn.dropout(
                self.y_first_order, self.dropout_keep_fm[0]
            )

            # second order term
            # sum-square-part
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * k
            self.summed_features_emb_square = tf.square(
                self.summed_features_emb
            )  # None * K

            # square-sum-part
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(
                self.squared_features_emb, 1
            )  # None * K

            # second order
            self.y_second_order = 0.5 * tf.subtract(
                self.summed_features_emb_square, self.squared_sum_features_emb
            )
            self.y_second_order = tf.nn.dropout(
                self.y_second_order, self.dropout_keep_fm[1]
            )

            # out
            concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            self.out = tf.add(
                tf.matmul(concat_input, self.weights["concat_projection"]),
                self.weights["concat_bias"],
            )

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))




    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings",
        )  # (256, 8)
        weights["feature_bias"] = tf.Variable(
            tf.random_normal([self.feature_size, 1], 0.0, 1.0), name="feature_bias"
        )  # (256, 1)

        input_size = self.field_size + self.embedding_size  # 39 + 8

        return weights

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {
            self.feat_index: Xi,
            self.feat_value: Xv,
            self.label: y,
            self.dropout_keep_fm: self.dropout_fm,
            self.dropout_keep_deep: self.dropout_deep,
            self.train_phase: True,
        }

        loss, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)

        return loss

    def fit(
        self,
        Xi_train,
        Xv_train,
        y_train,
        Xi_valid=None,
        Xv_valid=None,
        y_valid=None,
        early_stopping=False,
        refit=False,
    ):
        has_valid = Xv_valid is not None

        for epoch in range(self.epoch):
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(
                    Xi_train, Xv_train, y_train, self.batch_size, i
                )
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)

        pass
