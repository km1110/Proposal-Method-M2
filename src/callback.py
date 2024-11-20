import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.datasets import cifar10
from keras.metrics import categorical_accuracy


class PseudoCallback(Callback):
    def __init__(self, model, args):
        self.n_labeled_sample = args.labeled_num
        self.batch_size = args.batch_size
        self.model = model
        self.n_classes = 10

        # labeled_unlabeledの作成
        (X_train, y_train), (self.X_test, self.y_test) = cifar10.load_data()
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        self.X_train_labeled = X_train[indices[: args.labeled_num]]
        self.y_train_labeled = y_train[indices[: args.labeled_num]]
        self.X_train_unlabeled = X_train[indices[args.labeled_num :]]
        self.y_train_unlabeled_groundtruth = y_train[indices[args.labeled_num :]]

        # unlabeledの予測値
        self.y_train_unlabeled_prediction = np.random.randint(
            10, size=(self.y_train_unlabeled_groundtruth.shape[0], 1)
        )

        # steps_per_epoch
        self.train_steps_per_epoch = X_train.shape[0] // args.batch_size
        self.test_stepes_per_epoch = self.X_test.shape[0] // args.batch_size

        # unlabeledの重み
        self.alpha_t = 0.0
        self.delta1_init = args.delta1_init
        self.delta1 = args.delta1
        self.threshold1 = args.threshold1
        self.threshold2 = args.threshold2

        self.class_ratios = []

        # labeled/unlabeledの一致率推移
        self.unlabeled_accuracy = []
        self.labeled_accuracy = []

    def train_mixture(self):
        # 返り値：X, y, フラグ
        X_train_join = np.r_[self.X_train_labeled, self.X_train_unlabeled]
        y_train_join = np.r_[self.y_train_labeled, self.y_train_unlabeled_prediction]
        flag_join = np.r_[
            np.repeat(0.0, self.X_train_labeled.shape[0]),
            np.repeat(1.0, self.X_train_unlabeled.shape[0]),
        ].reshape(-1, 1)
        indices = np.arange(flag_join.shape[0])
        np.random.shuffle(indices)
        return X_train_join[indices], y_train_join[indices], flag_join[indices]

    def train_generator(self):
        while True:
            X, y, flag = self.train_mixture()
            n_batch = X.shape[0] // self.batch_size
            for i in range(n_batch):
                X_batch = (
                    X[i * self.batch_size : (i + 1) * self.batch_size] / 255.0
                ).astype(np.float32)
                y_batch = to_categorical(
                    y[i * self.batch_size : (i + 1) * self.batch_size], self.n_classes
                )
                y_batch = np.c_[
                    y_batch, flag[i * self.batch_size : (i + 1) * self.batch_size]
                ]
                yield X_batch, y_batch

    def test_generator(self):
        while True:
            indices = np.arange(self.y_test.shape[0])
            np.random.shuffle(indices)
            for i in range(len(indices) // self.batch_size):
                current_indices = indices[
                    i * self.batch_size : (i + 1) * self.batch_size
                ]
                X_batch = (self.X_test[current_indices] / 255.0).astype(np.float32)
                y_batch = to_categorical(self.y_test[current_indices], self.n_classes)
                y_batch = np.c_[
                    y_batch, np.repeat(0.0, y_batch.shape[0])
                ]  # flagは0とする
                yield X_batch, y_batch

    def calc_prior_loss(self, y_true, y_pred):
        prior = [0.1] * 10

        unlabel_index = tf.cast(y_true[:, -1] == 1.0, tf.float32)

        unlabel_pred = tf.boolean_mask(y_pred, unlabel_index)

        posterior_prob_sum = tf.reduce_sum(unlabel_pred)
        posterior_prob = tf.reduce_sum(unlabel_pred, axis=0) / posterior_prob_sum

        diff = tf.abs(prior - posterior_prob)
        penalty = tf.reduce_sum(-tf.math.log(1 - diff + 1e-12)) / tf.cast(
            10, dtype=tf.float32
        )

        return penalty

    def loss_function(self, y_true, y_pred):
        # penalty = self.calc_prior_loss(y_true, y_pred)
        y_true_item = y_true[:, : self.n_classes]
        unlabeled_flag = y_true[:, self.n_classes]
        entropies = keras.losses.categorical_crossentropy(y_true_item, y_pred)
        coefs = (
            1.0 - unlabeled_flag + self.alpha_t * unlabeled_flag
        )  # 1 if labeled, else alpha_t
        return coefs * entropies

    def calc_class_ratios(self, prediction):
        pred = np.argmax(prediction, axis=1)
        class_counts = np.bincount(pred, minlength=10)
        total_counts = len(pred)
        class_ratio = class_counts / total_counts
        self.class_ratios.append(class_ratio)

    def accuracy(self, y_true, y_pred):
        y_true_item = y_true[:, : self.n_classes]
        return categorical_accuracy(y_true_item, y_pred)

    def on_epoch_end(self, epoch, logs):
        # alpha(t)の更新
        if epoch < self.threshold1:
            self.alpha_t = self.delta1_init
        elif epoch >= self.threshold2:
            self.alpha_t = self.delta1
        else:
            self.alpha_t = (
                (epoch - self.threshold1)
                / (self.threshold2 - self.threshold1)
                * self.delta1
            )

        # unlabeled のラベルの更新
        pred = self.model.predict(self.X_train_unlabeled)
        self.y_train_unlabeled_prediction = np.argmax(
            pred,
            axis=-1,
        ).reshape(-1, 1)
        y_train_labeled_prediction = np.argmax(
            self.model.predict(self.X_train_labeled), axis=-1
        ).reshape(-1, 1)

        if epoch in [19, 69, 99]:
            self.calc_class_ratios(pred)

        # ground-truthとの一致率
        self.unlabeled_accuracy.append(
            np.mean(
                self.y_train_unlabeled_groundtruth == self.y_train_unlabeled_prediction
            )
        )
        self.labeled_accuracy.append(
            np.mean(self.y_train_labeled == y_train_labeled_prediction)
        )
        print(
            "labeled / unlabeled accuracy : ",
            self.labeled_accuracy[-1],
            "/",
            self.unlabeled_accuracy[-1],
        )

    # def on_train_end(self, logs):
    #     y_true = np.ravel(self.y_test)
    #     emb_model = Model(self.model.input, self.model.layers[-2].output)
    #     embedding = emb_model.predict(self.X_test / 255.0)
    #     proj = TSNE(n_components=2).fit_transform(embedding)
    #     cmp = plt.get_cmap("tab10")
    #     plt.figure()
    #     for i in range(10):
    #         select_flag = y_true == i
    #         plt_latent = proj[select_flag, :]
    #         plt.scatter(plt_latent[:, 0], plt_latent[:, 1], color=cmp(i), marker=".")
    #     plt.savefig(f"result_pseudo/embedding_{self.n_labeled_sample:05}.png")
