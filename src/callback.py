import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.datasets import cifar10
from keras.metrics import categorical_accuracy

from utils import apply_weak_augmentation


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
        self.X_train_unlabeled_weak_augmentation = apply_weak_augmentation(self.X_train_unlabeled)
        self.y_train_unlabeled_weak_augmentation = self.y_train_unlabeled_groundtruth
        

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
        X_train_join = np.r_[self.X_train_labeled, self.X_train_unlabeled]
        y_train_join = np.r_[self.y_train_labeled, self.y_train_unlabeled_prediction]
        
        X_train_join_weak_augmentation = np.r_[self.X_train_labeled, self.X_train_unlabeled_weak_augmentation]
        y_train_join_weak_augmentation = np.r_[self.y_train_labeled, self.y_train_unlabeled_prediction]
        
        flag_join = np.r_[
            np.repeat(0.0, self.X_train_labeled.shape[0]),
            np.repeat(1.0, self.X_train_unlabeled.shape[0]),
        ].reshape(-1, 1)
        
        flag_join_weak_augmentation = np.r_[
            np.repeat(0.0, self.X_train_labeled.shape[0]),
            np.repeat(2.0, self.X_train_unlabeled.shape[0]),
        ].reshape(-1, 1)
        
        indices = np.arange(flag_join.shape[0])
        np.random.shuffle(indices)
        
        return X_train_join[indices], y_train_join[indices], X_train_join_weak_augmentation[indices], y_train_join_weak_augmentation[indices], flag_join[indices], flag_join_weak_augmentation[indices]
        
    
    def genarate_batch_size_data(self, X, y, X_weak_aug, y_weak_aug, flag, flag_weak_aug, index):
        x_batch = X[index * self.batch_size : (index + 1) * self.batch_size]
        y_batch = to_categorical(y[index * self.batch_size : (index + 1) * self.batch_size], self.n_classes)
        y_batch_add_flag = np.c_[y_batch, flag[index * self.batch_size : (index + 1) * self.batch_size]]
        
        x_batch_weak_aug = X_weak_aug[index * self.batch_size : (index + 1) * self.batch_size]
        y_batch_weak_aug = to_categorical(y_weak_aug[index * self.batch_size : (index + 1) * self.batch_size], self.n_classes)
        y_batch_add_flag_weak_aug = np.c_[y_batch_weak_aug, flag_weak_aug[index * self.batch_size : (index + 1) * self.batch_size]]
        
        flag =  y_batch_add_flag_weak_aug[:, self.n_classes]
        mask = (flag == 2)
        x_batch_unlabel_weak_aug = x_batch_weak_aug[mask]
        y_batch_unlabel_weak_aug = y_batch_add_flag_weak_aug[mask]
        
        X_batch = np.r_[x_batch, x_batch_unlabel_weak_aug]
        y_batch = np.r_[y_batch_add_flag, y_batch_unlabel_weak_aug]
        
        X_batch = (X_batch / 255.0).astype(np.float32)
        
        return X_batch, y_batch

    def train_generator(self):
        while True:
            X, y, X_weak_aug, y_weak_aug, flag, flag_weak_aug = self.train_mixture()
            n_batch = X.shape[0] // self.batch_size
            for i in range(n_batch):
                X_batch, y_batch = self.genarate_batch_size_data(X, y, X_weak_aug, y_weak_aug, flag, flag_weak_aug, i)
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
        
        flag = y_true[:, self.n_classes]
        labeled_mask = (flag == 0)
        unlabel_mask = (flag == 1)
        augment_mask = (flag == 2)
        
        y_true_labeled = y_true_item[labeled_mask]
        y_pred_labeled = y_pred[labeled_mask]
        y_true_unlabel = y_true_item[unlabel_mask]
        y_pred_unlabel = y_pred[unlabel_mask]
        y_pred_augment = y_pred[augment_mask]
        
        l_s = keras.losses.categorical_crossentropy(y_true_labeled, y_pred_labeled)
        l_u = keras.losses.categorical_crossentropy(y_true_unlabel, y_pred_unlabel)
        
        l_r = tf.math.log(( y_pred_unlabel / y_pred_augment ) + 1e-8)
        
        l_u = self.alpha_t * l_u  + l_r
        
        if self.alpha_t == 0:
            print("Supervised Loss!")
        
        return l_s + l_u if self.alpha_t != 0 else l_s

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