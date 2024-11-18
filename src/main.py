import os
import warnings

import argment

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import tensorflow as tf


# from src.train import train
from train import train
from test import test


def set_gpus():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(
        [gpus[0]],  # , gpus[1], gpus[2]
        "GPU",
    )
    strategy = tf.distribute.MirroredStrategy()

    return strategy


if __name__ == "__main__":
    args = argment.get_arguments()
    strategy = set_gpus()
    with strategy.scope():
        model, x_test, y_test = train(args)
        test(model, x_test, y_test)
    # n_batches = [500, 1000, 5000, 10000]
    # for nb in n_batches:
    #     print(nb, "Starts")
    #     train(nb)
