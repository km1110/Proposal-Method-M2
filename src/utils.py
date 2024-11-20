import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

np.random.seed(seed=0)


# データ拡張の定義
def weak_augmentation(image):
    """
    弱いデータ拡張を適用する関数。
    - ランダムな水平反転
    - 明るさの調整
    - 彩度の調整
    """
    image = tf.image.random_flip_left_right(image)  # 水平反転
    image = tf.image.random_brightness(image, max_delta=0.2)  # 明るさ
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)  # 彩度
    return image


# データ拡張をバッチ全体に適用する関数
def apply_weak_augmentation(X):
    """
    データセット全体に弱いデータ拡張を適用。
    """
    X_augmented = tf.map_fn(weak_augmentation, X, fn_output_signature=tf.float32)
    return X_augmented


def plot_confusion_matrix(y_true, y_pred, save_dir, save_name, labels):
    # 混同行列を作成
    cm = confusion_matrix(y_true, y_pred)

    # 結果を10x10の表として表示 (Predicted labelを縦軸に)
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    # カラーバーを追加
    fig.colorbar(im, ax=ax, shrink=0.81)

    ax.set(
        xticks=np.arange(cm.shape[0]),
        yticks=np.arange(cm.shape[1]),
        # xticklabels=classes, yticklabels=classes,  # ラベル設定を削除
        title="Confusion Matrix",
        ylabel="Predicted label",
        xlabel="True label",
    )

    # 軸目盛りラベルを設定し、縦書きにする
    plt.xticks(np.arange(cm.shape[0]), labels, rotation=90)  # x軸ラベルを縦書きに
    plt.yticks(np.arange(cm.shape[1]), labels)  # y軸ラベルはそのまま

    # 各セルに数値を表示
    for i in range(cm.shape[1]):
        for j in range(cm.shape[0]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2.0 else "black",
            )

    # 画像を保存
    save_dir = os.path.join(save_dir, "confusion_matrix")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_name += ".png"
    save_file = os.path.join(save_dir, save_name)
    plt.savefig(save_file)
    plt.close()


def plot_epoch_per_pseudo_label(epochs, data, save_dir, save_name, labels):
    # グラフのプロット
    plt.figure(figsize=(12, 8))

    i = 0
    for epoch, epoch_data in zip(epochs, data):
        plt.plot(
            range(len(epoch_data)),
            epoch_data,
            marker="o",
            linewidth=1 + i,
            label=f"Epoch {epoch}",
        )
        i += 1

    # x軸の目盛りとラベルを設定
    plt.xticks(range(len(labels)), labels, rotation=45)  # ラベルを45度回転させて表示

    plt.xlabel("Class Number")
    plt.ylabel("Prediction Ratio")
    plt.title("Prediction Ratio per Class for Each Epoch")
    plt.legend()
    plt.grid(True)

    # 画像を保存
    save_dir = os.path.join(save_dir, "epoch_per_pseudo_label")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name_png = save_name + ".png"
    save_file = os.path.join(save_dir, save_name_png)
    plt.savefig(save_file)
    plt.close()

    save_data = os.path.join(save_dir, save_name)
    with open(save_data, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data)
