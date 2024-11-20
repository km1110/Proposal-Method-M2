import os

from model import create_cnn
from callback import PseudoCallback


def train(args):
    model = create_cnn()

    pseudo = PseudoCallback(model, args)
    model.compile("adam", loss=pseudo.loss_function, metrics=[pseudo.accuracy])

    if not os.path.exists("result_pseudo"):
        os.mkdir("result_pseudo")

    hist = model.fit_generator(
        pseudo.train_generator(),
        steps_per_epoch=pseudo.train_steps_per_epoch,
        callbacks=[pseudo],
        epochs=args.epochs,
    ).history
    hist["labeled_accuracy"] = pseudo.labeled_accuracy
    hist["unlabeled_accuracy"] = pseudo.unlabeled_accuracy

    return model, pseudo.X_test, pseudo.y_test
