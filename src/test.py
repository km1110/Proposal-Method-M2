import numpy as np
from sklearn.metrics import accuracy_score


def test(model, x, y):
    x_test = np.array(x, np.float32) / 255.0
    y_test = y.reshape(-1)

    bs = 1000
    test_pred = []

    print(f"y_test: {y_test[0]}")

    for i in range(0, len(x_test), bs):
        batch_pred = model(x_test[i : i + bs])
        test_pred.append(batch_pred)
    test_pred = np.concatenate(test_pred, axis=0)

    test_pred = np.argmax(test_pred, axis=1)
    print(f"y_pred: {test_pred[0]}")

    acc = accuracy_score(y_test, test_pred)
    print(f"Accuracy: {acc}")
