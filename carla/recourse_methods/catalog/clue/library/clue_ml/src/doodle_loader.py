# flake8: noqa
from sklearn.model_selection import train_test_split
from src.utils import *
from torchvision import datasets, transforms


def load_doodle_dset(
    class_names, doodle_folder="/homes/ja666/quickdraw/numpy_64/", test_ratio=0.2
):
    """Load my custom centered and prepocessed doodle dataset at 64x64 pixels of resoluion"""
    X = []
    y = []

    for i in range(len(class_names)):
        class_data = np.load(doodle_folder + class_names[i] + ".npy")
        X.append(class_data)
        y.append(np.ones(class_data.shape[0]) * i)

    X = np.concatenate(X, axis=0).astype(np.float32)
    y = np.concatenate(y, axis=0).astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, shuffle=True
    )

    #     X_train = X_train.reshape(X_train.shape[0], 28, 28)
    #     X_test = X_test.reshape(X_test.shape[0], 28, 28)

    trainset = Datafeed(X_train, y_train, transform=transforms.ToTensor())
    valset = Datafeed(X_test, y_test, transform=transforms.ToTensor())
    return trainset, valset
