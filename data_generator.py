from sklearn.datasets import make_moons
from scipy.stats import norm
from sklearn.datasets import make_multilabel_classification
import numpy as np


def create_twomoon_dataset(n, input_dim, noise_scale=1):
    relevant, y = make_moons(n_samples=n, shuffle=True, noise=1, random_state=None)
    noise_vector = norm.rvs(loc=0, scale=noise_scale, size=[n, input_dim - 2])
    data = np.concatenate([relevant, noise_vector], axis=1)
    return data, y

def create_multimoon_dataset(n, input_dim, n_classes, noise_scale = 1):
    data = []
    labels = []
    for i in range(n_classes):
        # Generate moon data
        data_i, labels_i = make_moons(n_samples=int(n/n_classes), noise=0.05)
        print(labels_i)
        # Add noise
        noise_vector = norm.rvs(loc=0, scale=noise_scale, size=[int(n/n_classes), input_dim - 2])
        data_i = np.concatenate([data_i, noise_vector], axis=1)
        # Append to dataset and labels
        data.append(data_i)
        labels.append(labels_i)
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    print(labels)

    return data, labels


def create_multilabel_dataset(n, p, n_classes, n_labels, noise_scale=0.5):
    X, y = make_multilabel_classification(n_samples=n, n_features=p, n_classes=n_classes, n_labels=n_labels,
                                                     random_state=None)
    noise = np.random.normal(scale=noise_scale, size=(n, p - 2))
    # X = np.concatenate([X, noise], axis=1)

    # Convert multilabel y to single label y
    return X, y