import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

def mse_between_vectors(vec1, vec2):
    mse_sum = 0
    for i in range(len(vec1)):
        for j in range(len(vec2)):
            mse_sum += mean_squared_error(vec1[i], vec2[j])
    return mse_sum / (len(vec1) * len(vec2))


def mean_squared_distance(x):
    n_samples, n_dims = x.shape
    distances = np.zeros((n_samples, n_dims, n_dims))
    for i in range(n_samples):
        for j in range(n_dims):
            for k in range(j+1, n_dims):
                distances[i, j, k] = (x[i, j] - x[i, k])**2
    mean_distance = np.mean(distances)
    return mean_distance


def get_label_mse(X_transformed, y):
    unique_labels = np.unique(y)
    label_mses = []
    label_combination_mses = []

    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        label_data = X_transformed[label_indices]
        label_mse = mse_between_vectors(label_data,label_data)
        label_mses.append(label_mse)

        for other_label in unique_labels:
            if other_label == label:
                continue
            other_label_indices = np.where(y == other_label)[0]
            other_label_data = X_transformed[other_label_indices]
            label_combination_mse = mse_between_vectors(label_data, other_label_data)
            label_combination_mses.append(label_combination_mse)

    return label_mses, label_combination_mses

def add_random_noise(X_transformed, noise_level=3):
    # Compute the mean and standard deviation of the data
    X_mean = np.mean(X_transformed, axis=0)
    X_std = np.std(X_transformed, axis=0)

    # Generate noise with the same shape as X_transformed
    noise = np.random.normal(loc=0.0, scale=noise_level*X_std, size=X_transformed.shape)

    # Add noise to the data
    X_noisy = X_transformed + noise

    return X_noisy


def add_relative_uniform_noise(X, d, min_mul, max_mul):
    # perform PCA decomposition
    pca = PCA(n_components=d)
    X_pca = pca.fit_transform(X)
    X_pca_noisy = np.copy(X_pca)

    # get the number of samples and dimensions
    n_samples, n_dims = X_pca.shape

    # randomly select half of the samples
    sample_indices = np.random.choice(n_samples, size=n_samples // 2, replace=False)

    # generate uniform multipliers for each dimension of each selected sample
    multipliers = np.random.uniform(min_mul, max_mul, size=(len(sample_indices), n_dims))
    print("multipliers ", multipliers.shape)

    # multiply the selected samples with the generated multipliers
    X_pca_noisy[sample_indices] *= multipliers

    # perform PCA reconstruction
    X_noisy = pca.inverse_transform(X_pca_noisy)
    return X_noisy


def mask_data(data, mask_prob):
    """
    Masks a percentage of dimensions in each sample of the input data randomly.

    Args:
        data (numpy.ndarray): The input data with shape (n_samples, n_dims).
        mask_prob (float): The percentage of dimensions to be masked (between 0 and 1).

    Returns:
        numpy.ndarray: The masked data with shape (n_samples, n_dims).
    """
    n_samples, n_dims = data.shape
    n_masked_dims = int(n_dims * mask_prob)
    masked_data = data.copy()

    for i in range(n_samples):
        masked_dims = np.random.choice(n_dims, n_masked_dims, replace=False)
        masked_data[i, masked_dims] = 0

    return masked_data

def generate_data(n_samples,three_classes = False):

    if three_classes:
        group_samples = n_samples //3
        # Generate label 0 data
        x0 = np.random.uniform(-4, 4, group_samples)
        y0 = np.random.normal(1.5, 1.5, group_samples)

        # Generate label 1 data
        x1 = np.random.normal(5, 1.5, group_samples)
        y1 = np.random.uniform(-4, 4, group_samples)

        # Generate label 2 data
        x2 = np.random.normal(5, 1.5, group_samples)
        y2 = np.random.normal(-5,1, group_samples)

        # Combine data and labels
        X = np.concatenate((np.stack((x0, y0), axis=-1),
                            np.stack((x1, y1), axis=-1),
                            np.stack((x2, y2), axis=-1)))

        y = np.concatenate((np.zeros(group_samples),
                            np.ones(group_samples),
                            np.full(group_samples, 2)))
        # plot_data(X, y)
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        return X, y


    half_samples = n_samples // 2

    # Generate label 0 data
    x0 = np.random.uniform(-2, 2, half_samples)
    y0 = np.random.normal(1.5, 0.5, half_samples)

    # Generate label 1 data
    x1 = np.random.normal(1.5, 0.5, half_samples)
    y1 = np.random.uniform(-2, 2, half_samples)

    # Combine data and labels
    X = np.concatenate((np.stack((x0, y0), axis=-1),
                        np.stack((x1, y1), axis=-1)))
    y = np.concatenate((np.ones(half_samples), np.zeros(half_samples)))

    return X, y


def add_noise_to_data(X,d_noise):
    noise_means = np.random.uniform(-2, 2, d_noise)
    noise_stds = np.random.normal(2, 0, d_noise)
    noise_matrix = np.random.normal(noise_means, noise_stds, size=(len(X), d_noise))


    X_noisy = np.concatenate([X, noise_matrix], axis=1)
    # X_noisy = (X_noisy - np.mean(X_noisy, axis=0)) / np.std(X_noisy, axis=0)

    return X_noisy


def plot_data(X, y):
    labels = np.unique(y)
    num_labels = len(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_labels))

    fig, ax = plt.subplots()
    for i, label in enumerate(labels):
        mask = np.where(y == label, True, False)
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], label='Label {}'.format(label))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Generated Data')
    ax.legend(title='Labels', loc='upper left')

    plt.savefig('generated_data.png')
    plt.show()


def transform_data(X, d):
    # generate transformation matrix V using MGS algorithm
    V = np.random.randn(d, X.shape[1])
    Q, R = np.linalg.qr(V)
    V = Q.T
    # normalize V
    norm = np.linalg.norm(V, axis=1)
    V = (V.T / norm)
    # transform X
    X_transformed = np.dot(X, V.T)
    # return transformed data and inverse transformation matrix
    return X_transformed, V

def get_data(n_samples, d, d_noise,three_classes = False):
    X, y = generate_data(n_samples,three_classes)
    X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X_transformed, V = transform_data(X_normalized, d=d)
    X_transformed_normalized = (X_transformed - np.mean(X_transformed, axis=0)) / np.std(X_transformed, axis=0)
    # X_transformed_noise_added = add_random_noise(X_transformed, noise_level=0.5)
    X_transformed_noise_concat = add_noise_to_data(X_transformed_normalized, d_noise)
    # plot_data(X_normalized,y)
    # label_mses, label_combination_mses = get_label_mse(X_transformed, y)
    # print("mse before noising:")
    # print("label mses:", label_mses, " label combination mses:", label_combination_mses)
    return X_transformed_noise_concat, y

def get_moons_data(n_samples, d, d_noise,noise = True):
    X, y = make_moons(n_samples=n_samples, noise=0.1)
    X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X_transformed, V = transform_data(X_normalized, d=d)
    X_transformed_normalized = (X_transformed - np.mean(X_transformed, axis=0)) / np.std(X_transformed, axis=0)
    if noise:
        # X_transformed_noise_added = add_random_noise(X_transformed, noise_level=0.5)
        X_transformed_noise_concat = add_noise_to_data(X_transformed_normalized, d_noise)
        return X_transformed_noise_concat, y

    return X_transformed_normalized, y

if __name__ == '__main__':
   X_transformed_noise_added, y = get_data(100,5,20)
   # print("mse after noising:")
   # label_mses, label_combination_mses = get_label_mse(X_transformed_noise_added, y)
   # print("label mses:", label_mses, " label combination mses:", label_combination_mses)