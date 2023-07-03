import csv
import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from two_class_data_generation import *
import auto_encoder_model
import auto_encoder_with_feature_selector
import auto_encoder_with_feature_selector_simsiam
import auto_encoder_simsiam_with_masking
from sklearn import svm
import auto_encoder_with_self_attention_simsiam
from auto_encoder_with_feature_selector import FeatureSelector
from vae_model import *


def create_pca_data(data, num_components):
    pca = PCA(n_components=num_components)
    pca_data = pca.fit_transform(data)

    # Perform inverse PCA transformation
    inverse_data = pca.inverse_transform(pca_data)

    # Print explained variance ratios for each component
    original_dims = np.zeros(data.shape[1])
    for i in range(len(pca.components_) // 2):
        component = pca.components_[i + len(pca.components_) // 2]
        feature_importance = np.abs(component) / np.sum(np.abs(component))
        original_dims += feature_importance

    return pca_data,inverse_data



def run_kmeans(data, labels):
    # Create dataset and split into train and validation sets
    valid_size = 0.3
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=valid_size, shuffle=True,
                                                                      stratify=labels)

    # Set up model and training parameters
    kmeans_model = KMeans(n_clusters=2)

    # Train and predict
    kmeans_model.fit(train_data)
    train_predictions = kmeans_model.predict(train_data)
    val_predictions = kmeans_model.predict(val_data)

    # Get the indices of the samples in each centroid
    centroid_0_indices = np.where(train_predictions == 0)[0]
    centroid_1_indices = np.where(train_predictions == 1)[0]

    # Calculate the majority class for each centroid
    centroid_0_majority = np.argmax(np.bincount(train_labels[centroid_0_indices].numpy().astype(int)))
    centroid_1_majority = np.argmax(np.bincount(train_labels[centroid_1_indices].numpy().astype(int)))

    # Set the labels of the centroids
    if centroid_0_majority == 1:
        centroid_0_label = 1
        centroid_1_label = 0
    else:
        centroid_0_label = 0
        centroid_1_label = 1

    # Calculate the precision of the centroids based on the majority class
    train_centroid_labels = np.zeros(train_predictions.shape)
    train_centroid_labels[centroid_0_indices] = centroid_0_label
    train_centroid_labels[centroid_1_indices] = centroid_1_label

    val_centroid_labels = np.zeros(val_predictions.shape)
    val_centroid_labels[val_predictions == 0] = centroid_0_label
    val_centroid_labels[val_predictions == 1] = centroid_1_label

    train_precision = precision_score(train_labels, train_centroid_labels)
    val_precision = precision_score(val_labels, val_centroid_labels)

    return train_precision, val_precision


def run_kmeans_multiple_labels(data, labels):
    # Create dataset and split into train and validation sets
    valid_size = 0.3
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=valid_size, shuffle=True,
                                                                      stratify=labels)

    # Set up model and training parameters
    kmeans_model = KMeans(n_clusters=3)

    # Train and predict
    kmeans_model.fit(train_data)
    train_predictions = kmeans_model.predict(train_data)
    val_predictions = kmeans_model.predict(val_data)

    # Get the indices of the samples in each centroid
    centroid_0_indices = np.where(train_predictions == 0)[0]
    centroid_1_indices = np.where(train_predictions == 1)[0]
    centroid_2_indices = np.where(train_predictions == 2)[0]

    # Calculate the majority class for each centroid
    centroid_0_majority = np.argmax(np.bincount(train_labels[centroid_0_indices].numpy().astype(int)))
    centroid_1_majority = np.argmax(np.bincount(train_labels[centroid_1_indices].numpy().astype(int)))
    centroid_2_majority = np.argmax(np.bincount(train_labels[centroid_2_indices].numpy().astype(int)))

    # Set the labels of the centroids
    centroid_labels = [0, 1, 2]  # Labels for each centroid
    centroid_majorities = [centroid_0_majority, centroid_1_majority, centroid_2_majority]
    centroid_labels = [centroid_labels[majority] for majority in centroid_majorities]

    # Calculate the precision of the centroids based on the majority class
    train_centroid_labels = np.zeros(train_predictions.shape)
    train_centroid_labels[centroid_0_indices] = centroid_labels[0]
    train_centroid_labels[centroid_1_indices] = centroid_labels[1]
    train_centroid_labels[centroid_2_indices] = centroid_labels[2]

    val_centroid_labels = np.zeros(val_predictions.shape)
    val_centroid_labels[val_predictions == 0] = centroid_labels[0]
    val_centroid_labels[val_predictions == 1] = centroid_labels[1]
    val_centroid_labels[val_predictions == 2] = centroid_labels[2]

    train_precision = precision_score(train_labels, train_centroid_labels, average='macro')
    val_precision = precision_score(val_labels, val_centroid_labels, average='macro')

    return train_precision, val_precision

def run_svm(data, labels):
    # Create dataset and split into train and validation sets
    valid_size = 0.3
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=valid_size, shuffle=True,
                                                                      stratify=labels)

    # Set up model and training parameters
    svm_model = svm.SVC()

    # Train and predict
    svm_model.fit(train_data, train_labels)
    train_predictions = svm_model.predict(train_data)
    val_predictions = svm_model.predict(val_data)

    # Calculate the precision
    train_precision = precision_score(train_labels, train_predictions, average='macro')
    val_precision = precision_score(val_labels, val_predictions, average='macro')

    return train_precision, val_precision


def main():
    precision_by_run_base = []
    precision_by_run_model = []

    n_trials = 100
    n_samples = 1000
    d = 5
    d_noise = 20
    noise = 'noise'
    vae = False
    pca = False
    masked = False
    masked_simsiam = False
    simsiam = True

    for i in range(n_trials):

        # Generate data
        if noise == 'noise':
            # data, labels = get_moons_data(n_samples=n_samples, d=d, d_noise=d_noise)
            data, labels = get_data(n_samples=n_samples, d=d, d_noise=d_noise, three_classes=True)

        else:
            # data, labels = generate_data(n_samples)
            data, labels = generate_data(n_samples,three_classes=True)

        # clean_data, clean_labels = generate_data(n_samples)

        # train_precision_base, val_precision_base = run_kmeans_multiple_labels(torch.Tensor(data), torch.Tensor(labels))
        train_precision_base, val_precision_base = run_svm(torch.Tensor(data), torch.Tensor(labels))



        if pca:
            pca_noise = "pca_noise"
            model_path = '../models/encoded_model_and_data_' + pca_noise + '.pt'
            saved_data = torch.load(model_path)
            auto_encoder_model = auto_encoder_model.Autoencoder(data.shape[1], saved_data['hidden_dim'])
            # Load the saved model state_dict
            auto_encoder_model.load_state_dict(saved_data['state_dict'])
            data = torch.Tensor(data)
            labels = torch.LongTensor(labels)
            auto_encoder_model.eval()
            data = auto_encoder_model.encoder(data).detach()


        elif vae:
            vae_noise = "mask_noise"
            model_path = '../models/vae_encoded_model_and_data_' + vae_noise + '.pt'
            saved_data = torch.load(model_path)
            vae_model = VariationalAutoencoder(data.shape[1], saved_data['hidden_dim'])
            # Load the saved model state_dict
            vae_model.load_state_dict(saved_data['state_dict'])
            labels = torch.LongTensor(labels)
            vae_model.eval()
            with torch.no_grad():
                data = vae_model.encoder(data)
            if pca:
                data = create_pca_data(data, d)


        elif masked:
            noise = "mask_noise"
            model_path = '../models/encoded_model_simsiam_masked.pt'
            saved_data = torch.load(model_path)
            auto_encoder_model = auto_encoder_simsiam_with_masking.Autoencoder(data.shape[1], saved_data['hidden_dim'])
            # Load the saved model state_dict
            auto_encoder_model.load_state_dict(saved_data['state_dict'])
            data = torch.Tensor(data)
            labels = torch.LongTensor(labels)
            auto_encoder_model.eval()
            with torch.no_grad():
                data = auto_encoder_model.encoder(data)


        elif simsiam:
            model_path = '../models/encoded_model_simsiam.pt'
            saved_data = torch.load(model_path)
            auto_encoder_model = auto_encoder_with_feature_selector_simsiam.Autoencoder(data.shape[1], saved_data['hidden_dim'])
            # Load the saved model state_dict
            auto_encoder_model.load_state_dict(saved_data['state_dict'])

            # Modify the feature selector weights
            weights = auto_encoder_model.feature_selector1.get_weights()
            # Calculate  threshold
            threshold = np.percentile(weights.detach().numpy(), 80)
            threshold_gate = torch.where(auto_encoder_model.feature_selector1.get_weights() > threshold,
                                         auto_encoder_model.feature_selector1.get_weights(), torch.tensor(float('-inf')))
            # Replace with your desired weights
            auto_encoder_model.feature_selector1.mu.data.copy_(threshold_gate)
            softmax_gate = saved_data['feature_selector_stochastic_weights1']
            # print("original stochastic gate1:")
            # print(softmax_gate)
            # print("new stochastic gate1")
            # print(auto_encoder_model.feature_selector1.hard_sigmoid(threshold_gate))
            #
            #
            # Modify the feature selector weights
            weights = auto_encoder_model.feature_selector2.get_weights()
            # Calculate the threshold
            threshold = np.percentile(weights.detach().numpy(), 80)
            threshold_gate = torch.where(auto_encoder_model.feature_selector2.get_weights() > threshold,
                                         auto_encoder_model.feature_selector2.get_weights(),
                                         torch.tensor(float('-inf')))
            # Replace with your desired weights
            auto_encoder_model.feature_selector2.mu.data.copy_(threshold_gate)
            softmax_gate = saved_data['feature_selector_stochastic_weights2']
            print("original stochastic gate2:")
            print(softmax_gate)
            print("new stochastic gate2")
            print(auto_encoder_model.feature_selector2.hard_sigmoid(threshold_gate))



            data = torch.Tensor(data)
            labels = torch.LongTensor(labels)
            auto_encoder_model.eval()
            data = auto_encoder_model.feature_selector1(data).detach()
            # data = data * threshold_gate.unsqueeze(0)

            # data = auto_encoder_model.encoder(data).detach()
            if pca:
                data = create_pca_data(data, d)
                data = torch.Tensor(data)


        # val_precision_base = run_kmeans_multiple_labels(data.detach(),labels.detach())
        train_precision_model, val_precision_model = run_svm(data.detach(),labels.detach())

        precision_by_run_base.append(val_precision_base)
        precision_by_run_model.append(val_precision_model)

        # Print validation precision
        print(f'Validation Precision base: {val_precision_base:.4f}'
              ,f'Validation Precision model: {val_precision_model:.4f}')

    csv_path_base = '../results/kmeans_two_centroids_precision_latent_ae_' + noise + '.csv'

    if pca:
        csv_path_model = '../results/kmeans_two_centroids_precision_pca_' + noise + '.csv'

    elif vae:
        csv_path_model = '../results/kmeans_two_centroids_precision_latent_vae_' + vae_noise + '.csv'

    elif masked_simsiam:
        csv_path_model = '../results/kmeans_two_centroids_precision_latent_simsiam_masekd.csv'

    elif simsiam:
        csv_path_model = '../results/kmeans_two_centroids_precision_latent_simsiam_feature_selector.csv'

    else:
        csv_path_model = '../results/simple_testing.csv'




    # Check if the CSV file already exists
    file_exists_base = os.path.isfile(csv_path_base)
    mean_precision_base = np.mean(precision_by_run_base)
    max_precision_base = np.max(precision_by_run_base)
    min_precision_base = np.min(precision_by_run_base)


    # Open the CSV file in append mode
    with open(csv_path_base, 'a', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)

        # Write the header row if the file is new
        if not file_exists_base:
            writer.writerow(['Mean Precision', 'Max Precision', 'Min Precision'])

        # Write the results row
        writer.writerow([mean_precision_base, max_precision_base, min_precision_base])

    # Print the mean, max, and min precision
    print(f"Mean precision base: {mean_precision_base:.2f}")
    print(f"Max precision base: {max_precision_base:.2f}")
    print(f"Min precision base: {min_precision_base:.2f}")

    # Check if the CSV file already exists
    file_exists_model = os.path.isfile(csv_path_model)
    mean_precision_model = np.mean(precision_by_run_model)
    max_precision_model = np.max(precision_by_run_model)
    min_precision_model = np.min(precision_by_run_model)

    # Open the CSV file in append mode
    with open(csv_path_model, 'a', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)

        # Write the header row if the file is new
        if not file_exists_model:
            writer.writerow(['Mean Precision', 'Max Precision', 'Min Precision'])

        # Write the results row
        writer.writerow([mean_precision_model, max_precision_model, min_precision_model])

    # Print the mean, max, and min precision
    print(f"Mean precision model: {mean_precision_model:.2f}")
    print(f"Max precision model: {max_precision_model:.2f}")
    print(f"Min precision model: {min_precision_model:.2f}")


if __name__ == '__main__':
    main()