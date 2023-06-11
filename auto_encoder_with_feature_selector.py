import math
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
from two_class_data_generation import *
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from old_pca_attempts import pca_model

# def calculate_distances(data, labels):
#     label_0 = 0.0
#     label_1 = 1.0
#     label_0_data = data[labels == label_0]
#     label_1_data = data[labels == label_1]
#
#     distances_within_label_0 = pairwise_distances(label_0_data)
#     distances_within_label_1 = pairwise_distances(label_1_data)
#     distances_between_labels = pairwise_distances(label_0_data, label_1_data)
#
#     return distances_within_label_0, distances_within_label_1, distances_between_labels

# def get_distances_between_labels(data,labels):
#     distances_within_label_1, distances_within_label_2, distances_between_labels = calculate_distances(data, labels)
#
#     # Print the distances within label 0
#     print("Distances within label 0:")
#     print(np.mean(distances_within_label_1))
#
#     # Print the distances within label 1
#     print("Distances within label 1:")
#     print(np.mean(distances_within_label_2))
#
#     # Print the distances between label 0 and label 1
#     print("Distances between label 0 and label 1:")
#     print(np.mean(distances_between_labels))

class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma = 1):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(0.01 * torch.randn(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size())
        self.stochastic_gate = torch.randn(self.mu.size())
        self.sigma = sigma

    def forward(self, prev_x):
        z = self.mu + self.sigma * self.noise.normal_() * self.training
        self.stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * self.stochastic_gate
        return new_x

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def get_weights(self):
        return self.mu

    def get_stochastic_weights(self):
        return self.stochastic_gate

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=50,sigma = 1):
        super(Autoencoder, self).__init__()

        # Feature Selector
        self.feature_selector = FeatureSelector(input_dim, sigma)

        # Encoder layers
        self.encoder = nn.Sequential(
            self.feature_selector,
            nn.Linear(input_dim, input_dim*10),
            nn.Dropout(0.2),
            nn.BatchNorm1d(input_dim*10),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout layer for regularization
            nn.Linear(input_dim*10, latent_dim)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim*10),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout layer for regularization
            nn.Linear(input_dim*10, input_dim)
        )

        # Batch normalization layers
        self.batch_norm2 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        # Apply the FeatureSelector
        x = self.feature_selector(x)

        # Encode the input
        x = self.encoder(x)
        x = self.batch_norm2(x)  # Apply batch normalization

        # Decode the encoded input
        x = self.decoder(x)

        return x

def cosine_distance_loss(input, output, power=2):
    input_norm = torch.nn.functional.normalize(input, p=2, dim=1)
    output_norm = torch.nn.functional.normalize(output, p=2, dim=1)
    cosine_distance = 1 - torch.sum(input_norm * output_norm, dim=1)
    cosine_distance = torch.pow(cosine_distance, power)
    return torch.mean(cosine_distance)


def InfoNCE_loss(pos_inputs, neg_inputs, temperature=1.0, negative_weight=2.0):
    pos_cos_dist = cosine_distance_loss(pos_inputs[:, 0], pos_inputs[:, 1])
    neg_cos_dist = cosine_distance_loss(neg_inputs[:, 0], neg_inputs[:, 1])
    pos_sim = torch.exp(-pos_cos_dist / temperature)
    neg_sim = torch.exp(-neg_cos_dist / temperature)

    # Apply weight to the negative similarity
    neg_sim_weighted = negative_weight * neg_sim

    numerator = pos_sim
    denominator = torch.sum(neg_sim_weighted) + pos_sim + 1e-8
    loss = -torch.log(numerator / denominator)
    return loss.mean()

def create_pairs(orig_latent, noisy_latent, labels, num_neg_pairs=10):
    n = orig_latent.shape[0]
    pos_pairs = []
    neg_pairs = []
    for i in range(n):
        label = labels[i]
        pos_pairs.append((orig_latent[i], noisy_latent[i]))
        neg_false_idx = np.random.choice(np.where((labels == label) & (np.arange(n) != i))[0])
        neg_pairs.append((orig_latent[neg_false_idx], noisy_latent[i]))
        for j in range(num_neg_pairs):
            neg_idx = np.random.choice(np.where(labels != label)[0])
            neg_pairs.append((orig_latent[neg_idx], noisy_latent[i]))
    pos_pairs = torch.stack([torch.stack(p) for p in pos_pairs])
    neg_pairs = torch.stack([torch.stack(p) for p in neg_pairs])
    return pos_pairs, neg_pairs


def regularization(model, lambda_reg):
    reg_loss = 0
    for param in model.parameters():
        reg_loss += torch.norm(param)
    return lambda_reg * reg_loss


def train(model, optimizer, recon_criterion, train_loader, valid_loader, device, epochs=10):
    train_loss_history = []
    valid_loss_history = []

    best_model_state_dict = None
    best_val_loss = float('inf')
    num_epochs_no_improve = 0

    for epoch in range(1, epochs+1):
        train_recon_loss = 0.0
        train_contrastive_loss = 0.0
        train_reg_loss = 0.0
        train_loss = 0.0
        valid_recon_loss = 0.0
        valid_contrastive_loss = 0.0
        valid_reg_loss = 0.0
        valid_loss = 0.0

        model.train()


        for batch_idx, (data_orig, labels) in enumerate(train_loader):
            data_orig, labels = data_orig.to(device), labels.to(device)
            data_noisy = model.feature_selector(data_orig)
            optimizer.zero_grad()

            # Compute latent representations
            latents_orig = model.encoder(data_orig)
            latents_noisy = model.encoder(data_noisy)



            # testing a regular autoencoder with feature selection
            # recon_loss = recon_criterion(model.decoder(latents_noisy), data_orig)
            # pos_pairs, neg_pairs = create_pairs(data_orig, data_noisy, labels)


            # Compute reconstruction loss

            recon_loss = recon_criterion(latents_noisy, latents_orig) + \
                         recon_criterion(model.decoder(latents_noisy), data_orig)
            train_recon_loss += recon_loss

            pos_pairs, neg_pairs = create_pairs(latents_orig, latents_noisy, labels)
            pos_latents = pos_pairs
            neg_latents = neg_pairs

            contrastive_loss = InfoNCE_loss(pos_latents, neg_latents)
            train_contrastive_loss += contrastive_loss

            # Add regularization loss
            reg_loss = regularization(model, lambda_reg=0.001)
            train_reg_loss+=reg_loss
            loss = 10*contrastive_loss + 0.5*recon_loss

            # Perform backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        with torch.no_grad():
            model.eval()
            for batch_idx, (data_orig, labels) in enumerate(valid_loader):
                # feature_selector_weights = model.feature_selector.get_weights()
                # print("weights of feature selector are:")
                # print(feature_selector_weights)
                data_orig, labels = data_orig.to(device), labels.to(device)
                data_noisy = model.feature_selector(data_orig)

                # Compute latent representations
                latents_orig = model.encoder(data_orig)
                latents_noisy = model.encoder(data_noisy)

                # Compute reconstruction loss
                recon_loss = recon_criterion(latents_noisy, latents_orig) \
                             + recon_criterion(model.decoder(latents_noisy), data_orig)
                valid_recon_loss += recon_loss

                pos_pairs, neg_pairs = create_pairs(latents_orig, latents_noisy, labels)
                pos_latents = pos_pairs
                neg_latents = neg_pairs

                contrastive_loss = InfoNCE_loss(pos_latents, neg_latents)
                valid_contrastive_loss += contrastive_loss

                # Add regularization loss
                reg_loss = regularization(model, lambda_reg=0.001)
                valid_reg_loss += reg_loss
                loss = 10 * contrastive_loss + 0.5 * recon_loss
                valid_loss += loss

        train_loss /= len(train_loader.dataset)
        train_recon_loss /= len(train_loader.dataset)
        train_contrastive_loss /= len(train_loader.dataset)
        train_reg_loss /= len(train_loader.dataset)

        valid_loss /= len(valid_loader.dataset)
        valid_recon_loss /= len(valid_loader.dataset)
        valid_contrastive_loss /= len(valid_loader.dataset)
        valid_reg_loss /= len(valid_loader.dataset)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        print('Epoch: {} \ttrain_recon_loss Loss: {:.6f} \ttrain_contrastive_loss Loss: {:.6f} '
              '\ttrain_reg_loss Loss: {:.6f}'.format(epoch, train_recon_loss, train_contrastive_loss, train_reg_loss))

        print('Epoch: {} \tvalid_recon_loss Loss: {:.6f} \tvalid_contrastive_loss Loss: {:.6f} '
              '\tvalid_reg_loss Loss: {:.6f}'.format(epoch, valid_recon_loss, valid_contrastive_loss, valid_reg_loss))

        if valid_loss < best_val_loss:
            best_model_state_dict = model.state_dict()
            best_val_loss = valid_loss
            num_epochs_no_improve = 0
        else:
            num_epochs_no_improve += 1
            if num_epochs_no_improve == 5:
                print(f"No improvement for {num_epochs_no_improve} epochs, stopping training")
                break

    return train_loss_history, valid_loss_history, best_model_state_dict

def main():
    # Set random seed for reproducibility
    print("we are in main")
    n_samples = 20000
    d = 5
    d_noise = 20
    data_orig, labels = get_data(n_samples=n_samples, d=d, d_noise=d_noise)
    # Create torch dataset
    data = torch.Tensor(data_orig)
    labels = torch.LongTensor(labels)
    dataset = torch.utils.data.TensorDataset(data, labels)
    # Split dataset into train and validation sets
    valid_size = 0.2
    # Split the data into train and validation sets
    train_idx, valid_idx = train_test_split(range(len(labels)), test_size=valid_size, shuffle=True)
    train_set = Subset(dataset, train_idx)
    valid_set = Subset(dataset, valid_idx)
    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer and loss function
    input_dim = data.shape[1]
    hidden_dim = d
    model = Autoencoder(input_dim, hidden_dim)
    criterion = cosine_distance_loss
    regularization = 0.003  # add L2 regularization
    optimizer = optim.Adagrad(model.parameters(), lr=0.001, weight_decay=regularization)

    # Train model
    epochs = 25
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loss_history, valid_loss_history,best_model_state_dict = train(model, optimizer, criterion,
                                                                         train_loader, valid_loader, device,epochs=epochs)
    # Plot loss and accuracy history
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(valid_loss_history, label='Validation Loss')
    plt.legend()
    plt.show()
    plt.savefig('plots_and_images/loss_vs_epochs_autoencoder_feature_selection.png')
    file_name = 'models/encoded_model_with_feature_selection.pt'

    torch.save({
        'state_dict': model.state_dict(),
        'labels': labels.detach(),
        'feature_selector_weights':model.feature_selector.get_weights(),
        'feature_selector_stochastic_weights': model.feature_selector.get_stochastic_weights(),
        'hidden_dim': hidden_dim
    }, file_name)


if __name__ == '__main__':
    main()