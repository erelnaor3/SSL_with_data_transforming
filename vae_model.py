import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset , DataLoader
import torch.nn as nn
from two_class_data_generation import *
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances


def calculate_distances(data, labels):
    label_0 = 0.0
    label_1 = 1.0
    label_0_data = data[labels == label_0]
    label_1_data = data[labels == label_1]

    distances_within_label_0 = pairwise_distances(label_0_data)
    distances_within_label_1 = pairwise_distances(label_1_data)
    distances_between_labels = pairwise_distances(label_0_data, label_1_data)

    return distances_within_label_0, distances_within_label_1, distances_between_labels

def get_distances_between_labels(data,labels):
    distances_within_label_1, distances_within_label_2, distances_between_labels = calculate_distances(data, labels)

    # Print the distances within label 0
    print("Distances within label 0:")
    print(np.mean(distances_within_label_1))

    # Print the distances within label 1
    print("Distances within label 1:")
    print(np.mean(distances_within_label_2))

    # Print the distances between label 0 and label 1
    print("Distances between label 0 and label 1:")
    print(np.mean(distances_between_labels))


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=50):
        super(VariationalAutoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 10),
            nn.Dropout(0.2),
            nn.BatchNorm1d(input_dim * 10),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout layer for regularization
            nn.Linear(input_dim * 10, latent_dim * 2)  # Two times latent_dim for mean and variance
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim * 10),  # Adjusted input_dim
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout layer for regularization
            nn.Linear(input_dim * 10, input_dim)
        )

        # Batch normalization layers
        self.batch_norm2 = nn.BatchNorm1d(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode the input
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        x = self.batch_norm2(z)  # Apply batch normalization

        # Decode the encoded input
        x = self.decoder(x)

        return x, mu, logvar

def vae_loss(mu, logvar):
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss

def cosine_distance_loss(input, output):
    input_norm = torch.nn.functional.normalize(input, p=2, dim=1)
    output_norm = torch.nn.functional.normalize(output, p=2, dim=1)
    cosine_distance = 1 - torch.sum(input_norm * output_norm, dim=1)
    return torch.mean(cosine_distance)

def InfoNCE_loss(pos_inputs, neg_inputs, temperature=1.0, negative_weight=1.0):
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
        train_kl_loss = 0.0
        train_loss = 0.0
        valid_recon_loss = 0.0
        valid_contrastive_loss = 0.0
        valid_kl_loss = 0.0
        valid_loss = 0.0

        model.train()


        for batch_idx, (data_orig,data_noisy, labels) in enumerate(train_loader):
            data_orig, data_noisy, labels = data_orig.to(device), data_noisy.to(device), labels.to(device)
            optimizer.zero_grad()

            # Compute latent representations
            latents_orig = model.encoder(data_orig)
            latents_noisy = model.encoder(data_noisy)
            # Compute reconstruction loss

            recon_loss = recon_criterion(latents_noisy, latents_orig) + \
                         recon_criterion(model.forward(data_noisy)[0], data_orig)

            # compute reconstruction loss
            pos_pairs, neg_pairs = create_pairs(latents_orig, latents_noisy, labels)
            pos_latents = pos_pairs
            neg_latents = neg_pairs

            contrastive_loss = InfoNCE_loss(pos_latents, neg_latents)

            # Compute vae loss
            output, mu, logvar = model(data_noisy)
            kl_loss = vae_loss(mu, logvar)

            # Add regularization loss
            train_recon_loss+=recon_loss
            train_contrastive_loss+=contrastive_loss
            train_kl_loss+=kl_loss
            # loss = recon_loss + contrastive_loss + reg_loss
            loss = 10*recon_loss + contrastive_loss

            # Perform backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        with torch.no_grad():
            model.eval()
            for batch_idx, (data_orig,data_noisy, labels) in enumerate(valid_loader):
                data_orig, data_noisy, labels = data_orig.to(device), data_noisy.to(device), labels.to(device)
                # Compute latent representations
                latents_orig = model.encoder(data_orig)
                latents_noisy = model.encoder(data_noisy)

                # Compute reconstruction loss
                recon_loss = recon_criterion(latents_noisy, latents_orig) + \
                             recon_criterion(model.forward(data_noisy)[0], data_orig)

                # compute reconstruction loss
                pos_pairs, neg_pairs = create_pairs(latents_orig, latents_noisy, labels)
                pos_latents = pos_pairs
                neg_latents = neg_pairs

                contrastive_loss = InfoNCE_loss(pos_latents, neg_latents)

                # Compute vae loss
                output, mu, logvar = model(data_noisy)
                kl_loss = vae_loss(mu, logvar)

                # Add regularization loss
                valid_recon_loss += recon_loss
                valid_contrastive_loss += contrastive_loss
                valid_kl_loss += kl_loss
                # loss = recon_loss + contrastive_loss + reg_loss
                loss = recon_loss + contrastive_loss
                valid_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        train_recon_loss /= len(train_loader.dataset)
        train_contrastive_loss /= len(train_loader.dataset)
        train_kl_loss /= len(train_loader.dataset)

        valid_loss /= len(valid_loader.dataset)
        valid_recon_loss /= len(valid_loader.dataset)
        valid_contrastive_loss /= len(valid_loader.dataset)
        valid_kl_loss /= len(valid_loader.dataset)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        print('Epoch: {} \ttrain_recon_loss Loss: {:.6f} \ttrain_contrastive_loss Loss: {:.6f} '
              '\ttrain_kl_loss Loss: {:.6f}'.format(epoch, train_recon_loss, train_contrastive_loss, train_kl_loss))

        print('Epoch: {} \tvalid_recon_loss Loss: {:.6f} \tvalid_contrastive_loss Loss: {:.6f} '
              '\tvalid_kl_loss Loss: {:.6f}'.format(epoch, valid_recon_loss, valid_contrastive_loss, valid_kl_loss))

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
    n_samples = 10000
    d = 5
    d_noise = 20
    pca_noise_flag = False
    mask_flag = True
    data_orig, labels = get_data(n_samples=n_samples, d=d, d_noise=d_noise)

    if pca_noise_flag:
        # here what we do we leave that the data in its pca latent variation as the inverse is poor
        # but then the recon error is useless, well all the errors are useless lets see how it goes
        data_noisy = add_relative_uniform_noise(data_orig, d, min_mul=0.8,max_mul=1.2)
        data_orig = add_relative_uniform_noise(data_orig, d, min_mul=1.0, max_mul=1.0)
    elif mask_flag:
        mask_ratio = 0.3
        data_noisy = mask_data(data_orig,mask_ratio)
    else:
        data_noisy = data_orig



    # Create torch dataset
    data_orig = torch.Tensor(data_orig)
    data_noisy = torch.Tensor(data_noisy)
    labels = torch.LongTensor(labels)
    dataset = torch.utils.data.TensorDataset(data_orig, data_noisy, labels)
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
    input_dim = data_orig.shape[1]
    hidden_dim = d # increase number of hidden dimensions
    model = VariationalAutoencoder(input_dim, hidden_dim)
    criterion = nn.MSELoss(reduction='mean')
    regularization = 0.003  # add L2 regularization
    optimizer = optim.Adagrad(model.parameters(), lr=0.001, weight_decay=regularization)

    # Train model
    epochs = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loss_history, valid_loss_history,best_model_state_dict = train(model, optimizer, criterion,
                                                                         train_loader, valid_loader, device,epochs=epochs)
    # Plot loss and accuracy history
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(valid_loss_history, label='Validation Loss')
    plt.legend()
    plt.show()
    plt.savefig('plots_and_images/loss_vs_epochs_vae_mask_noise'+'.png')

    # Save model and relevant parameters
    if pca_noise_flag:
        file_name = 'models/vae_encoded_model_and_data_pca_noise.pt'
    elif mask_flag:
        file_name = 'models/vae_encoded_model_and_data_mask_noise.pt'
    else:
        file_name = 'models/vae_encoded_model_and_data_no_noise.pt'

    torch.save({
        'state_dict': model.state_dict(),
        'labels': labels.detach(),
        'hidden_dim': hidden_dim
    }, file_name)


if __name__ == '__main__':
    main()