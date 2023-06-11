import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
from data_generator import create_twomoon_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset , DataLoader
import torch.nn as nn
from sklearn.decomposition import PCA


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

    return pca_data, inverse_data

def add_noise_to_pca(pca_data, noise_factor=5):
    pca_data_noisy = np.copy(pca_data)
    num_features = pca_data_noisy.shape[1]
    num_bottom_features = num_features//2
    noise = np.random.normal(scale=noise_factor, size=(pca_data_noisy.shape[0], num_bottom_features))
    # print(pca_data.shape)
    # print(noise.shape)
    # when adding to only a sample of the pca dims
    pca_data_noisy[:, num_bottom_features:] += noise
    # pca_data_noisy = pca_data+noise
    # print(pca_data_noisy[:, num_bottom_features:].shape)
    return pca_data_noisy


def create_torch_dataset(data, pca_data, labels, add_noise):
    if add_noise:
        dataset = [(torch.from_numpy(pca_data[i]).float(), torch.from_numpy(data[i]).float(), labels[i]) for i in range(len(data))]
    else:
        dataset = [(torch.from_numpy(pca_data[i]).float(), torch.from_numpy(data[i]).float(), labels[i]) for i in range(len(data))]
    return dataset


class ReconstructionModel(nn.Module):
    def __init__(self, input_dim,hidden_dim,output_dim):
        super(ReconstructionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


def InfoNCE_loss(pos_cos_sim, neg_cos_sim, temperature=0.5):
    pos_sim = torch.exp(pos_cos_sim / temperature)
    neg_sim = torch.exp(neg_cos_sim / temperature)
    numerator = pos_sim
    denominator = torch.sum(neg_sim) + pos_sim + 1e-8
    loss = -torch.log(numerator / denominator)
    return loss.mean()


def create_pairs(inputs, outputs, labels, num_neg_pairs=3):
    n = inputs.shape[0]
    pos_pairs = []
    neg_pairs = []
    for i in range(n):
        label = labels[i]
        pos_pairs.append((inputs[i], outputs[i]))
        for j in range(num_neg_pairs):
            neg_idx = np.random.choice(np.where(labels != label)[0])
            neg_pairs.append((inputs[neg_idx], outputs[i]))
    pos_pairs = torch.stack([torch.stack(p) for p in pos_pairs])
    neg_pairs = torch.stack([torch.stack(p) for p in neg_pairs])
    return pos_pairs, neg_pairs


def train(model, optimizer, recon_criterion, train_loader, valid_loader, device, epochs=10):
    train_loss_history = []
    valid_loss_history = []

    for epoch in range(1, epochs+1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()

        for batch_idx, (pca_data, original_data, labels) in enumerate(train_loader):
            pca_data, original_data, labels = pca_data.to(device), original_data.to(device), labels.to(device)
            optimizer.zero_grad()

            # Compute reconstruction loss
            output = model(pca_data)
            recon_loss = recon_criterion(output, original_data)

            # Compute contrastive loss
            pos_pairs, neg_pairs = create_pairs(original_data, output, labels)
            pos_pairs, neg_pairs = pos_pairs.to(device), neg_pairs.to(device)
            pos_cos_sim = F.cosine_similarity(pos_pairs[:, 0], pos_pairs[:, 1], dim=1)
            neg_cos_sim = F.cosine_similarity(neg_pairs[:, 0], neg_pairs[:, 1], dim=1)
            contrastive_loss = InfoNCE_loss(pos_cos_sim, neg_cos_sim)

            # Combine the losses and perform backpropagation
            loss = recon_loss + contrastive_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        with torch.no_grad():
            model.eval()
            for batch_idx, (pca_data, original_data, labels) in enumerate(valid_loader):
                pca_data, original_data, labels = pca_data.to(device), original_data.to(device), labels.to(device)
                # Compute reconstruction loss
                output = model(pca_data)
                recon_loss = recon_criterion(output, original_data)

                # Compute contrastive loss
                pos_pairs, neg_pairs = create_pairs(original_data, output, labels)
                pos_pairs, neg_pairs = pos_pairs.to(device), neg_pairs.to(device)
                pos_cos_sim = F.cosine_similarity(pos_pairs[:, 0], pos_pairs[:, 1], dim=1)
                neg_cos_sim = F.cosine_similarity(neg_pairs[:, 0], neg_pairs[:, 1], dim=1)
                contrastive_loss = InfoNCE_loss(pos_cos_sim, neg_cos_sim)

                # Combine the losses and perform backpropagation
                loss = recon_loss + contrastive_loss
                valid_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

    return train_loss_history, valid_loss_history


def main():
    # Set random seed for reproducibility
    num_samples = 100000
    input_dim = 40

    # Generate synthetic dataset
    data, labels = create_twomoon_dataset(num_samples, input_dim)

    # Create PCA data and add noise
    num_components = 10
    pca_data = create_pca_data(data, num_components)
    pca_data = add_noise_to_pca(pca_data)

    # Create torch dataset
    dataset = create_torch_dataset(data, pca_data, labels, add_noise=True)

    # Split dataset into train and validation sets
    valid_size = 0.1
    # Split the data into train and validation sets
    train_idx, valid_idx = train_test_split(range(len(labels)), test_size=valid_size, shuffle=True)

    train_set = Subset(dataset, train_idx)
    valid_set = Subset(dataset, valid_idx)

    print("train len:", len(train_set))
    print("valid len:", len(valid_set))

    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer and loss function
    hidden_dim = 100  # increase number of hidden dimensions
    output_dim = input_dim
    model = ReconstructionModel(num_components, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    regularization = 0.001  # add L2 regularization
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=regularization)

    # Train model
    epochs = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loss_history, valid_loss_history = train(model, optimizer, criterion, train_loader, valid_loader, device,
                                                   epochs=epochs)
    # Plot loss and accuracy history
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(valid_loss_history, label='Validation Loss')
    plt.legend()
    plt.show()

    # Save model and relevant parameters
    save_dict = {'hidden_dim': hidden_dim, 'num_components': num_components, 'model_state_dict': model.state_dict()}
    torch.save(save_dict, 'model.pth')

if __name__ == '__main__':
    main()