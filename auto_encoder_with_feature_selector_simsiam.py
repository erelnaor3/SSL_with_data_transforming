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


class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma=1):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(0.01 * torch.randn(input_dim,), requires_grad=True)
        self.noise = torch.randn(self.mu.size())
        self.stochastic_gate = torch.randn(self.mu.size())
        self.sigma = sigma

    def forward(self, prev_x):
        if self.training:
            strong_noise = torch.randn(self.mu.size())   # Adjust the scale of the strong noise
            z = self.mu + self.sigma * self.noise.normal_() + strong_noise
        else:
            z = self.mu
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
    def __init__(self, input_dim, latent_dim=50):
        super(Autoencoder, self).__init__()

        self.feature_selector1 = FeatureSelector(input_dim)
        self.feature_selector2 = FeatureSelector(input_dim)

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim*5),
            nn.Dropout(0.2),
            nn.BatchNorm1d(input_dim*5),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout layer for regularization
            nn.Linear(input_dim*5, latent_dim)
        )

        # predictor layers
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*5),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout layer for regularization
            nn.Linear(latent_dim*5, latent_dim)
        )

        # Batch normalization layers
        self.batch_norm2 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        # Apply the first feature selector
        x1_f = self.feature_selector1(x)

        # Apply the second feature selector
        x2_f = self.feature_selector2(x)

        # Encode the input
        z1 = self.encoder(x1_f)
        z1 = self.batch_norm2(z1)

        z2 = self.encoder(x2_f)
        z2 = self.batch_norm2(z2)

        # Decode the encoded input
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()

def regularization(model, lambda_reg):
    reg_loss = 0
    for param in model.parameters():
        reg_loss += torch.norm(param)
    return lambda_reg * reg_loss


def train(model, optimizer, criterion, train_loader, valid_loader, device, epochs=10):
    train_loss_history = []
    valid_loss_history = []

    best_model_state_dict = None
    best_val_loss = float('inf')
    num_epochs_no_improve = 0

    for epoch in range(1, epochs+1):
        train_sim_loss = 0.0
        train_reg_loss = 0.0
        train_loss = 0.0
        valid_sim_loss = 0.0
        valid_reg_loss = 0.0
        valid_loss = 0.0

        model.train()


        for batch_idx, (data,labels) in enumerate(train_loader):
            p1, p2, z1, z2 = model(data)
            optimizer.zero_grad()
            sim_loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            train_sim_loss += sim_loss
            # Add regularization loss
            reg_loss = regularization(model, lambda_reg=0.001)
            train_reg_loss+=reg_loss
            # loss = 10*contrastive_loss + 0.5*recon_loss
            loss = sim_loss + reg_loss

            # Perform backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        with torch.no_grad():
            model.eval()
            for batch_idx, (data,labels) in enumerate(valid_loader):
                p1, p2, z1, z2 = model(data)
                optimizer.zero_grad()
                sim_loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                valid_sim_loss += sim_loss
                # Add regularization loss
                reg_loss = regularization(model, lambda_reg=0.001)
                valid_reg_loss += reg_loss
                # loss = 10*contrastive_loss + 0.5*recon_loss
                loss = sim_loss + reg_loss
                valid_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        train_sim_loss /= len(train_loader.dataset)
        train_reg_loss /= len(train_loader.dataset)

        valid_loss /= len(valid_loader.dataset)
        valid_sim_loss /= len(valid_loader.dataset)
        valid_reg_loss /= len(valid_loader.dataset)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        print('\ttrain_sim_loss: {:.6f} \tvalid_sim_loss: {:.6f}'.format(train_sim_loss, valid_sim_loss))

        print('\ttrain_reg_loss: {:.6f} \tvalid_reg_loss: {:.6f}'.format(train_reg_loss, valid_reg_loss))

        if valid_loss < best_val_loss:
            best_model_state_dict = model.state_dict()
            best_val_loss = valid_loss
            num_epochs_no_improve = 0
        else:
            num_epochs_no_improve += 1
            if num_epochs_no_improve == 10:
                print(f"No improvement for {num_epochs_no_improve} epochs, stopping training")
                break

    return train_loss_history, valid_loss_history, best_model_state_dict

def main():
    # Set random seed for reproducibility
    print("we are in main")
    n_samples = 100000
    d = 5
    d_noise = 20
    data_orig, labels = get_data(n_samples=n_samples, d=d, d_noise=d_noise)
    # Create torch dataset

    # Create torch dataset
    data = torch.Tensor(data_orig)
    labels = torch.LongTensor(labels)
    dataset = torch.utils.data.TensorDataset(data, labels)

    # Split dataset into train and validation sets
    valid_size = 0.2
    num_samples = len(dataset)
    indices = list(range(num_samples))
    split = int(np.floor(valid_size * num_samples))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    train_set = torch.utils.data.Subset(dataset, train_idx)
    valid_set = torch.utils.data.Subset(dataset, valid_idx)

    # Create data loaders
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    # Initialize model, optimizer and loss function
    input_dim = data.shape[1]
    hidden_dim = d
    model = Autoencoder(input_dim, hidden_dim)
    criterion = nn.CosineSimilarity(dim=1)
    regularization = 0.005  # add L2 regularization
    optimizer = optim.Adagrad(model.parameters(), lr=0.001, weight_decay=regularization)

    # Train model
    epochs = 50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loss_history, valid_loss_history,best_model_state_dict = train(model, optimizer, criterion,
                                                                         train_loader, valid_loader, device,epochs=epochs)
    # Plot loss and accuracy history
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(valid_loss_history, label='Validation Loss')
    plt.legend()
    plt.show()
    plt.savefig('plots_and_images/autoencoer_simsiam.png')
    file_name = 'models/encoded_model_simsiam.pt'

    torch.save({
        'state_dict': model.state_dict(),
        'labels': labels.detach(),
        'feature_selector_weights1':model.feature_selector1.get_weights(),
        'feature_selector_stochastic_weights1': model.feature_selector1.get_stochastic_weights(),
        'feature_selector_weights2': model.feature_selector2.get_weights(),
        'feature_selector_stochastic_weights2': model.feature_selector2.get_stochastic_weights(),
        'hidden_dim': hidden_dim
    }, file_name)


if __name__ == '__main__':
    main()