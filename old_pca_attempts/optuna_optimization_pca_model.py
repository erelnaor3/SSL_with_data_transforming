import torch.optim as optim
from sklearn.model_selection import train_test_split
from data_generator import create_twomoon_dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import optuna

from old_pca_attempts.pca_model import ReconstructionModel, create_torch_dataset, create_pca_data, add_noise_to_pca,train


def main():
    # Set random seed for reproducibility
    num_samples = 10000
    input_dim = 40

    # Generate synthetic dataset
    data, labels = create_twomoon_dataset(num_samples, input_dim)

    # Create PCA data and add noise
    num_components = 10
    pca_data = create_pca_data(data, num_components)
    pca_data_noisy = add_noise_to_pca(pca_data)

    # Create PyTorch dataset
    dataset = create_torch_dataset(data, pca_data_noisy, labels, add_noise=True)

    # Split dataset into training and validation sets
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def objective(trial):
        # Sample hyperparameters
        hidden_dim = trial.suggest_int("hidden_dim", 50, 250, log=True)
        lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
        nesterov = trial.suggest_categorical("nesterov", [True, False])

        # Define model, optimizer and loss function
        model = ReconstructionModel(num_components, hidden_dim, input_dim).to(device)
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=nesterov)
        recon_criterion = nn.MSELoss()

        # Create data loaders
        batch_size = 256
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

        # Train the model
        epochs = 20
        train_loss_history, valid_loss = train(model, optimizer, recon_criterion, train_loader, valid_loader, device,
                                               epochs=epochs)

        # Return the best validation loss and the corresponding model state dict
        return valid_loss, model.state_dict()

    study = optuna.create_study(direction="minimize")
    best_valid_loss, best_model_state_dict = None, None
    result = study.optimize(objective, n_trials=3)

    if result is not None and len(result) == 2:
        best_valid_loss, best_model_state_dict = result
    else:
        print("Error: study.optimize() method did not return a tuple with two values.")

    print("Best valid loss:", best_valid_loss)
    # Print the best hyperparameters
    print("Best hyperparameters: ", study.best_params)

    # Load the best hyperparameters into a new model instance
    best_hidden_dim = study.best_params["hidden_dim"]
    best_lr = study.best_params["lr"]
    best_weight_decay = study.best_params["weight_decay"]
    best_temperature = study.best_params["temperature"]
    best_optimizer_name = study.best_params["optimizer"]
    best_nesterov = study.best_params["nesterov"]

    best_model = ReconstructionModel(num_components, best_hidden_dim, input_dim).to(device)
    if best_optimizer_name == "Adam":
        best_optimizer = optim.Adam(best_model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
    else:
        best_optimizer = optim.SGD(best_model.parameters(), lr=best_lr, momentum=0.9, weight_decay=best_weight_decay,
                                   nesterov=best_nesterov)
    best_recon_criterion = nn.MSELoss()

    # Load the state dict of the best model found during the hyperparameter search
    best_model.load_state_dict(best_model_state_dict)

    # Save the best model, its configuration parameters, and its weights
    torch.save({
        "model_state_dict": best_model_state_dict,
        "hidden_dim": best_hidden_dim,
        "lr": best_lr,
        "weight_decay": best_weight_decay,
        "temperature": best_temperature,
        "optimizer": best_optimizer,
        "nesterov": best_nesterov
    }, "best_model.pth")

if __name__ == '__main__':
    main()