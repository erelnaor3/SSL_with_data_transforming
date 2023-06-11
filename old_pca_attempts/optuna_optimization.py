import optuna
from optuna.samplers import TPESampler
from auto_encoder_model import *

def objective(trial):
    # Define the range of hyperparameters to search over
    latent_dim = trial.suggest_int('latent_dim', 20, 100)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)

    # Create the model
    model = Autoencoder(input_dim, latent_dim=latent_dim)
    model = model.to(device)

    # Define the optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction='mean')

    # Train the model
    train_loss_history, valid_loss_history = train(model, optimizer, criterion, train_loader, valid_loader, device)

    # Compute the validation loss
    val_loss = valid_loss_history[-1]

    # Return the validation loss
    return val_loss


if __name__ == "__main__":
    # Define the search space for hyperparameters
    sampler = TPESampler(seed=0)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=10)

    # Print the best hyperparameters and the best validation loss
    print("Best trial:")
    trial = study.best_trial
    print(f"  Loss: {trial.value:.3f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")