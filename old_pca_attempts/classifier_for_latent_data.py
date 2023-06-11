import csv
import os
from torch.utils.data import Subset
from auto_encoder_model import *
from two_class_data_generation import *
from sklearn.metrics import roc_auc_score


class FullyConnectedClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FullyConnectedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten the input tensor
        out = self.fc1(x)
        out = self.softmax(out)
        return out


def train(model, optimizer, loss_fn, train_data, device):
    # Set the model to training mode
    model.train()
    # Initialize the loss and accuracy
    total_loss = 0.0
    total_correct = 0
    # Iterate over the batches of training data
    for input, target in train_data:
        # Move the input and target tensors to the device
        input = input.to(device)
        target = target.to(device)
        # Clear the gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(input)
        # Compute the loss
        loss = loss_fn(output, target)
        # Backward pass
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Update the total loss and accuracy
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_correct += (predicted == target).sum().item()

    # Compute the average loss and accuracy
    avg_loss = total_loss / len(train_data.dataset)
    avg_acc = total_correct / len(train_data.dataset)

    return avg_loss, avg_acc


def validate(model, loss_fn, val_loader, device):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output_labels = model(data)
            val_loss += loss_fn(output_labels, target).item()
            _, predicted = torch.max(output_labels.data, 1)
            correct += (predicted == target).sum().item()
    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    return val_loss, accuracy


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction_by_run = []
    auc_roc_by_class = {0: [], 1: []}
    n_trials = 100
    noise = "pca_noise"
    for i in range(n_trials):
        # print("train + test i")
        # Generate data
        n_samples = 1000
        d = 5
        d_noise = 20
        data,labels = get_data(n_samples = n_samples,d = d,d_noise = d_noise,PCA_noise=False)
        labels = labels
        saved_data = torch.load('encoded_model_and_data_pca_noise.pt')
        auto_encoder_model = Autoencoder(data.shape[1], saved_data['hidden_dim'])
        auto_encoder_model.load_state_dict(saved_data['state_dict'])
        data = torch.Tensor(data)
        labels = torch.LongTensor(labels)
        data = auto_encoder_model.get_encoded_data(data).detach()

        # Create dataset and split into train and validation sets
        dataset = torch.utils.data.TensorDataset(data, labels)
        valid_size = 0.3
        train_idx, valid_idx = train_test_split(range(len(labels)), test_size=valid_size, shuffle=True)

        train_set = Subset(dataset, train_idx)
        valid_set = Subset(dataset, valid_idx)

        # Create data loaders
        batch_size = 128
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

        # Set up model, loss function, optimizer, and training parameters
        learning_rate = 0.001
        epochs = 20
        input_dim = data.shape[1]
        output_dim = 2  # number of classes
        clasifier_model = FullyConnectedClassifier(input_dim, output_dim).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(clasifier_model.parameters(), lr=learning_rate)
        train_losses = []
        val_losses = []

        # Train and validate
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train(clasifier_model, optimizer, loss_fn, train_loader, device)
            val_loss, val_acc = validate(clasifier_model, loss_fn, valid_loader, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if epoch % 5 == 0:
                print(f'Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}')

        # Evaluate on validation data after training
        clasifier_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            pred_probs = []
            targets = []
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                total += target.size(0)
                output = clasifier_model(data)
                pred_prob = nn.functional.softmax(output, dim=1)
                pred_probs.append(pred_prob.cpu().numpy())
                targets.append(target.cpu().numpy())
                pred = torch.argmax(output, dim=1)
                correct += (pred == target).sum().item()

            print('Final validation accuracy: {:.2f}%'.format(100 * correct / total))
            prediction_by_run.append(100 * correct / total)
            # Compute AUC-ROC for each class
            pred_probs = np.concatenate(pred_probs, axis=0)
            targets = np.concatenate(targets, axis=0)
            for c in [0, 1]:
                class_auc = roc_auc_score(targets == c, pred_probs[:, c])
                auc_roc_by_class[c].append(class_auc)

            # Plot train and validation losses
            plt.plot(range(1, epochs + 1), train_losses, label='Train')
            plt.plot(range(1, epochs + 1), val_losses, label='Validation')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Losses')
            plt.legend()
            # plt.show()
            plt.close()
    # Define the path of the CSV file
    csv_path = 'latent_classifier_' + noise + '.csv'
    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_path)

    mean_accuracy = np.mean(prediction_by_run)
    max_accuracy = np.max(prediction_by_run)
    min_accuracy = np.min(prediction_by_run)
    all_values_auc_roc = [val for sublist in auc_roc_by_class.values() for val in sublist]
    mean_auc_roc = np.mean(all_values_auc_roc)
    max_auc_roc = np.max(all_values_auc_roc)
    min_auc_roc = np.min(all_values_auc_roc)

    # Open the CSV file in append mode
    with open(csv_path, 'a', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)

        # Write the header row if the file is new
        if not file_exists:
            writer.writerow(
                ['Mean Accuracy', 'Max Accuracy', 'Min Accuracy', 'Mean AUC-ROC', 'Max AUC-ROC', 'Min AUC-ROC'])

        # Write the results row
        writer.writerow([mean_accuracy, max_accuracy, min_accuracy, mean_auc_roc, max_auc_roc, min_auc_roc])

    # Print the mean, max, and min prediction
    print(f"Mean accuracy: {mean_accuracy:.2f}%")
    print(f"Max accuracy: {max_accuracy:.2f}%")
    print(f"Min accuracy: {min_accuracy:.2f}%")
    print(f"Mean AUC-ROC: {mean_auc_roc:.2f}%")
    print(f"Max AUC-ROC: {max_auc_roc:.2f}%")
    print(f"Min AUC-ROC: {min_auc_roc:.2f}%")


if __name__ == '__main__':
    main()


