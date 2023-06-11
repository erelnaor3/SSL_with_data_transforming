import csv
import os
from sklearn.svm import SVC
from auto_encoder_model import *

from two_class_data_generation import *


def train(model, train_data, train_labels):
    # Fit the SVM model to the training data
    model.fit(train_data, train_labels)
    return model


def validate(model, val_data ,val_labels):
    # Evaluate the SVM model on the validation data
    val_acc = model.score(val_data, val_labels)
    return val_acc


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction_by_run = []
    n_trials = 100
    noise = "mask_noise"
    for i in range(n_trials):
        # Generate data
        n_samples = 1000
        d = 5
        d_noise = 1
        data, labels = get_data(n_samples=n_samples, d=d, d_noise=d_noise)
        labels = labels
        saved_data = torch.load('models/encoded_model_and_data_mask_noise.pt')
        auto_encoder_model = Autoencoder(data.shape[1], saved_data['hidden_dim'])
        # Load the saved model state_dict
        auto_encoder_model.load_state_dict(saved_data['state_dict'])
        data = torch.Tensor(data)
        labels = torch.LongTensor(labels)
        auto_encoder_model.eval()
        data = auto_encoder_model.encoder(data).detach()

        # Create dataset and split into train and validation sets
        valid_size = 0.3
        train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=valid_size,
                                                                          shuffle=True)
        # Set up model and training parameters
        classifier_model = SVC(kernel='linear', C=1, gamma='scale')

        # Train and validate
        classifier_model = train(classifier_model, train_data,train_labels)
        val_acc = validate(classifier_model, val_data ,val_labels)
        prediction_by_run.append(val_acc)

        # Print validation accuracy
        print(f'Validation Accuracy: {val_acc:.4f}')

    csv_path = 'results/latent_classifier_svm_' + noise + '.csv'
    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_path)
    mean_accuracy = np.mean(prediction_by_run)
    print(mean_accuracy)
    max_accuracy = np.max(prediction_by_run)
    min_accuracy = np.min(prediction_by_run)
    # Open the CSV file in append mode
    with open(csv_path, 'a', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)

        # Write the header row if the file is new
        if not file_exists:
            writer.writerow(['Mean Accuracy', 'Max Accuracy', 'Min Accuracy'])

        # Write the results row
        writer.writerow([mean_accuracy, max_accuracy, min_accuracy])

    # Print the mean, max, and min prediction
    print(f"Mean accuracy: {mean_accuracy:.2f}%")
    print(f"Max accuracy: {max_accuracy:.2f}%")
    print(f"Min accuracy: {min_accuracy:.2f}%")

if __name__ == '__main__':
    main()