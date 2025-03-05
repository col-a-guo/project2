import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import itertools  # For hyperparameter combinations
from sklearn.metrics import classification_report


# Load the data
X, y = load_classification("TwoPatterns")
print("Shape of X = ", X.shape)
print("Shape of y = ", y.shape)


# Preprocess data
def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Preprocesses the time series data for CNN training.

    Args:
        X: Input time series data (aeon format).
        y: Target labels.
        test_size: Proportion of data for testing.
        random_state: Random seed for splitting.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test) as PyTorch Tensors.
    """
    # Aeon data format is (n_samples, n_channels, time_series_length)
    # Convert to numpy array
    X = X.astype(np.float32)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Convert labels to numerical values using LabelEncoder
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)


    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)  # Use long for labels
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test, len(label_encoder.classes_), label_encoder.classes_


# Preprocess the data
X_train, X_test, y_train, y_test, num_classes, class_names = preprocess_data(X, y)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes, num_layers, kernel_size, neuron_multiplier):
        super(CNN, self).__init__()

        self.layers = nn.ModuleList()  # Store layers in a ModuleList

        # Determine the starting number of neurons for the first layer
        first_layer_neurons = 4 if neuron_multiplier == 8 else 8 if neuron_multiplier == 4 else 16

        # Input channel is always 1
        in_channels = 1
        num_neurons = first_layer_neurons

        for i in range(num_layers):
            self.layers.append(nn.Conv1d(in_channels, num_neurons, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)) #Added kernel_size as a parameter
            self.layers.append(nn.BatchNorm1d(num_neurons))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = num_neurons
            num_neurons *= neuron_multiplier


        # Adaptive average pooling to handle variable sequence lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer.  Important: Calculate input size based on last conv layer
        self.fc = nn.Linear(in_channels, num_classes)


    def forward(self, x):
        # x shape: (N, 1, 284)
        for layer in self.layers:
            x = layer(x)

        # Adaptive pooling
        x = self.adaptive_pool(x) # Output: (N, num_neurons_last_layer, 1)

        # Flatten the tensor
        x = torch.flatten(x, 1)

        # Fully connected layer
        x = self.fc(x)
        return x


# Hyperparameter search space
num_layers_options = [4]
kernel_size_options = [4]
neuron_multiplier_options = [2]
optimizer_options = ['adam', 'sgd']

# Training loop function (to avoid code duplication)
def train_and_evaluate(model, optimizer, criterion, train_loader, test_loader, device, class_names, epochs=50, patience=10):  # Reduced epochs for quicker testing
    best_loss = float('inf')
    counter = 0
    train_losses = []
    val_losses = []
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)  # Move model to GPU if available

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in enumerate(test_loader):  #enumerate was causing error with test loader
                inputs, labels = test_loader.dataset[inputs][0].unsqueeze(0).to(device), test_loader.dataset[inputs][1].unsqueeze(0).to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    print("Finished Training")

    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluation and Classification Report
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=class_names))  # Output Classification Report


    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {accuracy:.2f}%")

    return accuracy, train_losses, val_losses



# Hyperparameter Search Loop
best_accuracy = 0
best_hyperparameters = None
all_results = []

# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(num_layers_options, kernel_size_options, neuron_multiplier_options, optimizer_options))

for num_layers, kernel_size, neuron_multiplier, optimizer_name in param_combinations:
    print(f"Training with num_layers={num_layers}, kernel_size={kernel_size}, neuron_multiplier={neuron_multiplier}, optimizer={optimizer_name}")
    batch_size=32

    # Instantiate the model with the current hyperparameters
    model = CNN(num_classes, num_layers, kernel_size, neuron_multiplier) #Passing in kernel_size
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)  # Move model to GPU if available

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.005)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Data loaders (assuming they are already created)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train and evaluate the model
    accuracy, train_losses, val_losses = train_and_evaluate(model, optimizer, criterion, train_loader, test_loader, device, class_names)

    # Store results
    results = {
        "num_layers": num_layers,
        "kernel_size": kernel_size,
        "neuron_multiplier": neuron_multiplier,
        "optimizer": optimizer_name,
        "accuracy": accuracy,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    all_results.append(results)


    # Update best accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_hyperparameters = {"num_layers": num_layers, "kernel_size": kernel_size, "neuron_multiplier": neuron_multiplier, "optimizer": optimizer_name}


print("Hyperparameter Search Complete")
print(f"Best Hyperparameters: {best_hyperparameters}, Best Accuracy: {best_accuracy:.2f}%")


# Plotting (Example - you might want to plot multiple results)
# Choose which run to plot - e.g., the best one
best_result = next(item for item in all_results if item["num_layers"] == best_hyperparameters["num_layers"] and item["neuron_multiplier"] == best_hyperparameters["neuron_multiplier"]  and item["kernel_size"] == best_hyperparameters["kernel_size"]and item["optimizer"] == best_hyperparameters["optimizer"])
# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(best_result["train_losses"], label='Training Loss')
plt.plot(best_result["val_losses"], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (Best Model)')
plt.legend()
plt.grid(True)
plt.savefig('best_model_loss_plot.png')  # Save the plot to a file
plt.show()