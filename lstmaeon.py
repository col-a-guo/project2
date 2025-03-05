import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import itertools  # For hyperparameter combinations

# Load the data
X, y = load_classification("TwoPatterns")
print("Shape of X = ", X.shape)
print("Shape of y = ", y.shape)


# Preprocess data
def preprocess_data(X, y, window_size, test_size=0.2, random_state=42):
    """
    Preprocesses the time series data for LSTM training with windowing.

    Args:
        X: Input time series data (aeon format).
        y: Target labels.
        window_size: Size of the sliding window.
        test_size: Proportion of data for testing.
        random_state: Random seed for splitting.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test) as PyTorch Tensors.
    """
    # Aeon data format is (n_samples, n_channels, time_series_length)
    # Convert to numpy array
    X = X.astype(np.float32)

    # Apply windowing
    X_windowed = []
    for series in X:
        series_length = series.shape[1]  # Get the actual series length
        if window_size < series_length:
            # Truncate if window size is smaller
            start_index = series_length - window_size  # Take the last 'window_size' elements
            truncated_series = series[:, start_index:]
            X_windowed.append(truncated_series)
        elif window_size > series_length:
            # Pad if window size is larger
            padding_size = window_size - series_length
            padded_series = np.pad(series, ((0, 0), (padding_size, 0)), mode='constant')  # Pad at the beginning
            X_windowed.append(padded_series)
        else:
            # No change if window size is equal
            X_windowed.append(series)

    X_windowed = np.array(X_windowed)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_windowed, y, test_size=test_size, random_state=42)

    # Convert labels to numerical values using LabelEncoder
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)  # Use long for labels
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test, len(label_encoder.classes_)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
      # Set initial states
      h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
      c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

      # Forward propagate LSTM
      out, _ = self.lstm(x, (h0, c0))

      # Decode the hidden state of the last time step
      out = self.fc(out[:, -1, :])  # Taking the last time step

      return out

# Training loop function (to avoid code duplication)
def train_and_evaluate(model, optimizer, criterion, train_loader, test_loader, device, epochs=50, patience=10):  # Reduced epochs for quicker testing
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

    # Evaluation
    model.eval()
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

# Hyperparameter search space - LSTM specific params
hidden_size = 100
optimizer_name = 'adam'  # Keeping these fixed
dropout = 0.2

num_layers_options = [1, 2, 3, 4]
window_size_options = [64, 128]  # Example window sizes - adjust as needed


# Hyperparameter Search Loop
best_accuracy = 0
best_hyperparameters = None
all_results = []

# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(num_layers_options, window_size_options))

for num_layers, window_size in param_combinations:
    print(f"Training with num_layers={num_layers}, window_size={window_size}")
    batch_size=32

    # Preprocess the data with the current window size
    X_train, X_test, y_train, y_test, num_classes = preprocess_data(X, y, window_size)


    # Instantiate the model with the current hyperparameters
    input_size = X_train.shape[2]  #  time series length after windowing
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes, dropout)
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
    accuracy, train_losses, val_losses = train_and_evaluate(model, optimizer, criterion, train_loader, test_loader, device)

    # Store results
    results = {
        "num_layers": num_layers,
        "window_size": window_size,
        "accuracy": accuracy,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    all_results.append(results)

    # Update best accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_hyperparameters = {"num_layers": num_layers, "window_size": window_size}

print("Hyperparameter Search Complete")
print(f"Best Hyperparameters: {best_hyperparameters}, Best Accuracy: {best_accuracy:.2f}%")

# Plotting (Example - you might want to plot multiple results)
# Choose which run to plot - e.g., the best one
best_result = next(item for item in all_results if item["num_layers"] == best_hyperparameters["num_layers"] and item["window_size"] == best_hyperparameters["window_size"])
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