import pandas as pd
import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe
from sklearn.preprocessing import StandardScaler  # Changed to StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product
from prettytable import PrettyTable

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load the data using sktime
train_X, train_y = load_from_tsfile_to_dataframe("FloodModeling1_TRAIN.ts")
test_X, test_y = load_from_tsfile_to_dataframe("FloodModeling1_TEST.ts")

# Explicitly convert to numeric type (float64 is a good choice)
train_y = train_y.astype(np.float64)
test_y = test_y.astype(np.float64)

# 2. Hyperparameter Grid
param_grid = {
    'hidden_size': [25, 50],  # Neuron count
    'num_layers': [1, 2],  # Number of LSTM layers
    'optimizer': ['Adam', 'SGD'],  # Optimizer
    'window_size': [133, 266] # Window Size
}

# Generate all possible combinations of hyperparameters
param_combinations = list(product(*param_grid.values()))
param_names = list(param_grid.keys())


# Function to preprocess data with variable window size
def preprocess_data(X, window_size):
    X_flat = np.array([series.values.flatten() for series in X[X.columns[0]]])
    scaler = StandardScaler()  # Changed to StandardScaler
    X_scaled = scaler.fit_transform(X_flat)

    # Handle cases where window_size is larger than the series length
    series_length = X[X.columns[0]].iloc[0].shape[0]
    if window_size > series_length:
        window_size = series_length
        print(f"Warning: window_size exceeds series length.  Using window_size = {window_size}")

    X_reshaped = X_scaled.reshape(X_scaled.shape[0], window_size, X_scaled.shape[1] // window_size)  # Reshape for LSTM

    return X_reshaped, scaler


# 3. Define the LSTM Model in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)  # Added dropout
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Move to correct device
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Move to correct device

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# 4. Hyperparameter Tuning Loop
best_mse = float('inf')
best_params = None

# Create a table to store the results
table = PrettyTable()
table.field_names = param_names + ["MSE"]

results = [] #List to store results for the final table

for i, params in enumerate(param_combinations):
    # Unpack hyperparameters
    hidden_size, num_layers, optimizer_name, window_size = params

    print(f"Starting training with parameter combination {i+1}/{len(param_combinations)}:")
    for name, value in zip(param_names, params):
        print(f"\t{name}: {value}")

    # Preprocess data with the current window size
    train_X_processed, train_scaler = preprocess_data(train_X, window_size)
    test_X_processed, test_scaler = preprocess_data(test_X, window_size)

    # Scale the target variables
    target_scaler = StandardScaler()  # Changed to StandardScaler
    train_y_scaled = target_scaler.fit_transform(train_y.reshape(-1, 1))
    test_y_scaled = target_scaler.transform(test_y.reshape(-1, 1))

    # Convert numpy arrays to torch tensors
    train_X_tensor = torch.tensor(train_X_processed, dtype=torch.float32).to(device)
    test_X_tensor = torch.tensor(test_X_processed, dtype=torch.float32).to(device)
    train_y_tensor = torch.tensor(train_y_scaled, dtype=torch.float32).to(device)
    test_y_tensor = torch.tensor(test_y_scaled, dtype=torch.float32).to(device)

    # Create the model
    input_size = train_X_tensor.shape[2]  # Features
    output_size = 1
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

    # Define optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01)

    criterion = nn.MSELoss()

    # DataLoader
    batch_size = 32
    train_dataset = torch.utils.data.TensorDataset(train_X_tensor, train_y_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop with early stopping
    num_epochs = 20  # Reduced epochs for tuning
    patience = 5
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation step (using test set for simplicity in this example)
        model.eval()
        with torch.no_grad():
            val_outputs = model(test_X_tensor)
            val_loss = criterion(val_outputs, test_y_tensor)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break


    # 5. Evaluate and Store Results
    model.eval()
    with torch.no_grad():
        predicted_scaled = model(test_X_tensor)
        predicted_scaled_cpu = predicted_scaled.cpu().numpy()
        test_y_cpu = test_y_tensor.cpu().numpy()

    predicted = target_scaler.inverse_transform(predicted_scaled_cpu)
    test_y_original = target_scaler.inverse_transform(test_y_cpu)

    mse = mean_squared_error(test_y, predicted)

    print(f"MSE: {mse:.4f}")

    #Add the results to the table
    results.append(list(params) + [mse])

    if mse < best_mse:
        best_mse = mse
        best_params = params
        print(f"New best MSE: {best_mse:.4f} with parameters: {best_params}")

print("\nHyperparameter Tuning Complete!")
print(f"Best MSE: {best_mse:.4f}")
print(f"Best Parameters: {best_params}")


#Add rows to the PrettyTable
for row in results:
    table.add_row(row)

print(table) #Print the full table