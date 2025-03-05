import pandas as pd
import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from itertools import product
from prettytable import PrettyTable
import matplotlib.pyplot as plt

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
    'num_filters': [16, 32],
    'kernel_size': [3, 5],
    'optimizer': ['Adam', 'SGD'],
    'padding': ['same'],
    'stride': [1],
    'window_size': [266]  # Fixed to 266
}

# Generate all possible combinations of hyperparameters
param_combinations = list(product(*param_grid.values()))
param_names = list(param_grid.keys())

# Function to preprocess data with a fixed window size of 266
def preprocess_data(X):
    X_flat = np.array([series.values.flatten() for series in X[X.columns[0]]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    # Pad or Truncate the input to 266
    series_length = X_scaled.shape[1]
    if series_length < 266:
        # Pad with zeros if shorter
        padding_length = 266 - series_length
        X_padded = np.pad(X_scaled, ((0, 0), (0, padding_length)), mode='constant')
        X_reshaped = X_padded.reshape(X_padded.shape[0], 1, 266)
    elif series_length > 266:
        # Truncate if longer
        X_truncated = X_scaled[:, :266]
        X_reshaped = X_truncated.reshape(X_truncated.shape[0], 1, 266)
    else:
        # No change if already 266
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, 266)

    return X_reshaped, scaler


# 3. Define the 1D CNN Model in PyTorch
class CNN1DModel(nn.Module):
    def __init__(self, num_filters, kernel_size, padding, stride):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_filters * 133, 64) #Dynamically calculating fc1 input
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x


# Function to create lift chart
def plot_lift_chart(y_true, y_pred):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df = df.sort_values('y_pred', ascending=False)
    df['cumulative_actual'] = df['y_true'].cumsum()
    df['cumulative_index'] = np.arange(1, len(df) + 1)
    df['cumulative_random'] = df['cumulative_index'] * df['y_true'].sum() / len(df)

    plt.figure(figsize=(10, 6))
    plt.plot(df['cumulative_index'], df['cumulative_actual'], label='Model')
    plt.plot(df['cumulative_index'], df['cumulative_random'], label='Random')
    plt.xlabel('Number of Predictions')
    plt.ylabel('Cumulative Actual Value')
    plt.title('Lift Chart')
    plt.legend()
    plt.grid(True)
    plt.show()

# 4. Hyperparameter Tuning Loop
best_mse = float('inf')
best_params = None
best_model = None
test_y_original_best = None
predicted_best = None


# Create a table to store the results
table = PrettyTable()
table.field_names = param_names + ["MSE", "RMSE"]

results = [] #List to store results for the final table


for i, params in enumerate(param_combinations):
    # Unpack hyperparameters
    num_filters, kernel_size, optimizer_name, padding, stride, window_size = params

    print(f"Starting training with parameter combination {i+1}/{len(param_combinations)}:")
    for name, value in zip(param_names, params):
        print(f"\t{name}: {value}")

    # Preprocess data
    train_X_processed, train_scaler = preprocess_data(train_X)
    test_X_processed, test_scaler = preprocess_data(test_X)

    # Scale the target variables
    target_scaler = StandardScaler()
    train_y_scaled = target_scaler.fit_transform(train_y.reshape(-1, 1))
    test_y_scaled = target_scaler.transform(test_y.reshape(-1, 1))

    # Convert numpy arrays to torch tensors
    train_X_tensor = torch.tensor(train_X_processed, dtype=torch.float32).to(device)
    test_X_tensor = torch.tensor(test_X_processed, dtype=torch.float32).to(device)
    train_y_tensor = torch.tensor(train_y_scaled, dtype=torch.float32).to(device)
    test_y_tensor = torch.tensor(test_y_scaled, dtype=torch.float32).to(device)

    # Create the model
    model = CNN1DModel(num_filters, kernel_size, padding, stride).to(device)

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
    num_epochs = 20
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

        # Validation step
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
    rmse = math.sqrt(mse)



    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}")

    #Add the results to the table
    results.append(list(params) + [mse, rmse])

    if mse < best_mse:
        best_mse = mse
        best_rmse = rmse
        best_params = params
        best_model = model # Store the model
        test_y_original_best = test_y_original #Store the result
        predicted_best = predicted

        print(f"New best MSE: {best_mse:.4f}, RMSE: {best_rmse:.4f} with parameters: {best_params}")

print("\nHyperparameter Tuning Complete!")
print(f"Best MSE: {best_mse:.4f}")
print(f"Best RMSE: {best_rmse:.4f}")
print(f"Best Parameters: {best_params}")

# Plot lift chart for the best model
if best_model is not None:
    plot_lift_chart(test_y, predicted_best.flatten()) #Flatten to make it 1d
else:
    print("No model trained.")


#Add rows to the PrettyTable
for row in results:
    table.add_row(row)

print(table) #Print the full table