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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_X, train_y = load_from_tsfile_to_dataframe("IEEEPPG_TRAIN.ts")
test_X, test_y = load_from_tsfile_to_dataframe("IEEEPPG_TEST.ts")

train_y = train_y.astype(np.float64)
test_y = test_y.astype(np.float64)

param_grid = {
    'num_filters': [16, 32],
    'kernel_size': [(1, 3), (3, 3)],
    'optimizer': ['Adam'],
    'learning_rate': [0.001],
    'padding': ['same'],
    'stride': [1],
    'num_layers': [1, 2],
    'neurons_per_layer': [32, 64]
}

param_combinations = list(product(*param_grid.values()))
param_names = list(param_grid.keys())

def preprocess_data(X):
    X_flat = np.array([series.values for series in X[X.columns[0]]])
    # (num_samples, 200, 5)

    # (batch_size, channels, height, width) for 2D CNN
    X_reshaped = X_flat.reshape(X_flat.shape[0], 5, 1, 200)  # Correct Reshape

    return X_reshaped, None  #No scaling needed


# 3. Define the 2D CNN Model in PyTorch
class CNN2DModel(nn.Module):
    def __init__(self, num_filters, kernel_size, padding, stride, num_layers, neurons_per_layer):
        super(CNN2DModel, self).__init__()
        self.layers = nn.ModuleList()

        # Input channels for the first layer
        in_channels = 5  # Changed to 5

        for i in range(num_layers):
            self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size, padding=padding, stride=stride))
            self.layers.append(nn.BatchNorm2d(num_filters))  # Batch Normalization
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(kernel_size=(1, 2)))  # Pooling only along the width
            in_channels = num_filters  # Output channels for the next layer


        self.flatten = nn.Flatten()
        # Dynamically calculate fc1 input size
        # Assuming input size after conv/pool layers is (num_filters, height, width)
        # height = 1 (no change)
        # width = (input_width - kernel_size + 2 * padding) / stride + 1  (after conv)
        # width = width / 2 (after pooling)
        input_width = 200 #Original width of the input data
        for i in range(num_layers):
            #Convolutional Layer Formula For "same" padding, the output size is the same as input size
            conv_output_width = input_width
            
            # Calculate the output size after pooling layer
            pooled_output_width = conv_output_width / 2 #pooling happens along the width, kernel_size=(1,2)
            
            #Update input_width for next layer
            input_width = pooled_output_width

        fc_input_size = num_filters * 1 * int(input_width)


        self.fc1 = nn.Linear(fc_input_size, neurons_per_layer)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(neurons_per_layer, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x



def plot_lift_chart(y_true, y_pred, window_size=10):  #Added default window size
    """
    Generates a lift chart with a moving average for smoothing.

    Args:
        y_true (np.ndarray): Array of actual target values.
        y_pred (np.ndarray): Array of predicted target values.
        window_size (int): Size of the moving average window.  Adjust as needed.
    """
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df = df.sort_values(['y_pred', 'y_true'], ascending=[False, False])  # Tie-breaking sort

    df['cumulative_actual'] = df['y_true'].cumsum()
    df['cumulative_index'] = np.arange(1, len(df) + 1)
    df['cumulative_random'] = df['cumulative_index'] * df['y_true'].sum() / len(df)

    df['lift'] = df['cumulative_actual'] / df['cumulative_random']

    # Calculate Moving Average
    df['smoothed_lift'] = df['lift'].rolling(window=window_size, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(df['cumulative_index'], df['smoothed_lift'], label='Smoothed Lift')
    #Optional: Plot the original lift for comparison
    #plt.plot(df['cumulative_index'], df['lift'], label='Original Lift', alpha=0.3)

    plt.xlabel('Number of Predictions')
    plt.ylabel('Lift')
    plt.title('Lift Chart with Moving Average')
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
    num_filters, kernel_size, optimizer_name, learning_rate, padding, stride, num_layers, neurons_per_layer = params

    print(f"Starting training with parameter combination {i+1}/{len(param_combinations)}:")
    for name, value in zip(param_names, params):
        print(f"\t{name}: {value}")

    # Preprocess data
    train_X_processed, _ = preprocess_data(train_X) #No scaler returned
    test_X_processed, _ = preprocess_data(test_X)  #No scaler returned

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
    model = CNN2DModel(num_filters, kernel_size, padding, stride, num_layers, neurons_per_layer).to(device)

    # Define optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

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

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust max_norm as needed

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