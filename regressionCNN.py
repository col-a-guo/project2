import pandas as pd
import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_X, train_y = load_from_tsfile_to_dataframe("IEEEPPG_TRAIN.ts")
test_X, test_y = load_from_tsfile_to_dataframe("IEEEPPG_TEST.ts")

# Explicitly convert to numeric type (float64 is a good choice)
train_y = train_y.astype(np.float64)
test_y = test_y.astype(np.float64)


# Function to preprocess data with fixed window size
def preprocess_data(X, window_size):
    X_flat = np.array([series.values.flatten() for series in X[X.columns[0]]])
    scaler = MinMaxScaler()
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
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):  #Added dropout_rate as a parameter
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout_rate = dropout_rate  #Store for later


    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Move to correct device
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Move to correct device

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# 4. Locked-in Hyperparameters
best_params = (32, 1, 'Adam', 100)  # (hidden_size, num_layers, optimizer, window_size)
dropout_rate = 0.5 #Experiment with higher dropout

# Preprocess data with the best window size
train_X_processed, train_scaler = preprocess_data(train_X, best_params[3])
test_X_processed, test_scaler = preprocess_data(test_X, best_params[3])

# Scale the target variables
target_scaler = MinMaxScaler()
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
model = LSTMModel(input_size, best_params[0], best_params[1], output_size, dropout_rate=dropout_rate).to(device) #Pass the dropout_rate

# Define optimizer
if best_params[2] == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=0.001)
else:
    optimizer = optim.SGD(model.parameters(), lr=0.01)

criterion = nn.MSELoss()

# DataLoader
batch_size = 32
train_dataset = torch.utils.data.TensorDataset(train_X_tensor, train_y_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop with early stopping for final training
num_epochs = 50 #Back to proper length
patience = 10
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
r2 = r2_score(test_y,predicted)

print(f"Test MSE (Original Scale): {mse:.4f}")
print(f"Test R^2 Score: {r2:.4f}")

# Print some predictions with X values
print("Sample Predictions")
for i in range(min(10, len(predicted))):  # Print first 10
    #Get the input X for this sample
    input_x = test_X_processed[i]

    print(f"Predicted: {predicted[i][0]}, Actual: {test_y[i]}")


# 6. Lift Chart
# Sort the actual values and corresponding predictions
df_lift = pd.DataFrame({'actual': test_y.flatten(), 'predicted': predicted.flatten()}) #Flatten to avoid shape issues
df_lift = df_lift.sort_values(by=['actual'], ascending=False)

# Calculate cumulative gains
df_lift['cumulative_actual'] = df_lift['actual'].cumsum()
df_lift['cumulative_predicted'] = df_lift['predicted'].cumsum()
df_lift['cumulative_index'] = np.arange(1, len(df_lift) + 1)

# Calculate the baseline (random model)
total_actual = df_lift['actual'].sum()
df_lift['cumulative_baseline'] = (df_lift['cumulative_index'] / len(df_lift)) * total_actual

# Plot the lift chart
plt.figure(figsize=(10, 6))
sns.lineplot(x='cumulative_index', y='cumulative_actual', data=df_lift, label='Actual')
sns.lineplot(x='cumulative_index', y='cumulative_predicted', data=df_lift, label='Predicted')
sns.lineplot(x='cumulative_index', y='cumulative_baseline', data=df_lift, label='Baseline')

plt.xlabel('Number of Samples')
plt.ylabel('Cumulative Target Value')
plt.title('Lift Chart')
plt.legend()
plt.grid(True)
plt.show()