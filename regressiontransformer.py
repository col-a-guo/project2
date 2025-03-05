import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Import for plotting

from sktime.datasets import load_from_tsfile_to_dataframe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    train_X, train_y = load_from_tsfile_to_dataframe("IEEEPPG_TRAIN.ts")
    test_X, test_y = load_from_tsfile_to_dataframe("IEEEPPG_TEST.ts")
except FileNotFoundError:
    print("Ensure IEEEPPG_TRAIN.ts and IEEEPPG_TEST.ts are in the same directory or provide their full paths.")
    exit()


def preprocess_data(train_X, train_y, test_X, test_y):
    """
    Preprocesses the time series data, converts to numpy arrays,
    pads sequences to the same length, and standardizes the data.

    Args:
        train_X: Training data (pandas DataFrame).
        train_y: Training labels (pandas Series).
        test_X: Testing data (pandas DataFrame).
        test_y: Testing labels (pandas Series).

    Returns:
        train_X_padded, train_y_scaled, test_X_padded, test_y_scaled, max_len
    """

    train_X_np = [np.array(series).reshape(5, 200).transpose(1, 0) for series in train_X.iloc[:, 0]]  # Reshape to (200, 5)
    test_X_np = [np.array(series).reshape(5, 200).transpose(1, 0) for series in test_X.iloc[:, 0]]  # Reshape to (200, 5)

    # Find the maximum sequence length for padding
    max_len = max(max(len(seq) for seq in train_X_np), max(len(seq) for seq in test_X_np))

    # Pad sequences to the maximum length. Pad the front as well.
    train_X_padded = [np.pad(seq, ((0, max_len - len(seq)), (0,0)), 'constant') for seq in train_X_np]
    test_X_padded = [np.pad(seq, ((0, max_len - len(seq)), (0,0)), 'constant') for seq in test_X_np]


    train_X_padded = np.array(train_X_padded)
    test_X_padded = np.array(test_X_padded)

    # Scale the target variables (important for regression)
    scaler = StandardScaler()
    train_y_scaled = scaler.fit_transform(train_y.reshape(-1, 1)).flatten()
    test_y_scaled = scaler.transform(test_y.reshape(-1, 1)).flatten()

    return train_X_padded, train_y_scaled, test_X_padded, test_y_scaled, max_len, scaler # return scaler

train_X_padded, train_y_scaled, test_X_padded, test_y_scaled, max_len, target_scaler = preprocess_data(train_X, train_y, test_X, test_y)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)  # Use Float for regression

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 32
train_dataset = TimeSeriesDataset(train_X_padded, train_y_scaled)

#val set
train_X_split, val_X_split, train_y_split, val_y_split = train_test_split(
    train_X_padded, train_y_scaled, test_size=0.2, random_state=42
)

val_dataset = TimeSeriesDataset(val_X_split, val_y_split)

test_dataset = TimeSeriesDataset(test_X_padded, test_y_scaled)  # Use the *preprocessed* test data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # No need to shuffle validation data
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class TimeSeriesTransformerRegression(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, 1)  # Output layer for regression (single value)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)


    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        # Embedding
        x = self.input_embedding(x) # (batch_size, sequence_length, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer Encoder expects (sequence_length, batch_size, d_model)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)

        # Take the mean over the sequence length
        x = x.mean(dim=0)

        # Fully connected layer for regression
        x = self.fc(x)  # (batch_size, 1)
        return x.squeeze() #remove the extra dimension

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

input_size = 5  # Input size is now 5 (height)
d_model = 64 # Reduced d_model for faster training and potentially better generalization
nhead = 4
num_layers = 2

model = TimeSeriesTransformerRegression(input_size=input_size, d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)  # Instantiate model
criterion = nn.MSELoss()  # MSE for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)  

num_epochs = 50
patience = 5  # Number of epochs to wait for improvement
best_val_loss = float('inf')
epochs_no_improve = 0
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)


    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

    # Early Stopping Check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()  # Save the best model state
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print("Early stopping triggered!")
            model.load_state_dict(best_model_state)  # Load the best model
            break


if epochs_no_improve > 0:
    model.load_state_dict(best_model_state)


model.eval()
total_loss = 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = criterion(output, y)
        total_loss += loss.item()

avg_loss = total_loss / len(test_loader)
print(f"Average MSE Loss on the test set: {avg_loss}")

def predict(model, data, device, scaler):
    """
    Predicts the target variable for new time series data and inverse transforms the predictions.

    Args:
        model: Trained PyTorch model.
        data: New time series data (numpy array).
        device: Device to use for prediction.
        scaler: StandardScaler object used for scaling the target variable.

    Returns:
        predictions: List of predicted target values (unscaled).
    """
    model.eval()  # Set the model to evaluation mode
    data = torch.tensor(data, dtype=torch.float32).to(device) # Move to device
    with torch.no_grad():
        output = model(data)
    predictions_scaled = output.cpu().numpy()
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    return predictions

# Example Usage (assuming you have new_data ready):
# new_data = test_X_padded[:5]  # Taking the first 5 entries of the test data for example
# predictions = predict(model, new_data, device, target_scaler) # Pass scaler
# print("Predictions:", predictions)



def create_lift_chart(model, test_loader, device, scaler):
    """
    Creates a lift chart to evaluate the model's performance.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for the test set.
        device: Device to use for prediction.
        scaler: StandardScaler object used for scaling the target variable.
    """
    model.eval()
    true_values = []
    predicted_values = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            predicted_scaled = output.cpu().numpy()
            predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()
            true_scaled = y.cpu().numpy()
            true = scaler.inverse_transform(true_scaled.reshape(-1, 1)).flatten()  # Inverse transform

            true_values.extend(true)
            predicted_values.extend(predicted)

    # Sort predictions and true values together by predicted values
    combined = sorted(zip(predicted_values, true_values), reverse=True)
    sorted_predicted, sorted_true = zip(*combined)

    # Calculate cumulative gains
    cumulative_gains = np.cumsum(sorted_true)
    total_actual = np.sum(sorted_true)

    # Calculate lift
    lift = cumulative_gains / np.arange(1, len(sorted_true) + 1)
    baseline = np.mean(sorted_true) # avg of the true target value
    lift /= baseline

    # Plot the lift chart
    plt.figure(figsize=(10, 6))
    plt.plot(lift)
    plt.xlabel("Percentage of Data")
    plt.ylabel("Lift")
    plt.title("Lift Chart")
    plt.grid(True)

    # Mark the x-axis as percentages
    num_ticks = 11  # Number of ticks (0%, 10%, ..., 100%)
    indices = np.linspace(0, len(sorted_true) - 1, num_ticks, dtype=int)  # Evenly spaced indices
    percentage_labels = [f"{i}%" for i in range(0, 101, 10)]  # Labels for 0% to 100%
    plt.xticks(indices, percentage_labels)

    plt.show()




# Create the lift chart using the test data
create_lift_chart(model, test_loader, device, target_scaler)