
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import copy


# fetch dataset 
appliances_energy_prediction = fetch_ucirepo(id=374) 
  
# data (as pandas dataframes) 
X = appliances_energy_prediction.data.features 
y = appliances_energy_prediction.data.targets 
print(appliances_energy_prediction.variables)
df = pd.concat([X, y], axis=1)

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d%H:%M:%S', errors='coerce')
# variable information 


df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['decimal_hour'] = df['date'].dt.hour + df['date'].dt.minute / 60
print(df.head(10))

clean_df = df.drop(columns=['date'])
y = clean_df['Appliances']
X = clean_df.drop(columns=['Appliances'])
print("Shape of X = ", X.shape)
print("Shape of y = ", y.shape)

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
    X = X.to_numpy()
    y = y.to_numpy()
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)  # Use long for labels
    y_test = torch.tensor(y_test, dtype=torch.long)

    # X_train = X_train.view(-1, 1, 1, 31)
    # X_test = X_test.view(-1, 1, 1, 31)
    return X_train, X_test, y_train, y_test


# Preprocess the data

X_train, X_test, y_train, y_test= preprocess_data(X, y)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


window_size = 10

class EnergyDataset(Dataset):
    def __init__(self, X, y, window_size):
        self.X = X
        self.y = y
        self.window_size = window_size

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.window_size]
        y_seq = self.y[idx + self.window_size]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)


train_dataset = EnergyDataset(X_train, y_train, window_size)
test_dataset = EnergyDataset(X_test, y_test, window_size)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(torch.cuda.is_available())
