import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.model import CarPricePredictor
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_data(csv_path, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    
    # Identify the target column (handle different possible names)
    target_column = None
    for col in ['Price_USD', 'price', 'Price']:
        if col in df.columns:
            target_column = col
            break
    
    if target_column is None:
        raise ValueError("Could not find target column (Price_USD, price, or Price)")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Convert categorical (string) columns to numbers using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    
    # Convert to numpy arrays
    X = X.values
    y = y.values.reshape(-1, 1)
    
    # Standardize features
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test, model_instance):

    # Set the criterion of model to measure the error, how far off the predictions are from the data
    criterion = nn.MSELoss()

    # Choose optimizer, set learning rate
    optimizer = optim.Adam(model_instance.parameters(), lr=0.005)

    """
    Train the car price prediction model.
    """
    # Define epochs
    epochs = 100
    losses = []
    for i in range(epochs):
        # Go forward and get predictions
        y_pred = model_instance(X_train)

        # Measure the loss/error (expected to be high at first)
        loss = criterion(y_pred, y_train)

        # Keep track of the loss
        losses.append(loss.item())

        # Print every 10 epochs
        if i % 10 == 0:
            print(f'Epoch {i} | Loss: {loss.item()}')
        
        # Back propagation: take the eroor rate of forward pass and feed it back through the model to fine tune weights
        optimizer.zero_grad()  # Zero the gradients before running the backward pass
        loss.backward()        # Backward pass: compute gradient of the loss with respect to model
        optimizer.step()       # Update the model parameters based on the gradients

    # Graph loss over time
    plt.plot(range(epochs), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.show()