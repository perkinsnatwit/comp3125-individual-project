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

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

def train_model(X_train, y_train, X_test, y_test, model_instance, learning_rate=0.005, epochs=1000):

    print("Starting training...")
    print("learning_rate:", learning_rate)
    print("epochs:", epochs)

    # Set the criterion of model to measure the error, how far off the predictions are from the data
    criterion = nn.MSELoss()

    # Choose optimizer, set learning rate
    optimizer = optim.Adam(model_instance.parameters(), lr=learning_rate, weight_decay=0.0001)

    best_test_loss = float('inf')
    patience_counter = 0
    patience = 50  
    
    epochs = epochs
    losses = []
    test_losses = []

    for i in range(epochs):
        # Go forward and get predictions
        y_pred = model_instance(X_train)

        # Measure the loss/error (expected to be high at first)
        train_loss = criterion(y_pred, y_train)

        # Keep track of the loss
        losses.append(train_loss.item())
        
        # Back propagation: take the eroor rate of forward pass and feed it back through the model to fine tune weights
        optimizer.zero_grad()  # Zero the gradients before running the backward pass (accumulated by default)
        train_loss.backward()        # Backward pass: compute gradient of the loss with respect to model
        optimizer.step()       # Update the model parameters based on the gradients

        # Validation (test loss)
        with torch.no_grad():
            y_test_pred = model_instance(X_test)
            test_loss = criterion(y_test_pred, y_test)
            test_losses.append(test_loss.item())

        # Early stopping logic
        if test_loss.item() < best_test_loss:
            best_test_loss = test_loss.item()
            patience_counter = 0
            # Save best model?
            # torch.save(model_instance.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if i % 10 == 0:
            print(f'Epoch {i} | Train Loss: {train_loss.item():.6f} | Test Loss: {test_loss.item():.6f}')

        # Stop if no improvement
        if patience_counter >= patience:
            print(f'Early stopping at epoch {i}')
            break

    # Plot both train and test loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(losses)), losses, label='Train Loss')
    plt.plot(range(len(test_losses)), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Test Loss')
    plt.legend()
    plt.show()

def evaluate_test_data(X_test, y_test, model_instance):
    # Disable back propogation
    with torch.no_grad():
        y_eval = model_instance(X_test) # Features from test set, y_eval are the predicted prices
        criterion = nn.MSELoss()
        loss = criterion(y_eval, y_test) # Find the loss between predicted prices and actual prices
        print(f'Test Loss: {loss.item()}')

def correction(X_test, y_test, model_instance, scaler_y, tolerance=15000):
    correct = 0
    with torch.no_grad():
        y_pred = model_instance(X_test)

        # Inverse-transform to get actual dollar amounts
        y_pred_unscaled = scaler_y.inverse_transform(y_pred.cpu().numpy())
        y_test_unscaled = scaler_y.inverse_transform(y_test.cpu().numpy())

        for i in range(len(y_test)):
            predicted_price = y_pred_unscaled[i][0]
            actual_price = y_test_unscaled[i][0]
            difference = abs(predicted_price - actual_price)

            print(f'{i+1}.) Predicted: ${predicted_price:.2f}, Actual: ${actual_price:.2f}, Diff: ${difference:.2f}')
            
            # Check if prediction is within tolerance
            if difference <= tolerance:
                correct += 1
    
    accuracy = (correct / len(X_test)) * 100
    print(f'\nPredictions within ${tolerance}: {correct}/{len(X_test)} ({accuracy:.1f}%)')
