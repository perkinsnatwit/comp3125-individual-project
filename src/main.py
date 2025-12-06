import torch
import pandas as pd
from src.model import CarPricePredictor
from src.loader import load_data, train_model, evaluate_test_data, correction



def main():
    # Check availability of GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    """
    Configuration Parameters
    """
    path = 'data/car_price_dataset_munged.csv' # Path to the dataset CSV file

    hidden_size = 64 # Size of hidden layers in the model
    dropout_prob = 0.5 # Dropout probability for regularization

    lr = 0.0005 # Learning rate for model optimizer
    epoch_num = 500 # Number of training epochs

    tolerance = 20000 # Tolerance for price correction
    
    """
    Entry point for training the car price prediction model.
    """
    print (f'Loading data from: {path}')
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_data(path)
    X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Price range: ${y_train.min():.2f} - ${y_train.max():.2f}")

    input_features = X_train.shape[1]

    model = CarPricePredictor(input_features=input_features, hidden_size=hidden_size, dropout_prob=dropout_prob)   

    model.to(device)

    train_model(X_train, y_train, X_test, y_test, model, learning_rate=lr, epochs=epoch_num)

    evaluate_test_data(X_test, y_test, model)

    correction(X_test, y_test, model, scaler_y, tolerance=tolerance)

    torch.save(model.state_dict(), 'src/car_price_model.pth')
    print("Model saved to car_price_model.pth")

if __name__ == '__main__':
    main()
