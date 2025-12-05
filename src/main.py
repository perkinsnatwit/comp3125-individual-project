from src.model import CarPricePredictor
from src.loader import load_data, train_model



def main():
    """Main entry point for the car price prediction project."""
    X_train, X_test, y_train, y_test = load_data('data/car_price_dataset_medium.csv')

    input_features = X_train.shape[1]

    model = CarPricePredictor(input_features=input_features)   

    train_model(X_train, y_train, X_test, y_test, model)

if __name__ == '__main__':
    main()
