# Car Price Prediction Model

A deep learning project to predict used car prices using neural networks trained on comprehensive car feature data.

## Project Overview

This project builds a PyTorch-based neural network to predict car prices using a dataset of vehicle specifications and characteristics. The pipeline includes data munging, feature engineering, model training, and evaluation.

## Dataset

- **Source**: `data/car_price_dataset_medium.csv`
- **Size**: 1000 vehicles
- **Target Variable**: Price (in USD)
- **Key Features**: Brand, Model Year, Engine specs, Transmission, Fuel Type, Mileage, Kilometers Driven, and more

### Data Files

- `car_price_dataset_medium.csv` - Raw dataset with original features
- `car_price_dataset_munged.csv` - Processed dataset with engineered features
- `car_price_dataset_cleaned.csv` - Alternative cleaned dataset

## Project Structure

```
src/
├── main.py           # Entry point for training the model
├── model.py          # Neural network architecture
├── loader.py         # Data loading and preprocessing
├── munger.py         # Feature engineering pipeline
└── quality_check.py  # Data validation utilities

models/
├── car_price_model.pth    # Trained PyTorch model weights
└── sklearn_model.joblib   # Alternative sklearn model (if applicable)

data/                 # Dataset files (CSVs)
tests/               # Unit tests
```

## Features

### Original Features
- **Car Details**: Brand, Model Year, Car ID
- **Engine Specs**: Engine CC (displacement), Max Power (bhp), Mileage (kmpl)
- **Drivetrain**: Transmission type, Fuel type
- **Usage**: Kilometers driven, Seats, Owner type

### Engineered Features

**Numeric Features**:
- `Vehicle_Age` - Car age calculated from model year
- `Price_Per_Year` - Price depreciation per year of age
- `Power_Per_CC` - Engine power efficiency (bhp per 1000cc)
- `Efficiency_to_Power` - Fuel efficiency relative to power output
- `Usage_Intensity` - Average kilometers driven per year
- `Depreciation_Rate` - Depreciation metric based on age and power
- `Is_Automatic` - Binary indicator for automatic transmission
- `Owner_History_Numeric` - Owner type as numeric (0-2)
- `Price_Per_Seat` - Price normalized by seating capacity

**Categorical Features**:
- `Mileage_Category` - Quartile-based categorization (Low, Medium, High, Very High)
- `Price_Category` - Price range categorization (Budget, Mid-range, Premium, Luxury)
- `Engine_Size_Category` - Engine displacement categorization (Small, Medium, Large)
- `Power_Rating` - Power output categorization (Low, Medium, High, Very High)
- `Fuel_Type_Category` - Broad fuel type grouping (Traditional, Eco-Friendly)

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd comp3125-personal-project
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\Activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

3. Install dependencies:
```bash
pip install pandas numpy torch scikit-learn matplotlib
```

## Usage

### 1. Data Preparation

Run the munger to create engineered features from raw data:
```bash
python -m src.munger
```

This generates `car_price_dataset_munged.csv` with all engineered features.

### 2. Train the Model

Execute the main training script:
```bash
python -m src.main
```

This will:
- Load and preprocess the data
- Split into train/test sets (80/20)
- Initialize the neural network
- Train for 500 epochs
- Evaluate on test data
- Save the trained model to `models/car_price_model.pth`

### Configuration Parameters (in `main.py`)

- `hidden_size` (default: 32) - Size of hidden layers
- `dropout_prob` (default: 0.5) - Dropout probability for regularization
- `lr` (default: 0.00065) - Learning rate
- `epoch_num` (default: 500) - Number of training epochs
- `tolerance` (default: 20000) - Price tolerance for correction analysis

## Model Architecture

**CarPricePredictor** - A 3-layer feedforward neural network:
- **Layer Size** - Modular; 32 is the settled upon size
- **Dropout** - Modular; 50% is the settled upon rate

```
Input Layer (N features)
    ↓
Linear (N → 32) + ReLU + Dropout(0.5)
    ↓
Linear (32 → 16) + ReLU + Dropout(0.5)
    ↓
Linear (16 → 1) [Price output]
```

**Features**:
- Dropout layers to prevent overfitting
- ReLU activation functions
- GPU support (CUDA if available, CPU fallback)
- Data normalization with StandardScaler

## Results

The model evaluates performance on test data with metrics including:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Price predictions with correction analysis

## Technology Stack

- **Deep Learning**: PyTorch
- **Data Processing**: Pandas, NumPy
- **Preprocessing**: Scikit-learn
- **Visualization**: Matplotlib

