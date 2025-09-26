# Predicting Bike Sharing Demand with Linear Regression

## Dataset Introduction

The dataset used in this project comes from the public [dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) of the University of California, Irvine (UCI).

You can refer to the website for various information about this dataset, and you can also download and use the dataset directly from the website. However, this bike sharing dataset is in a fixed-dock format, which is different from the dockless bike-sharing systems like Mobike or Ofo. The data was collected during 2011-2012 and is divided into daily and hourly data.

Detailed description of features:

- instant: Record index
- dteday: Date
- season: Season (1: Spring, 2: Summer, 3: Fall, 4: Winter)
- yr: Year (0: 2011, 1: 2012)
- mnth: Month (1-12)
- hr: Hour (0-23)
- holiday: Whether it is a holiday
- weekday: Day of the week
- workingday: If it is neither a weekend nor a holiday, the value is 1, otherwise 0
- weathersit:
    1. Clear, few clouds, partly cloudy
    2. Mist + cloudy, mist + broken clouds, mist + few clouds, mist
    3. Light snow, light rain + thunderstorm + scattered clouds, light rain + scattered clouds
    4. Heavy rain + ice pellets + thunderstorm + mist, snow + fog
- temp: Normalized temperature in Celsius
    Formula: (t-t_min)/(t_max-t_min), t_min = -8, t_max = +39 (only within hourly range)
- atemp: Normalized feeling temperature in Celsius
    Formula: (t-t_min)/(t_max-t_min), t_min = -16, t_max = +50 (only within hourly range)
- hum: Normalized humidity. The values are divided by 100 (max value)
- windspeed: Normalized wind speed. The values are divided by 67 (max value)
- casual: Number of casual users
- registered: Number of registered users
- cnt: Total number of bike rentals, including both casual and registered users

## Project Overview

This project uses a linear regression model to predict bike sharing demand. By analyzing various factors that influence bike sharing usage (such as weather, season, temperature, etc.), we build a model that can accurately predict the daily total number of bike rentals.

## Tech Stack

- Python 3
- pandas: Data processing and analysis
- scikit-learn: Machine learning model building and evaluation
- matplotlib: Data visualization

## Project Structure

```
LinearRegression_Bike/
├── data/
│   └── bike-day.csv      # Daily bike sharing dataset
├── bikePredict.py        # Main code file, including data processing, model training and evaluation
├── READ_CN.md            # Chinese project documentation
└── README_EN.md          # English project documentation
```

## Implementation Steps

1. **Data Preprocessing**:
   - Read the CSV format dataset
   - Remove irrelevant features (such as 'instant', 'dteday')
   - Remove 'casual' and 'registered' features, as their sum is our prediction target 'cnt'

2. **Dataset Split**:
   - Split the dataset into training set (70%) and test set (30%)
   - Use a fixed random seed (random_state=42) to ensure reproducibility

3. **Model Training**:
   - Use scikit-learn's LinearRegression model
   - Fit the model with training set data

4. **Model Evaluation**:
   - Make predictions on the test set
   - Calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE)
   - Plot a comparison curve of predicted values versus actual values

## How to Run

After ensuring that the required Python libraries are installed, run the following command in the project root directory:

```bash
python bikePredict.py
```

## Result Analysis

The model will output MAE and MSE metrics to evaluate prediction accuracy. At the same time, it will generate a visualization chart showing the comparison between predicted values and actual values, helping to intuitively understand the model's performance.

## Future Improvements

1. Try more complex models, such as Random Forest or Gradient Boosting Trees
2. Perform feature engineering, create new features or transform existing features
3. Use cross-validation to optimize model hyperparameters
4. Consider time series characteristics and use time series prediction methods
5. Analyze feature importance to understand which factors have the greatest impact on bike sharing demand