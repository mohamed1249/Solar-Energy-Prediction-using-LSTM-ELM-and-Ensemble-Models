# Solar Energy Prediction using LSTM, ELM, and Ensemble Models

## Overview

This repository contains code for predicting solar energy using LSTM (Long Short-Term Memory), ELM (Extreme Learning Machine), and an Ensemble model combining both. The project includes hyperparameter tuning using Bayesian Optimization, data preprocessing, and performance evaluation.

## Files

- `Full Code.ipynb`: This notebook contains the complete workflow for data preprocessing, model training, hyperparameter tuning, and evaluation of the models.
- `Graph Code.ipynb`: This notebook generates future time points for predictions and utilizes the trained ensemble model to forecast solar energy values.

## Requirements

- TensorFlow
- Keras
- Scikit-learn
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Bayesian-Optimization
- Statsmodels
- Requests

## Usage

### Full Code.ipynb

#### 1. GPU Setup

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if not gpu_devices:
    print("No GPU devices found. TensorFlow is running on CPU.")
else:
    print("Available GPU devices:")
    for device in gpu_devices:
        print(device.name)
    print(f"TensorFlow is running on GPU: {tf.test.is_gpu_available()}")
```

#### 2. Data Loading and Preprocessing

The data is loaded from CSV files in the `input data` directory. Missing values and unnecessary columns are removed, and the data is grouped by hourly intervals.

```python
import pandas as pd
import os

directory = 'input data'
df_list = [pd.read_csv(os.path.join(directory, filename)) for filename in os.listdir(directory) if filename.endswith('.csv')]
data = pd.concat(df_list, ignore_index=True)

data = data.drop(['Record ID','Unnamed: 3'], axis=1).dropna()
data['Date Time'] = pd.to_datetime(data['Date Time']).dt.round('H')
data = data.loc[data['Date Time'] <= '2023-03'].groupby('Date Time').mean().reset_index().sort_values(by='Date Time')
data.set_index('Date Time', inplace=True)
```

#### 3. Feature Engineering

Lag features are generated to capture temporal dependencies in the data.

```python
for i in range(1, 14):
    data[f"lag_{i}"] = data['Solar Avg'].shift(i)
data.dropna(inplace=True)
```

#### 4. Data Scaling and Splitting

Data is scaled and split into training and testing sets.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = data.drop(['Solar Avg'], axis=1)
y = data['Solar Avg']

data_scaler = StandardScaler()
target_scaler = StandardScaler()
scaled_data = data_scaler.fit_transform(X.values)
scaled_target = target_scaler.fit_transform(y.values.reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(scaled_data, scaled_target, test_size=0.2, shuffle=False)
```

#### 5. LSTM Model and Hyperparameter Tuning

The LSTM model is defined and optimized using Bayesian Optimization.

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
from bayes_opt import BayesianOptimization

def create_lstm_model(units_1, units_2, learning_rate):
    model = Sequential()
    model.add(LSTM(units_1, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units_2, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=learning_rate))
    return model

pbounds = {'units_1': (64, 256), 'units_2': (32, 128), 'learning_rate': (0.001, 0.1)}

def lstm_cv(units_1, units_2, learning_rate):
    model = create_lstm_model(units_1=int(units_1), units_2=int(units_2), learning_rate=learning_rate)
    model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=0)
    mse = mean_squared_error(y_test, model.predict(x_test))
    return -mse

lstm_bo = BayesianOptimization(f=lstm_cv, pbounds=pbounds, random_state=42)
lstm_bo.maximize(init_points=5, n_iter=10)

best_params = lstm_bo.max['params']
lstm_model = create_lstm_model(learning_rate=best_params['learning_rate'], units_1=round(best_params['units_1']), units_2=round(best_params['units_2']))
lstm_model.fit(x_train, y_train, epochs=50, callbacks=[callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.001, restore_best_weights=True)])
```

#### 6. ELM Model and Hyperparameter Tuning

The ELM model is defined and optimized using Bayesian Optimization.

```python
def create_elm_model(hidden_units, activation, learning_rate):
    elm_model = Sequential()
    elm_model.add(Dense(hidden_units, input_dim=x_train.shape[1], activation=activation, kernel_initializer='he_uniform'))
    elm_model.add(Dense(1))
    optimizer = optimizers.Adam(lr=learning_rate)
    elm_model.compile(loss='mean_squared_error', optimizer=optimizer)
    return elm_model

def hyp_opt(hidden_units, learning_rate):
    model = KerasRegressor(build_fn=create_elm_model, hidden_units=hidden_units, activation='relu', learning_rate=learning_rate, verbose=0)
    scores = -1 * cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return np.mean(scores)

pbounds = {'hidden_units': (32, 128), 'learning_rate': (0.001, 0.1)}
optimizer = BayesianOptimization(f=hyp_opt, pbounds=pbounds, random_state=42)
optimizer.maximize(n_iter=10, init_points=5)

best_params = optimizer.max['params']
elm_model = create_elm_model(activation='relu', hidden_units=round(best_params['hidden_units']), learning_rate=best_params['learning_rate'])
elm_model.fit(x_train, y_train, epochs=50, callbacks=[callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.001, restore_best_weights=True)])
```

#### 7. Ensemble Model

An ensemble model is created by combining the outputs of the LSTM and ELM models.

```python
from keras.models import Model
from keras.layers import Input, Concatenate

input_layer = Input(shape=(X.shape[1],))
lstm_output = lstm_model(input_layer)
elm_output = elm_model(input_layer)
concat_output = Concatenate()([lstm_output, elm_output])
dense_output = Dense(1)(concat_output)

ensemble_model = Model(inputs=input_layer, outputs=dense_output)
ensemble_model.compile(loss='mean_squared_error', optimizer='adam')
ensemble_model.fit(x_train, y_train, epochs=50, callbacks=[callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.001, restore_best_weights=True)])
```

#### 8. Model Evaluation and Predictions

The performance of the models is evaluated using various metrics and predictions are plotted.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

def evaluate_model(model, x_test, y_test, target_scaler):
    predictions = model.predict(x_test)
    predictions = target_scaler.inverse_transform(predictions)
    y_test_descaled = target_scaler.inverse_transform(y_test)
    mae = mean_absolute_error(y_test_descaled, predictions)
    mse = mean_squared_error(y_test_descaled, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_descaled, predictions)
    mape = mean_absolute_percentage_error(y_test_descaled, predictions) * 100
    return mae, mse, rmse, r2, mape

lstm_metrics = evaluate_model(lstm_model, x_test, y_test, target_scaler)
elm_metrics = evaluate_model(elm_model, x_test, y_test, target_scaler)
ensemble_metrics = evaluate_model(ensemble_model, x_test, y_test, target_scaler)

print(f"LSTM Model: MAE={lstm_metrics[0]}, MSE={lstm_metrics[1]}, RMSE={lstm_metrics[2]}, R²={lstm_metrics[3]}, MAPE={lstm_metrics[4]}")
print(f"ELM Model: MAE={elm_metrics[0]}, MSE={elm_metrics[1]}, RMSE={elm_metrics[2]}, R²={elm_metrics[3]}, MAPE={elm_metrics[4]}")
print(f"Ensemble Model: MAE={ensemble_metrics[0]}, MSE={ensemble_metrics[1]}, RMSE={ensemble_metrics[2]}, R²={ensemble_metrics[3]}, MAPE={ensemble_metrics[4]}")

# Plotting predictions
def plot_predictions(valid, predictions, title):
    valid['Predictions'] = predictions
    plt.figure(figsize=(30, 9))
    plt.title(title)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Solar', fontsize=18

)
    plt.plot(valid[['Solar Avg', 'Predictions']])
    plt.legend(['Val', 'Preds'], loc='lower right')
    plt.show()

lstm_predictions = lstm_model.predict(x_test)
elm_predictions = elm_model.predict(x_test)
ensemble_predictions = ensemble_model.predict(x_test)

valid = data[['Solar Avg']][len(data) - len(y_test):]
plot_predictions(valid, target_scaler.inverse_transform(lstm_predictions), 'LSTM Model')
plot_predictions(valid, target_scaler.inverse_transform(elm_predictions), 'ELM Model')
plot_predictions(valid, target_scaler.inverse_transform(ensemble_predictions), 'Ensemble Model')
```

### Graph Code.ipynb

This notebook generates future time points and utilizes the trained ensemble model to forecast solar energy values for those points.

#### 1. Generate Future Time Points

```python
import pandas as pd

# Generate future time points
start_date = '2023-03-01'
end_date = '2023-12-31'
date_range = pd.date_range(start=start_date, end=end_date, freq='H')
future_data = pd.DataFrame(index=date_range, columns=data.columns)
```

#### 2. Generate Lag Features

```python
for i in range(1, 14):
    future_data[f"lag_{i}"] = future_data['Solar Avg'].shift(i)
future_data.dropna(inplace=True)
```

#### 3. Scale Future Data

```python
scaled_future_data = data_scaler.transform(future_data.drop(['Solar Avg'], axis=1).values)
```

#### 4. Make Predictions

```python
future_predictions = ensemble_model.predict(scaled_future_data)
future_predictions = target_scaler.inverse_transform(future_predictions)
future_data['Predictions'] = future_predictions

# Plotting future predictions
plt.figure(figsize=(30, 9))
plt.title('Future Predictions')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Solar', fontsize=18)
plt.plot(future_data['Predictions'])
plt.show()
```

## License

This project is licensed under the MIT License.
