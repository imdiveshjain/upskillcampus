import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv(r'data/raw/traffic_train.csv')
df_test = pd.read_csv(r'data/raw/traffic_test.csv')

df_train['DateTime'] = pd.to_datetime(df_train['DateTime'], format='%d-%m-%Y %H:%M')
df_train['day_of_week'] = df_train['DateTime'].dt.dayofweek
df_train['hour'] = df_train['DateTime'].dt.hour
df_train['month'] = df_train['DateTime'].dt.month

X = df_train.drop(['DateTime', 'Vehicles'], axis=1)
y = df_train['Vehicles']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)
print(f'Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, rf_preds))}')

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)
print(f'XGBoost RMSE: {np.sqrt(mean_squared_error(y_test, xgb_preds))}')

X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test))

lstm_preds = lstm_model.predict(X_test_lstm)
print(f'LSTM RMSE: {np.sqrt(mean_squared_error(y_test, lstm_preds))}')
