import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/raw/traffic_train.csv')

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
else:
    raise ValueError("The dataset must contain a 'date' column.")

df['day_of_week'] = df['date'].dt.dayofweek
df['hour'] = df['date'].dt.hour
df['is_holiday'] = df['date'].isin(['2024-01-01', '2024-12-25'])  # Update with relevant holidays
df['is_weekend'] = df['day_of_week'].isin([5, 6])

df.fillna(method='ffill', inplace=True)

required_columns = ['junction', 'day_of_week', 'hour', 'is_holiday', 'is_weekend', 'weather_conditions']
if not all(col in df.columns for col in required_columns):
    raise ValueError("The dataset must contain the following columns: 'junction', 'day_of_week', 'hour', 'is_holiday', 'is_weekend', 'weather_conditions'.")

X = df[['junction', 'day_of_week', 'hour', 'is_holiday', 'is_weekend', 'weather_conditions']]
y = df['traffic_volume']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
