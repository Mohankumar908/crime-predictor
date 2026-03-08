import pandas as pd
import numpy as np

df = pd.read_csv("Crimes_2001_to_Present.csv")

df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')

df = df.dropna(subset=['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Hour'] = df['Date'].dt.hour
df['DayOfWeek'] = df['Date'].dt.day_name()

print(df.head())

# Aggregate monthly crime count
monthly_crime = df.groupby(['Year', 'Month']).size().reset_index(name='Crime_Count')

print(monthly_crime.head())
print("Total months:", len(monthly_crime))

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(monthly_crime['Crime_Count'])
plt.title("Monthly Crime Trend")
plt.xlabel("Month Index")
plt.ylabel("Crime Count")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# create date column from year and month
monthly_crime['Date'] = pd.to_datetime(
    monthly_crime['Year'].astype(str) + '-' + monthly_crime['Month'].astype(str)
)

# now sort
monthly_crime = monthly_crime.sort_values('Date')

# Create lag feature
monthly_crime['Prev_Month_Crime'] = monthly_crime['Crime_Count'].shift(1)

# Drop first row (will have NaN)
monthly_crime = monthly_crime.dropna()

X = monthly_crime[['Year', 'Month', 'Prev_Month_Crime']]
y = monthly_crime['Crime_Count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

import joblib 

joblib.dump(model, "crime_prediction_model.pkl")
