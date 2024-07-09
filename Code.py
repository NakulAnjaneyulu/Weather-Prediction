import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Generate synthetic weather data
np.random.seed(0)
n_samples = 1000

humidity = np.random.uniform(0, 100, n_samples)
pressure = np.random.uniform(980, 1050, n_samples)
wind_speed = np.random.uniform(0, 30, n_samples)
temperature = 20 + 0.5 * humidity - 0.02 * pressure + wind_speed * np.random.normal(0, 2, n_samples)

# Create a dataframe from the generated data
weather_data = pd.DataFrame({'humidity': humidity, 'pressure': pressure,
                             'wind_speed': wind_speed, 'temperature': temperature})

# Visualise the data
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.scatter(weather_data['humidity'], weather_data['temperature'], alpha=0.5)
plt.xlabel('Humidity')
plt.ylabel('Temperature')
plt.title('Humidity Vs. Temperature')

plt.subplot(2, 2, 2)
plt.scatter(weather_data['pressure'], weather_data['temperature'], alpha=0.5)
plt.xlabel('Pressure')
plt.ylabel('Temperature')
plt.title('Pressure Vs. Temperature')

plt.subplot(2, 2, 3)
plt.scatter(weather_data['wind_speed'], weather_data['temperature'], alpha=0.5)
plt.xlabel('Wind Speed')
plt.ylabel('Temperature')
plt.title('Wind Speed Vs. Temperature')

# Prepare data for training
X = weather_data[['humidity', 'pressure', 'wind_speed']]
y = weather_data['temperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

plt.subplot(2, 2, 4)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual Vs. Predicted Temperature')

plt.tight_layout()
plt.show()

# Predicting temperature for new data
new_data = pd.DataFrame({'humidity': [65], 'pressure': [1005], 'wind_speed': [15]})
prediction = model.predict(new_data)
print(f'Predicted Temperature: {prediction[0]}')
