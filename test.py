import pandas as pd
import requests


# Load one real observation from the UCI HAR test dataset
X_test = pd.read_csv(
    'data/X_test.txt',
    sep=r'\s+',
    header=None
)

# Convert the first observation to a Python list
features = X_test.iloc[0].tolist()

# Prepare the request
url = 'http://127.0.0.1:9696/predict'

data = {
    'features': features
}

# Send the observation to the Flask API
response = requests.post(url, json=data, timeout=30)

print('Status code:', response.status_code)
print('Prediction:', response.json())