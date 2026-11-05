import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Load the original training data
X_train = pd.read_csv(
    'data/X_train.txt',
    sep=r'\s+',
    header=None
)

y_train = pd.read_csv('data/y_train.txt', header=None)[0]


# Standardize features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)


# Train the final model
model = LogisticRegression(
    C=1,
    max_iter=2000,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# Save the scaler and model together
with open('model.bin', 'wb') as f_out:
    pickle.dump((scaler, model), f_out)

print('Model saved to model.bin')