import pickle
import numpy as np

from flask import Flask, request, jsonify


# Load the trained scaler and model
with open('model.bin', 'rb') as f_in:
    scaler, model = pickle.load(f_in)


# Activity labels from the UCI HAR dataset
activity_names = {
    1: 'Walking',
    2: 'Walking Upstairs',
    3: 'Walking Downstairs',
    4: 'Sitting',
    5: 'Standing',
    6: 'Laying'
}


app = Flask('har')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)

    # Check the input
    if not isinstance(data, dict) or not isinstance(data.get('features'), list):
        return jsonify({'error': 'Expected a list of 561 features'}), 400

    if len(data['features']) != 561:
        return jsonify({'error': 'Exactly 561 features are required'}), 400

    try:
        features = np.array(data['features'], dtype=float).reshape(1, -1)
    except (ValueError, TypeError):
        return jsonify({'error': 'Features must be numeric'}), 400

    if not np.isfinite(features).all():
        return jsonify({'error': 'Features must be finite numbers'}), 400

    # Apply the same preprocessing used during training
    features_scaled = scaler.transform(features)

    # Predict the activity
    prediction = int(model.predict(features_scaled)[0])

    result = {
        'class_id': prediction,
        'activity': activity_names[prediction]
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)