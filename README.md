# Human Activity Recognition Using Smartphone Sensors

**ML Zoomcamp 2026 — Midterm Project**

## 1. Problem Description

This project develops an end-to-end machine learning system to classify human physical activities using pre-extracted smartphone accelerometer and gyroscope features.

The model predicts one of six activities:

1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying

The project covers exploratory data analysis, model training, hyperparameter tuning, evaluation, and deployment through a Flask prediction API and Docker.

### Potential Application

The model could be integrated into a research or activity-analysis application that processes smartphone sensor recordings. Once the required 561 features have been extracted using a compatible preprocessing pipeline, the application can send them to the prediction API and receive the predicted activity.

This could support the analysis of human movement patterns in controlled experimental settings. The current implementation focuses on feature-based activity classification and does not process raw sensor signals directly.

## 2. Dataset

**Source:** [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

The dataset contains:

- 10,299 observations
- 561 pre-extracted numerical features
- 30 participants
- 6 activity classes

The original dataset provides separate training and test sets containing different participants.

A further participant-wise validation split was created from the original training set to prevent subject leakage during model selection.

## 3. Model Training and Evaluation

Four machine learning algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Models were compared using validation accuracy and macro-averaged F1-score.

The best-performing model was **Logistic Regression with C=1**, using StandardScaler for feature standardization.

## 4. Results

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.9577 | 0.9555 |
| Macro-F1 | 0.9543 | 0.9554 |

The final model was trained on all 7,352 original training observations and evaluated on the official test set of 2,947 observations from nine unseen participants.

The results demonstrate strong activity classification performance under the dataset's experimental conditions.

This is an educational machine learning prototype and is not validated for unrestricted real-world activity monitoring.

## 5. Downloading the Dataset

The original UCI HAR dataset is not included in this repository.

To reproduce the project, download the dataset from the UCI Machine Learning Repository.

Run the following commands from the project root directory:

```bash
mkdir -p data

curl -fL "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip" -o /tmp/har.zip

unzip -q -o /tmp/har.zip -d /tmp

cp /tmp/"UCI HAR Dataset"/train/{X_train.txt,y_train.txt,subject_train.txt} data/

cp /tmp/"UCI HAR Dataset"/test/{X_test.txt,y_test.txt,subject_test.txt} data/

cp /tmp/"UCI HAR Dataset"/features.txt data/
cp /tmp/"UCI HAR Dataset"/activity_labels.txt data/
```

After downloading, the `data/` directory contains the original training and test files.

The dataset is excluded from Git using `.gitignore`.

## 6. Installation and Model Training

### 6.1 Python Environment

The project was developed using Python 3.14.2.

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment (Linux/macOS):

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

### 6.2 Train the Model

After downloading the dataset and installing the dependencies, run:

```bash
python train.py
```

The script:

- Loads the original UCI HAR training data.
- Standardizes the 561 numerical features using StandardScaler.
- Trains a Logistic Regression model with C=1.
- Saves the trained scaler and classifier to `model.bin`.

Expected output:

```text
Model saved to model.bin
```

The saved model is used by the Flask prediction service.

## 7. Running the Prediction API

The project provides a Flask API that predicts human activities from 561 pre-extracted smartphone sensor features.

### 7.1 Start the API

After installing the dependencies, run:

```bash
python predict.py
```

The Flask development server will start on:

http://127.0.0.1:9696

### 7.2 Test the API

Keep the Flask server running and open a second terminal.

Run:

```bash
python test.py
```

The test script loads the first observation from the official UCI HAR test dataset and sends its 561 features to the `/predict` endpoint.

The API returns a JSON response containing the predicted activity and its class ID.

Example response from our local test:

```json
{
  "activity": "Standing",
  "class_id": 5
}
```

The API accepts HTTP POST requests with a JSON object containing a `features` list of exactly 561 numerical values.

The Flask development server is intended for local testing. Docker deployment uses Gunicorn.

## 8. Docker Deployment

The prediction service can also run inside a Docker container.

Docker packages the Python environment, dependencies, trained model, and prediction API into a reproducible image.

### 8.1 Build the Docker Image

From the project root directory, run:

```bash
docker build -t har-prediction .
```

The Docker image includes:
- Python 3.14.2
- Required dependencies
- Flask prediction service (`predict.py`)
- Trained model and scaler (`model.bin`)
- Gunicorn production WSGI server

### 8.2 Run the Docker Container

Run:

```bash
docker run --rm -p 9696:9696 har-prediction
```

The prediction API will be available at:

http://127.0.0.1:9696/predict

Keep the container running.

### 8.3 Test the Dockerized API

Open a second terminal.

After downloading the dataset and installing the local dependencies, run:

```bash
python test.py
```

Expected response from the tested sample:

```text
Status code: 200
Prediction: {'activity': 'Standing', 'class_id': 5}
```

The API was successfully tested both locally and inside Docker.

The Docker image does not require the original dataset for inference, since it includes the trained model. The local test script uses the dataset to prepare a sample request.

## 9. Reproducing the Notebook

The complete machine learning workflow is available in `notebook.ipynb`.

It includes exploratory data analysis, participant-wise validation, model comparison, hyperparameter tuning, and final evaluation.

### 9.1 Install Notebook Dependencies

After creating and activating the virtual environment, run:

```bash
python -m pip install -r requirements-notebook.txt
```

This installs the dependencies required for the training pipeline, visualizations, and XGBoost experiments.

### 9.2 Run the Notebook

1. Download the dataset following Section 5.
2. Open `notebook.ipynb` in VS Code with the Jupyter extension.
3. Select the project's `.venv` Python kernel.
4. Click **Restart & Run All**.

The notebook reproduces the model training, validation results, and final test evaluation.

All experiments use `random_state=42` where applicable, and the official UCI test set is reserved for final evaluation.

## 10. Cloud Deployment

The trained Human Activity Recognition model is deployed as a public prediction API on Render using Docker, Flask, and Gunicorn.

**Live API:** https://human-activity-recognition-mlzoomcamp.onrender.com

**Prediction endpoint:** `POST /predict`

The deployed model uses StandardScaler and Logistic Regression to classify six human activities from 561 pre-extracted smartphone sensor features.

### 10.1 Test the Cloud API

A ready-to-use example request is provided in `sample_request.json`.

The file contains 561 numerical features from one observation in the official UCI HAR test dataset.

To test the deployed model, run the following command from the repository directory:

```bash
curl --max-time 180 -X POST \
  https://human-activity-recognition-mlzoomcamp.onrender.com/predict \
  -H "Content-Type: application/json" \
  --data-binary @sample_request.json
```

**Expected response:**

```json
{
  "activity": "Standing",
  "class_id": 5
}
```

The API returns the predicted activity and its corresponding class ID (1–6).

### 10.2 Input Requirements

The prediction endpoint accepts an HTTP POST request containing a JSON object with a `features` field.

The `features` field must contain exactly 561 finite numerical values in the original UCI HAR feature order.

The model expects pre-extracted and normalized sensor features, not raw accelerometer or gyroscope signals.

The example request allows reviewers to test the cloud API without downloading the original dataset.

### 10.3 Deployment Details

- **Platform:** Render
- **Runtime:** Docker
- **Web framework:** Flask
- **Application server:** Gunicorn
- **Port:** 9696
- **Model:** Logistic Regression (C=1)
- **Preprocessing:** StandardScaler

The application uses Render's free service tier, which can spin down after inactivity. The first request after an idle period may take additional time.

This deployment is intended as an educational machine learning demonstration. Performance on arbitrary real-world smartphone recordings has not been established.