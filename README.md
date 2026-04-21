# Cashew Quality Grading System

A college project for automated cashew kernel quality grading using deep learning. The system allows users to upload a cashew kernel image, sends it to a FastAPI backend, runs a trained TensorFlow model, and displays the predicted grade with confidence scores in a React dashboard.

## Project Overview

Cashew grading is usually performed manually by visually inspecting kernel size, shape, and quality. Manual grading can be slow, inconsistent, and dependent on human judgement. This project demonstrates how computer vision and deep learning can support automated cashew grade classification.

The current dataset is configured for W-grade cashew classification:

| Grade | Description |
| --- | --- |
| W180 | Premium large kernel |
| W210 | Jumbo kernel |
| W300 | Standard kernel |
| W500 | Small kernel |

The application is divided into two main parts:

| Part | Description |
| --- | --- |
| Backend | FastAPI service for model loading, image preprocessing, prediction, and training utilities |
| Frontend | React dashboard for uploading images, viewing predictions, charts, confidence values, and API/model status |

## Group Members

| Sr. No. | Name |
| --- | --- |
| 1 | Adwet Wandhare |
| 2 | Aditya Gaurkar |
| 3 | Aditya Gaikwad |
| 4 | Dhananjay Raut |
| 5 | Anuj Raipurkar |
| 6 | Anurag Bhise |

## Objectives

- Build a machine-learning based system for classifying cashew kernel grades.
- Provide an easy web interface for image upload and prediction.
- Train a deep learning model using the updated cashew dataset.
- Display prediction confidence and class probability distribution.
- Provide backend API endpoints for health checking and image prediction.
- Reduce manual effort and demonstrate the use of AI in agricultural quality inspection.

## Key Features

- Image upload from browser.
- Live backend connection status.
- Model weight loading status.
- Cashew grade prediction.
- Confidence score display.
- Class probability bar chart and pie chart.
- Batch/session prediction log.
- FastAPI Swagger documentation.
- Transfer learning based training pipeline using MobileNetV2.
- Support for class-folder datasets and Roboflow COCO annotation files.
- Automatic prevention of loading an old incompatible model with new labels.

## Tech Stack

### Frontend

| Technology | Purpose |
| --- | --- |
| React | User interface |
| Recharts | Charts and visual analytics |
| Lucide React | Icons |
| CSS | Dashboard styling |

### Backend

| Technology | Purpose |
| --- | --- |
| Python | Backend and ML code |
| FastAPI | REST API server |
| Uvicorn | ASGI server |
| TensorFlow/Keras | Deep learning model training and inference |
| MobileNetV2 | Transfer learning base model |
| OpenCV | Image decoding and preprocessing |
| NumPy | Numerical processing |
| python-multipart | File upload handling |

## Project Structure

```text
Cashew_Quality_Grading/
├── README.md
├── backend/
│   ├── Dataset/
│   │   ├── train/
│   │   ├── valid/
│   │   ├── test/
│   │   ├── README.dataset.txt
│   │   └── README.roboflow.txt
│   ├── cashew_best_model.keras
│   ├── cashew_labels.json
│   ├── cashew_model.h5
│   ├── Converter.py
│   ├── main.py
│   ├── model.py
│   ├── processor.py
│   ├── requirements.txt
│   └── train.py
└── frontend/
    ├── public/
    ├── src/
    │   ├── App.css
    │   ├── App.js
    │   ├── index.css
    │   └── index.js
    ├── package.json
    ├── package-lock.json
    └── yarn.lock
```

## Dataset

The dataset is stored inside:

```text
backend/Dataset/
```

The project currently uses the following classes:

```text
W180
W210
W300
W500
```

Expected dataset layout:

```text
Dataset/
├── train/
│   ├── W180/
│   ├── W210/
│   ├── W300/
│   └── W500/
├── valid/
│   ├── _annotations.coco.json
│   └── image files
└── test/
    ├── _annotations.coco.json
    └── image files
```

The training script can read:

- Class-folder format, such as `Dataset/train/W180/*.jpg`.
- COCO annotation format, such as `Dataset/valid/_annotations.coco.json`.

## Machine Learning Workflow

1. Images are collected and arranged into train, validation, and test splits.
2. The training script discovers classes from the dataset.
3. Images are resized to `224 x 224`.
4. Images are preprocessed using MobileNetV2 preprocessing.
5. A MobileNetV2 transfer learning model is trained in two phases.
6. The trained model is saved as `cashew_model.h5`.
7. The best checkpoint is saved as `cashew_best_model.keras`.
8. Class names are saved in `cashew_labels.json`.
9. The FastAPI backend loads the compatible model and serves predictions.
10. The React frontend displays the prediction result and confidence values.

## Backend Setup

Open PowerShell in the backend folder:

```powershell
cd C:\Cashew_Quality_Grading\backend
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the backend server:

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Backend will run at:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

## Frontend Setup

Open PowerShell in the frontend folder:

```powershell
cd C:\Cashew_Quality_Grading\frontend
```

Install dependencies:

```powershell
npm install
```

Start the frontend:

```powershell
npm start
```

Frontend will run at:

```text
http://localhost:3000
```

If `npm start` fails because of a local npm installation issue, use:

```powershell
node node_modules\react-scripts\bin\react-scripts.js start
```

## Training the Model

To train the model with the current dataset:

```powershell
cd C:\Cashew_Quality_Grading\backend
python train.py
```

After successful training, these files are created or updated:

| File | Purpose |
| --- | --- |
| `cashew_model.h5` | Final trained model used by the backend |
| `cashew_best_model.keras` | Best validation checkpoint |
| `cashew_labels.json` | Class labels used by the model |

Important: If the dataset classes are changed, retrain the model before running predictions. The backend checks that the loaded model output count matches the number of labels.

## API Endpoints

### Health Check

```http
GET /health
```

Returns backend status, model loading status, class labels, and model compatibility details.

Example response:

```json
{
  "status": "ok",
  "model_path": "cashew_model.h5",
  "labels_path": "cashew_labels.json",
  "model_loaded": true,
  "classes": ["W180", "W210", "W300", "W500"],
  "expected_classes": 4,
  "model_output_classes": 4,
  "loaded_model_path": "cashew_model.h5"
}
```

### Predict Cashew Grade

```http
POST /predict
```

Request type:

```text
multipart/form-data
```

Field:

| Field | Type | Description |
| --- | --- | --- |
| file | Image file | Cashew kernel image |

Example response:

```json
{
  "predicted_grade": "W210",
  "confidence": 0.86,
  "probability_spread": 0.72,
  "top_predictions": [
    {
      "grade": "W210",
      "confidence": 0.86
    },
    {
      "grade": "W180",
      "confidence": 0.08
    },
    {
      "grade": "W300",
      "confidence": 0.04
    }
  ],
  "class_probabilities": {
    "W180": 0.08,
    "W210": 0.86,
    "W300": 0.04,
    "W500": 0.02
  }
}
```

## How to Use the Application

1. Start the backend server.
2. Start the frontend app.
3. Open `http://localhost:3000`.
4. Check that backend status shows connected.
5. Check that model weights show loaded.
6. Click `Load Kernel Image`.
7. Select a cashew image.
8. Click `Start Grading Run`.
9. View predicted grade, confidence, charts, and log output.

## COCO Dataset Converter

The project includes `Converter.py` for converting a Roboflow COCO split into class folders.

Example:

```powershell
cd C:\Cashew_Quality_Grading\backend
python Converter.py --split valid
```

This creates:

```text
Dataset/valid_sorted/
```

The current training script does not require this conversion for validation and test splits because it can read COCO annotation files directly.

## Troubleshooting

### Backend is offline

Make sure the backend is running:

```powershell
cd C:\Cashew_Quality_Grading\backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Model weights missing

Train the model:

```powershell
python train.py
```

### Same confidence for every class

If the model returns almost equal probabilities for every class, such as around `25%` for four classes, it usually means:

- The model has not been trained properly.
- The wrong model file is being loaded.
- The model does not match the current labels.
- Training and prediction preprocessing are not matching.

Retrain the model and check `/health` to confirm `model_loaded` is `true` and `model_output_classes` is `4`.

### npm command not working

If normal npm commands fail, run React directly:

```powershell
node node_modules\react-scripts\bin\react-scripts.js start
```

## Future Scope

- Add object detection to locate multiple cashew kernels in one image.
- Store prediction history in a database.
- Add user authentication for lab or factory operators.
- Add model evaluation reports and confusion matrix visualization.
- Deploy backend and frontend to cloud hosting.
- Add support for more cashew grades and defect categories.
- Improve model accuracy using more balanced and larger datasets.

## Academic Note

This project was developed as a college project to demonstrate the use of deep learning, computer vision, backend APIs, and frontend dashboards for agricultural product quality grading.

