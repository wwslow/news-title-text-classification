# News Title Text Classification

A news title text classification system based on deep learning and machine learning, supporting multiple models including BERT, BERT-BiLSTM, and Random Forest.

## Project Overview

This project implements a news title text classification system that automatically categorizes news titles into predefined classes. It supports multiple model architectures, including:

- **BERT Classification Model** - Text classification based on pre-trained BERT
- **BiLSTM Model** - Bidirectional LSTM classifier
- **Model Distillation** - Knowledge distillation from BERT model to a lightweight BiLSTM model
- **Random Forest** - Traditional machine learning method

## Features

- Supports training and evaluation of multiple classification models
- Provides Flask RESTful API service
- Web-based prediction interface
- Model distillation functionality to transfer knowledge from large models to lightweight models

## Data Description

Data files are located in the `data/` directory:

- `train.csv` - Training dataset
- `dev.csv` - Validation dataset
- `test.csv` - Test dataset
- `class.txt` - Class label file

## Dependencies

```
torch
transformers
flask
scikit-learn
pandas
numpy
```

## Quick Start

### 1. Train Models

```bash
# Train BERT classification model
cd macbert
python model_train_eval.py

# Train Random Forest model
cd rf
python rf_train_eval.py
```

### 2. Start API Service

```bash
cd macbert
python service_api.py
```

### 3. Use Web Prediction Interface

After starting the service, access `html/index.html` to use the visual prediction interface.

## Project Structure

```
.
├── data/                    # Dataset
│   ├── train.csv
│   ├── dev.csv
│   ├── test.csv
│   └── class.txt
├── data_show/              # Data visualization
│   └── show_data.py
├── html/                   # Web frontend
│   └── index.html
├── macbert/                # BERT series models
│   ├── classification_model.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── model_train_eval.py
│   ├── model_distill.py
│   ├── service_api.py
│   ├── student_model.py
│   └── BiLSTM_predict_service.py
└── rf/                     # Random Forest model
    ├── config.py
    ├── data_preprocessing.py
    └── rf_train_eval.py
```

## API Usage

Send a POST request to the `/predict` endpoint:

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "your news title"}'
```

## License

MIT License