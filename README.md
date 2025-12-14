# Anemia Type Classification with Deep Learning

This project implements an Artificial Neural Network (ANN) model to classify anemia types or healthy status based on Complete Blood Count (CBC) data.

## About the Project

The model analyzes hemogram parameters (WBC, HGB, MCV, etc.) to classify patients into one of 9 distinct categories (e.g., Iron deficiency anemia, Normocytic hypochromic anemia, Healthy, etc.).

## Dataset
link: https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification/data

Dataset used: `diagnosed_cbc_data_v4.csv`

Input Features:
- WBC (White Blood Cell)
- RBC (Red Blood Cell)
- HGB (Hemoglobin)
- HCT (Hematocrit)
- MCV, MCH, MCHC
- PLT (Platelet)
- And other relevant blood indices.

## Technologies and Libraries

- Python 3
- Pandas (Data processing)
- NumPy (Numerical operations)
- Scikit-learn (Preprocessing and train/test split)
- TensorFlow / Keras (Deep learning model)

## Model Architecture

The model is built using the Keras Sequential API:
1. Input Layer: 64 neurons, ReLU activation
2. Hidden Layer: 32 neurons, ReLU activation
3. Hidden Layer: 64 neurons, ReLU activation
4. Output Layer: 9 neurons (corresponding to classes), Softmax activation

## Installation and Usage

1. Install the required libraries:
   pip install pandas numpy scikit-learn tensorflow

2. Clone or download the project.

3. Run the `anemia-type-prediction.ipynb` file via Jupyter Notebook or Google Colab.

## Training Details

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Epochs: 128
- Batch Size: 32
- Validation Split: 20%
