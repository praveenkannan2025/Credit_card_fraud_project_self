# Credit Card Fraud Detection

## Overview
This project aims to develop a machine learning model to detect fraudulent credit card transactions. The dataset used contains transactions labeled as fraudulent or legitimate, and various data preprocessing and feature engineering techniques are applied to improve model performance.

## Dataset
The dataset used for this project is the **[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** from Kaggle. It consists of anonymized transaction features and a fraud label.

- **Number of transactions:** 284,807
- **Fraudulent transactions:** 492 (~0.172%)
- **Features:** 30 (including `Time`, `Amount`, and 28 anonymized PCA components)

## Project Workflow
1. **Data Preprocessing**
   - Handling class imbalance (e.g., SMOTE, undersampling, oversampling)
   - Normalization & Standardization
   - Feature selection
   
2. **Exploratory Data Analysis (EDA)**
   - Distribution of transaction amounts
   - Fraud vs. non-fraud comparisons
   - Correlation analysis

3. **Model Training & Evaluation**
   - Supervised learning models: Logistic Regression, Decision Trees, Random Forest, XGBoost
   - Deep learning approach: Neural Networks
   - Performance metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC

4. **Deployment (Optional)**
   - Flask API for real-time fraud detection
- Model serving with FastAPI or Streamlit

## Installation
To run the project locally, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the training script:
```bash
python train.py
```
Make predictions:
```bash
python predict.py --input test_data.csv
```

## Results
The model achieves an **AUC-ROC score of ~0.98**, effectively distinguishing fraudulent transactions.

## Technologies Used
- Python (NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch)
- Data Visualization (Matplotlib, Seaborn)
- Model Deployment (Flask, FastAPI, Streamlit)

## Future Improvements
- Hyperparameter tuning for better model performance
- Implementing an anomaly detection approach
- Enhancing real-time fraud detection with streaming data

## Contributing
Feel free to fork the repository and submit pull requests for improvements!

## License
This project is licensed under the MIT License.
