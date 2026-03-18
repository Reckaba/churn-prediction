# 🔮 Customer Churn Prediction

Predicting customer churn using ML classification models, deployed as a REST API with Docker.

## 📊 Results
| Model               | Accuracy | ROC-AUC | 
|---------------------|----------|---------|
| Logistic Regression | 79%      | 0.835   |
| Random Forest       | 78%      | 0.811   |

> Logistic Regression outperformed Random Forest with ROC-AUC of 0.835.

## 🔍 Key Insights from EDA
- Customers on **month-to-month contracts** churn at 3x the rate of annual subscribers
- **Tenure < 12 months** is a strong churn predictor
- **TotalCharges** and **MonthlyCharges** are the top 2 most important features

## 🛠 Tech Stack
- **ML:** scikit-learn, pandas, numpy
- **Experiment Tracking:** MLflow
- **API:** FastAPI
- **Containerization:** Docker
- **Visualization:** matplotlib, seaborn

## 🚀 Quick Start

### Run locally
```bash
git clone https://github.com/Reckaba/churn-prediction
cd churn-prediction
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Run with Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## 📁 Project Structure
```
churn-prediction/
├── data/               # Dataset and saved plots
├── notebooks/          # EDA and modeling notebooks
├── app/                # FastAPI application
├── train.py            # Model training script
├── Dockerfile
└── requirements.txt
```
