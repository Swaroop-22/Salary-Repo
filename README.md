
# Employee Salary Prediction & Regression Analysis

This repository contains an end-to-end Machine Learning project focused on analyzing professional datasets to predict employee salaries. By leveraging supervised learning regression algorithms, this project maps how various independent factors—such as years of experience, education level, job role, and location—impact overall compensation structures.

---

## 📌 Project Architecture & Pipeline

The pipeline implements a robust workflow to clean raw data and build accurate predictive models:

1. **Exploratory Data Analysis (EDA):** Visualizes income distributions and identifies core feature relationships using correlation matrices and scatter plots (e.g., Experience vs. Salary).
2. **Data Preprocessing:** Handles missing entries, eliminates invalid rows, and applies `OneHotEncoder` or `LabelEncoder` to convert categorical text fields (like Job Title or Department) into numeric arrays.
3. **Feature Scaling:** Normalizes continuous variables using `StandardScaler` or `MinMaxScaler` to optimize gradient descent performance for specific algorithms.
4. **Model Development & Tuning:** Evaluates multiple regression models to find the line or curve of best fit.
5. **Performance Evaluation:** Quantifies prediction errors using standard statistical regression metrics.

---

## 🛠️ Installation & Dependencies

To execute the notebooks or run the python training scripts locally, install the following required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn

```

---

## 📊 Feature Dictionary (Typical Structure)

The training data maps client metrics to their respective target values:

| Feature Name | Type | Description |
| --- | --- | --- |
| **`Years of Experience`** | Continuous | Total professional tenure in the relevant industry |
| **`Age`** | Continuous | Chronological age of the employee |
| **`Education Level`** | Categorical | Highest achieved degree (e.g., Bachelor's, Master's, PhD) |
| **`Job Title / Role`** | Categorical | Specific occupational designation within the firm |
| **`Salary`** | Continuous | **Target Variable ($Y$):** Total annual compensation |

---

## 🤖 Deployed Regression Models

The project benchmarks several regression architectures to minimize overall prediction errors:

* **Linear Regression:** Establishes a baseline linear relationship ($Y = \beta_0 + \beta_1 X_1 + \dots + \beta_n X_n$).
* **Decision Tree Regressor:** Captures non-linear feature splits and interaction boundaries.
* **Random Forest Regressor:** An ensemble method that averages multiple decision trees to mitigate overfitting and improve structural stability.

---

## 💻 Code Implementation Snippet

### Training the Predictive Model

Below is the core implementation block used to split the dataset, fit a Random Forest regressor, and score the test inferences:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Split features and target
X = data.drop(['Salary'], axis=1)
y = data['Salary']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the ensemble regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Generate predictions
y_pred = model.predict(X_test)

```

### Evaluation Output Metrics

The test inferences are scored against the following evaluation metrics:

```python
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Root Mean Squared Error (RMSE): {mean_squared_error(y_test, y_pred, squared=False):.2f}")

```

---

## 📈 Key Findings & Insights

* **Tenure Dominance:** "Years of Experience" consistently acts as the strongest positive linear predictor for salary adjustments.
* **Ensemble Advantage:** Non-linear ensemble models (like Random Forest) generally yield significantly higher $R^2$ scores and lower root-mean-squared errors compared to vanilla Linear Regression, due to their ability to capture complex corporate salary bands.

---

## 🔮 Future Enhancements

* **Hyperparameter Optimization:** Integrate `GridSearchCV` or `RandomizedSearchCV` to fine-tune model parameters (e.g., max depth, number of estimators).
* **Web Interface:** Build a lightweight UI using **Streamlit** to let users input their experience and role to receive an instant salary estimation.

```

```
