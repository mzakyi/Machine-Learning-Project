# Machine Learning Project: Predicting Molecular Solubility

This project applies **Machine Learning regression models** to predict **molecular solubility (logS)** using molecular descriptors.  
Two models were built and compared: **Linear Regression** and **Random Forest Regressor**.

---

## 🎯 Project Purpose

The purpose of this project is to leverage molecular descriptors to predict solubility values. 
Accurate predictions of solubility are important in **drug discovery and materials science** where solubility affects absorption, bioavailability, and formulation.

---

## 📂 Project Structure
```
├── ae605631-55b8-4b3d-a7ab-f93f57868144.ipynb   # Main Jupyter Notebook
├── delaney_solubility_with_descriptors.csv      ['Dataset']
├── README.md                                   # Documentation
```

---

## ⚙️ Requirements

Install the required Python libraries:

```bash
pip install pandas scikit-learn matplotlib jupyter
```

---

## 🚀 Steps in the Project

1. **Data Loading & Preparation**
   - Loaded the dataset [`delaney_solubility_with_descriptors.csv` ](https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv)
   - Features: MolLogP, MolWt, NumRotatableBonds, AromaticProportion.
   - Target: `logS` (solubility).
   - Split dataset into **80% training** and **20% testing**.

2. **Model Building**
   - **Linear Regression (LR):**
     - Trained on training set.
     - Predictions generated for both training and test sets.
     - Evaluated with **Mean Squared Error (MSE)** and **R²**.
   - **Random Forest Regressor (RF):**
     - Configured with `max_depth=2`, `random_state=100`.
     - Predictions generated for training and test sets.
     - Evaluated with same metrics.

3. **Evaluation**
   - Compared model performance in a summary table.
   - Visualized **Experimental vs Predicted values** for the Linear Regression model.
   - Added regression line for better interpretation.

---

## 📈 Results

- **Linear Regression** provided a simple baseline with moderate performance.  
- **Random Forest Regressor** showed improved accuracy compared to Linear Regression.  
- Visualizations confirmed alignment between predicted and experimental solubility values, though with some variance.  
- Both models captured general trends, but Random Forest performed better in handling feature interactions.  

---

## ✅ Conclusion

- **Random Forest Regressor outperformed Linear Regression** in predicting molecular solubility.  
- The project highlights how machine learning can assist in **chemoinformatics** by predicting properties of molecules before experimental validation.  
- Future improvements may include:
  - Hyperparameter tuning for Random Forest.
  - Testing more advanced models like Gradient Boosting or Neural Networks.
  - Incorporating additional molecular descriptors for richer feature representation.

---

## 📌 Example Usage

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
df = pd.read_csv('delaney_solubility_with_descriptors.csv')

X = df.drop('logS', axis=1)
y = df['logS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Train Random Forest
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
```

---
