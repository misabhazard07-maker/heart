# Heart Disease Prediction

A machine learning project that predicts the presence of heart disease using the Cleveland Heart Disease dataset. The notebook covers end-to-end data analysis, visualization, model training, and evaluation using Python and scikit-learn.

---

## Dataset

The project uses `heart.csv`, a dataset with **1,025 patient records** and **14 features**:

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy |
| `thal` | Thalassemia type |
| `target` | **Label** — 1 = Heart Disease, 0 = No Disease |

**Class balance:** 526 positive (disease) vs 499 negative (no disease) — roughly balanced.

---

## Project Structure

```
├── heart.csv             # Dataset
├── notebook.ipynb        # Main Jupyter notebook
└── README.md
```

---

## Installation

```bash
pip install pandas scikit-learn matplotlib seaborn
```

---

## Workflow

### 1. Exploratory Data Analysis
- Dataset shape, data types, and missing value check (no nulls found)
- Class balance check on the target variable
- Age distribution by disease status
- Correlation heatmap across all features

### 2. Visualizations
- Heart disease count (bar chart)
- Age distribution of patients (histogram)
- Average cholesterol by disease status (bar chart)
- Max heart rate by disease status (box plot)
- Patient sex distribution (pie chart)

### 3. Model Training
The dataset is split 80/20 (train/test, `random_state=42`):
- **Train size:** 820 samples
- **Test size:** 205 samples

Three classifiers are trained and compared:

| Model | Accuracy |
|---|---|
| Decision Tree (max_depth=5) | 84.39% |
| Random Forest (100 estimators) | **98.54%** |
| Logistic Regression | 79.51% |

### 4. Evaluation
- Accuracy score
- Confusion matrix (plotted with `ConfusionMatrixDisplay`)
- Feature importance plot (Decision Tree)

**Decision Tree Confusion Matrix (on test set):**
```
              Predicted
              No Disease  Disease
Actual  No Disease  [ 75        27 ]
        Disease     [  5        98 ]
```

---

## Results

The **Random Forest** classifier achieved the best performance at **98.54% accuracy**. The Decision Tree model reached 84.39% with a depth of 5, and Logistic Regression achieved 79.51%.

Feature importance analysis from the Decision Tree reveals which clinical indicators are most predictive of heart disease.

---

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn
