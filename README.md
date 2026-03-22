# Veritas Health Engine ⚕️

**Veritas Health Engine** is a clinical decision support system built to predict diabetes risk using an intelligent, dynamic Machine Learning ensemble. It goes beyond static predictions by evaluating multiple models on the fly and providing transparent, interpretable reasoning via **Explainable AI (SHAP)**.

This repository contains the codebase for the engine, the data preprocessing pipelines, and a responsive **Streamlit clinical dashboard**.

---

## 🚀 Key Features

### 1. Dynamic Top-2 Model Ensemble (Tier 1)
Instead of relying on a single algorithm, the engine trains **five baseline models**:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

During inference, it calculates a **Confidence Score** for each model, dynamically selects the **Top-2 best-performing models** for that specific patient profile, and averages their probabilities to output the final risk prediction.

### 2. SHAP Interpretability (Tier 2)
Black-box AI isn't suitable for clinical environments. Veritas wraps the dynamic ensemble in a custom `shap.KernelExplainer` to extract the exact features driving every individual prediction. It translates complex SHAP force vectors into:
- Visual **Feature Importance Bar Charts**
- Readable **Text Summaries** (e.g., *🔴 Increased Risk Factors: Fasting Blood Sugar, BMI*)

### 3. Streamlit Clinical Dashboard
A modern, responsive user interface allowing practitioners to:
- Upload any clinical `dataset.csv` directly from the sidebar.
- Automatically trigger the preprocessing, one-hot encoding, and scaling pipelines.
- Train the models and view real-time Test AUC scores.
- Simulate patient data via automatically generated sliders and dropdowns.
- Instantly pull an existing random patient from the dataset for rapid edge-case testing.

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **Data Pipeline:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`
- **Explainable AI:** `shap`, `matplotlib`, `seaborn`
- **Frontend Dashboard:** `streamlit`

---

## 📂 Project Structure
```text
📦 Veritas-Health-Engine
 ┣ 📜 app.py                  # Streamlit frontend clinical dashboard
 ┣ 📜 engine.py               # Core ML logic (VeritasHealthEngine class)
 ┣ 📜 dataset.csv             # The real 500+ row dataset (user-provided)
 ┣ 📜 requirements.txt        # Required python dependencies
 ┗ 📜 README.md               # Project documentation
```

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/veritas-health-engine.git
cd veritas-health-engine
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Clinical Dashboard**
```bash
streamlit run app.py
```
*The app will automatically open in your browser at `http://localhost:8501`.*

---

## 📊 How It Works (The Pipeline)
1. **Raw Data Ingestion:** Cleans categorical inputs (lowercases, strips spaces).
2. **Imputation:** Maps `mode` for categorical missing values and `median` for numericals.
3. **Encoding & Scaling:** Applies `pd.get_dummies` for Categorical One-Hot Encoding and `StandardScaler` (strictly required for SVM/KNN convergence).
4. **Stratified Splitting:** Uses an 80/20 train/test split.
5. **Ensemble & SHAP:** Trains models, ranks via Confidence Score, ensemble averages the Top-2, and plots SHAP values.

---

### License
This project was developed as a research implementation. Licensed under the MIT License.
