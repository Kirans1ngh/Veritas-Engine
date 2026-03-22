import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

class VeritasHealthEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(kernel='rbf', probability=True, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5)
        }
        self.trained_models = {}
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_test = None
        self.feature_names = None
        self.background_data = None
        
        # Keep track of preprocessing metadata
        self.train_columns = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.imputers = {}
        
    def preprocess(self, df):
        df = df.copy()
        
        # Target Encoding
        if 'Diabetes_Status' in df.columns:
            y = df['Diabetes_Status'].str.strip().str.lower().map({'yes': 1, 'no': 0})
            X = df.drop('Diabetes_Status', axis=1)
        else:
            y = None
            X = df
            
        # Clean categorical columns
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
        
        for col in self.categorical_cols:
            X[col] = X[col].astype(str).str.lower().str.strip()
            
        # Impute missing values
        for col in self.categorical_cols:
            mode_val = X[col].mode()[0] if not X[col].mode().empty else 'unknown'
            X[col] = X[col].fillna(mode_val)
            if y is not None:  # Only save during training
                self.imputers[col] = mode_val
                
        for col in self.numerical_cols:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            if y is not None:
                self.imputers[col] = median_val
                
        # Use imputers if predicting
        if y is None:
            for col in self.categorical_cols:
                X[col] = X[col].fillna(self.imputers.get(col, 'unknown'))
            for col in self.numerical_cols:
                X[col] = X[col].fillna(self.imputers.get(col, 0))
                
        # One-Hot Encoding
        X_encoded = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True)
        
        # Ensure strict column matching between train and predict
        if y is not None:
            self.train_columns = X_encoded.columns.tolist()
        else:
            # Reindex to match training columns
            X_encoded = X_encoded.reindex(columns=self.train_columns, fill_value=0)
            
        self.feature_names = self.train_columns
        return X_encoded, y
        
    def train(self, df):
        X, y = self.preprocess(df)
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        # Feature Scaling
        self.X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=self.train_columns)
        self.X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=self.train_columns)
        self.y_test = y_test
        
        # Train all 5 models
        for name, model in self.models.items():
            model.fit(self.X_train_scaled, y_train)
            self.trained_models[name] = model
            
        # Initialize background data for SHAP
        # Using 10 samples instead of 50 since the mock dataset is only 50 rows total (train is 40)
        bg_size = min(50, len(self.X_train_scaled))
        self.background_data = shap.sample(self.X_train_scaled, bg_size)
        
        # Calculate Test AUC for a quick success metric
        auc_scores = {}
        for name, model in self.trained_models.items():
            preds = model.predict_proba(self.X_test_scaled)[:, 1]
            auc_scores[name] = roc_auc_score(y_test, preds)
            
        return auc_scores
        
    def predict_dynamic_ensemble(self, X_patient_scaled):
        # Calculate probabilities from all 5 models
        model_probs = {}
        model_confs = {}
        
        for name, model in self.trained_models.items():
            proba = model.predict_proba(X_patient_scaled)[0]
            model_probs[name] = proba
            model_confs[name] = np.max(proba) # Confidence Score
            
        # Rank by confidence score descending
        ranked_models = sorted(model_confs.items(), key=lambda x: x[1], reverse=True)
        top_2_names = [ranked_models[0][0], ranked_models[1][0]]
        
        # Average probability of Top-2
        prob_1 = model_probs[top_2_names[0]]
        prob_2 = model_probs[top_2_names[1]]
        
        final_prob = (prob_1 + prob_2) / 2
        
        # We need the probability of positive class (index 1)
        final_risk_prob = final_prob[1]
        prediction = "Positive" if final_risk_prob >= 0.5 else "Negative"
        confidence = np.max(final_prob) * 100
        
        return prediction, confidence, top_2_names, final_risk_prob

    def _get_ensemble_wrapper(self, top_2_names):
        def wrapper(X_array):
            # X_array is a numpy array passed by shap
            df_x = pd.DataFrame(X_array, columns=self.feature_names)
            preds = []
            for name in top_2_names:
                preds.append(self.trained_models[name].predict_proba(df_x)[:, 1])
            return np.mean(preds, axis=0)
        return wrapper

    def explain_prediction(self, X_patient_scaled, top_2_names):
        wrapper_func = self._get_ensemble_wrapper(top_2_names)
        explainer = shap.KernelExplainer(wrapper_func, self.background_data)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_patient_scaled, nsamples=100)
        
        # Create robust plot
        fig = plt.figure(figsize=(10, 6))
        # shap_values from KernelExplainer might be a list (if multiclass handling bleeds) or array
        # Provide robust handling
        vals = shap_values[0] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(vals, X_patient_scaled, plot_type="bar", show=False)
        plt.tight_layout()
        
        # Generate Text Explanation
        patient_vals = vals[0]
        feature_impacts = []
        for i, val in enumerate(patient_vals):
            if abs(val) > 0.005:
                feature_name = self.feature_names[i]
                feature_impacts.append((feature_name, val))
                
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        top_positive = [f.replace('_', ' ') for f, v in feature_impacts if v > 0][:3]
        top_negative = [f.replace('_', ' ') for f, v in feature_impacts if v < 0][:3]
        
        text_explanation = "**Prediction Rationale (Powered by SHAP):**\n\n"
        if top_positive:
            text_explanation += f"- 🔴 **Increased Risk Factors**: {', '.join(top_positive)}\n"
        if top_negative:
            text_explanation += f"- 🟢 **Decreased Risk Factors**: {', '.join(top_negative)}\n"
        if not top_positive and not top_negative:
            text_explanation += "The prediction is very close to the baseline average risk."
            
        return fig, text_explanation
