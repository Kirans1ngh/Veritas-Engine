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
            df_x = pd.DataFrame(X_array, columns=self.feature_names)
            preds = []
            for name in top_2_names:
                preds.append(self.trained_models[name].predict_proba(df_x)[:, 1])
            return np.mean(preds, axis=0)
        return wrapper

    def explain_prediction(self, X_patient_scaled, top_2_names, patient_data=None):
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
        
        # Plain English Generator
        patient_vals = vals[0]
        feature_impacts = []
        for i, val in enumerate(patient_vals):
            if abs(val) > 0.005:
                feature_name = self.feature_names[i]
                feature_impacts.append((feature_name, val))
                
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Generator for curated personalized sentences
        def get_insight_sentence(feature, impact_type):
            if patient_data is None:
                return f"your {feature.replace('_', ' ').lower()}"
                
            base_feat = None
            for k in patient_data.keys():
                if feature.startswith(k):
                    base_feat = k
                    break
            
            if not base_feat:
                return f"your {feature.replace('_', ' ').lower()}"

            val = patient_data[base_feat]
            
            if impact_type == 'risk':
                if base_feat == 'Smoking_Status' and str(val).lower() in ['yes', 'former']:
                    return f"your history of smoking ('{val}') is a major risk factor—quitting or maintaining strict cessation is critical"
                elif base_feat == 'Fasting_Blood_Sugar':
                    return f"your fasting blood sugar level of {val} mg/dL is elevated, heavily driving up your risk"
                elif base_feat == 'Postprandial_Blood_Sugar':
                    return f"your post-meal blood sugar level of {val} mg/dL is higher than optimal after eating"
                elif base_feat == 'HBA1C':
                    return f"your HbA1c level of {val}% indicates prolonged high blood sugar over recent months"
                elif base_feat == 'Glucose_Tolerance_Test_Result':
                    return f"your glucose tolerance test result of {val} mg/dL points to impaired metabolic processing"
                elif base_feat == 'BMI':
                    return f"your BMI of {val} puts added strain on your metabolic system—weight management is strongly advised"
                elif base_feat == 'Age':
                    return f"your age ({val} years) naturally limits metabolic flexibility, driving up the baseline risk"
                elif base_feat == 'Family_History' and str(val).lower() == 'yes':
                    return "your family history of diabetes creates a strong genetic predisposition"
                elif base_feat == 'Hypertension' and str(val).lower() == 'yes':
                    return "your current battle with high blood pressure (hypertension) significantly compounds your metabolic risk"
                elif base_feat == 'Physical_Activity' and str(val).lower() == 'low':
                    return "your low level of physical activity is slowing your metabolism—incorporating daily exercise would greatly help"
                elif base_feat == 'Diet_Type':
                    return f"your current dietary habits ({val}) are notably increasing your risk"
                elif base_feat == 'Stress_Level' and str(val).lower() == 'high':
                    return "your high stress levels are negatively impacting your hormonal balance and blood sugar"
                else:
                    return f"your {base_feat.replace('_', ' ').lower()} ({val}) is elevating your risk profile"
            else: # Protective
                if base_feat == 'Physical_Activity' and str(val).lower() in ['high', 'medium']:
                    return f"your strong commitment to physical activity ('{val}') is doing an excellent job protecting your metabolism"
                elif base_feat == 'Diet_Type':
                    return f"your healthy dietary choices ('{val}') actively keep your risk low by providing good nutritional balance"
                elif base_feat == 'Fasting_Blood_Sugar':
                    return f"maintaining a perfectly healthy fasting blood sugar level of {val} mg/dL is greatly reducing your risk"
                elif base_feat == 'HBA1C':
                    return f"your excellent HbA1c level of {val}% shows outstanding long-term blood sugar control"
                elif base_feat == 'BMI':
                    return f"maintaining a healthy BMI of {val} acts as a massive protective factor for your system"
                elif base_feat == 'Stress_Level' and str(val).lower() == 'low':
                    return "your well-managed stress levels keep harmful cortisol spikes away, contributing positively to your health"
                elif base_feat == 'Smoking_Status' and str(val).lower() == 'no':
                    return "the fact that you completely avoid smoking acts as a powerful shield for your cardiovascular and metabolic health"
                else:
                    return f"your {base_feat.replace('_', ' ').lower()} ({val}) is actively protecting your health"

        top_positive = [get_insight_sentence(f, 'risk') for f, v in feature_impacts if v > 0][:3]
        top_negative = [get_insight_sentence(f, 'protective') for f, v in feature_impacts if v < 0][:3]
        
        # Remove duplicates while preserving order
        top_positive = list(dict.fromkeys(top_positive))
        top_negative = list(dict.fromkeys(top_negative))
        
        text_explanation = "🩺 **Your Personalized Health Insights:**\n\n"
        
        if top_positive:
            text_explanation += "### Areas of Concern (Driving Risk Up):\n"
            for insight in top_positive:
                text_explanation += f"- {insight.capitalize()}.\n"
            text_explanation += "\n*Finding ways to manage or improve these areas alongside a medical professional could significantly lower your overall risk.*\n\n"
            
        if top_negative:
            text_explanation += "### Protective Factors (Keeping Risk Low):\n"
            for insight in top_negative:
                text_explanation += f"- {insight.capitalize()}.\n"
            text_explanation += "\n*You are doing excellent in these areas. Keep up the good work!*\n"
            
        if not top_positive and not top_negative:
            text_explanation += "Your health profile is remarkably well balanced. No single factor is drastically pushing your risk up or down."
            
        return fig, text_explanation
