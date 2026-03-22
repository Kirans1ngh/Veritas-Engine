import streamlit as st
import pandas as pd
from engine import VeritasHealthEngine
import matplotlib.pyplot as plt

st.set_page_config(page_title="Veritas Health Engine", layout="wide")

if 'engine' not in st.session_state:
    st.session_state.engine = VeritasHealthEngine()
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'auc_scores' not in st.session_state:
    st.session_state.auc_scores = {}
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}

# Sidebar
st.sidebar.title("Data & Training")
uploaded_file = st.sidebar.file_uploader("Upload Clinical Dataset (CSV)", type=['csv'])

if st.sidebar.button("Load Data & Train Engine"):
    with st.spinner("Training models..."):
        try:
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.sidebar.info(f"Loaded custom dataset: {uploaded_file.name}")
            else:
                import os
                file_path = 'dataset.csv' if os.path.exists('dataset.csv') else 'diabetes.csv'
                df = pd.read_csv(file_path)
                st.sidebar.info(f"Loaded local dataset: {file_path}")
            
            auc_scores = st.session_state.engine.train(df)
            st.session_state.trained = True
            st.session_state.auc_scores = auc_scores
            st.session_state.raw_df = df
            st.sidebar.success("Training Complete!")
            st.sidebar.write("Test AUC Scores:")
            for m, s in auc_scores.items():
                st.sidebar.write(f"- {m}: {s:.2f}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# Main Page
st.title("Veritas Health Engine: Diabetes Risk Analysis")

if not st.session_state.trained:
    st.warning("Please load data and train the engine from the sidebar first.")
else:
    st.subheader("Patient Vitals Input")
    
    if st.button("🎲 Select Random Patient from Dataset"):
        st.session_state.patient_data = st.session_state.raw_df.sample(1).iloc[0].to_dict()
        
    col1, col2, col3 = st.columns(3)
    patient_data = {}
    
    # Categorical
    for i, col_name in enumerate(st.session_state.engine.categorical_cols):
        cat_mode = st.session_state.engine.imputers.get(col_name, 'unknown')
        default_val = st.session_state.patient_data.get(col_name, cat_mode)
        if pd.isna(default_val): default_val = cat_mode
        
        if i % 3 == 0: col = col1
        elif i % 3 == 1: col = col2
        else: col = col3
        patient_data[col_name] = col.text_input(col_name, value=str(default_val))
        
    # Numerical
    for i, col_name in enumerate(st.session_state.engine.numerical_cols):
        num_median = st.session_state.engine.imputers.get(col_name, 0.0)
        default_val = st.session_state.patient_data.get(col_name, num_median)
        if pd.isna(default_val): default_val = num_median
        
        idx = i + len(st.session_state.engine.categorical_cols)
        if idx % 3 == 0: col = col1
        elif idx % 3 == 1: col = col2
        else: col = col3
        patient_data[col_name] = col.number_input(col_name, value=float(default_val))
        
    if st.button("Generate Diagnostic Report"):
        with st.spinner("Analyzing patient data..."):
            try:
                df_patient = pd.DataFrame([patient_data])
                X_encoded, _ = st.session_state.engine.preprocess(df_patient)
                X_scaled_array = st.session_state.engine.scaler.transform(X_encoded)
                X_patient_scaled = pd.DataFrame(X_scaled_array, columns=st.session_state.engine.train_columns)
                
                pred, conf, top2, final_risk = st.session_state.engine.predict_dynamic_ensemble(X_patient_scaled)
                
                st.subheader("Diagnostic Report")
                c1, c2, c3 = st.columns(3)
                c1.metric("Final Risk", pred)
                c2.metric("Confidence", f"{conf:.1f}%")
                c3.info(f"Top-2 Models Used: {top2[0]}, {top2[1]}")
                
                st.subheader("Feature Interpretability (SHAP)")
                fig, text_reasoning = st.session_state.engine.explain_prediction(X_patient_scaled, top2, patient_data)
                st.info(text_reasoning)
                st.pyplot(fig)
                plt.close(fig) # Prevent memory leaks
            except Exception as e:
                import traceback
                st.error(f"Error generating report: {e}")
                st.code(traceback.format_exc())
