import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import io

st.set_page_config(
    page_title="Pass/Fail Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Pass/Fail Prediction Model")
st.markdown("Train a machine learning model to predict pass or fail outcomes based on your data.")

if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'feature_encoders' not in st.session_state:
    st.session_state.feature_encoders = {}
if 'feature_types' not in st.session_state:
    st.session_state.feature_types = {}

def generate_sample_data():
    np.random.seed(42)
    n_samples = 200
    
    study_hours = np.random.uniform(1, 10, n_samples)
    attendance = np.random.uniform(50, 100, n_samples)
    previous_score = np.random.uniform(30, 100, n_samples)
    assignments_completed = np.random.randint(0, 10, n_samples)
    
    pass_probability = (
        0.3 * (study_hours / 10) +
        0.25 * (attendance / 100) +
        0.3 * (previous_score / 100) +
        0.15 * (assignments_completed / 10)
    )
    pass_probability = pass_probability + np.random.normal(0, 0.1, n_samples)
    result = ['Pass' if p > 0.5 else 'Fail' for p in pass_probability]
    
    df = pd.DataFrame({
        'Study_Hours': np.round(study_hours, 1),
        'Attendance_Percent': np.round(attendance, 1),
        'Previous_Score': np.round(previous_score, 1),
        'Assignments_Completed': assignments_completed,
        'Result': result
    })
    
    return df

tab1, tab2, tab3 = st.tabs(["📁 Data & Training", "🔮 Prediction", "📈 Model Info"])

with tab1:
    st.header("Step 1: Load Your Data")
    
    data_option = st.radio(
        "Choose data source:",
        ["Use Sample Dataset", "Upload CSV File"],
        horizontal=True
    )
    
    df = None
    
    if data_option == "Use Sample Dataset":
        df = generate_sample_data()
        st.success("Sample student performance dataset loaded!")
        st.info("This dataset contains: Study Hours, Attendance %, Previous Score, Assignments Completed, and Pass/Fail Result")
    else:
        uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! Shape: {df.shape}")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Columns", len(df.columns))
        
        st.subheader("Step 2: Configure Model")
        
        target_col = st.selectbox(
            "Select the target column (Pass/Fail column):",
            df.columns.tolist(),
            index=len(df.columns) - 1
        )
        
        feature_cols = st.multiselect(
            "Select feature columns for prediction:",
            [col for col in df.columns if col != target_col],
            default=[col for col in df.columns if col != target_col]
        )
        
        if len(feature_cols) > 0:
            col1, col2 = st.columns(2)
            with col1:
                model_type = st.selectbox(
                    "Select Model:",
                    ["Logistic Regression", "Random Forest"]
                )
            with col2:
                test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
            
            st.subheader("Step 3: Train Model")
            
            if st.button("Train Model", type="primary", use_container_width=True):
                with st.spinner("Training model..."):
                    try:
                        X = df[feature_cols].copy()
                        y = df[target_col].copy()
                        
                        feature_encoders = {}
                        feature_types = {}
                        
                        for col in X.columns:
                            if X[col].dtype == 'object':
                                le = LabelEncoder()
                                X[col] = le.fit_transform(X[col].astype(str))
                                feature_encoders[col] = le
                                feature_types[col] = 'categorical'
                            else:
                                feature_types[col] = 'numeric'
                        
                        label_encoder = LabelEncoder()
                        y_encoded = label_encoder.fit_transform(y.astype(str))
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y_encoded, test_size=test_size, random_state=42
                        )
                        
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        if model_type == "Logistic Regression":
                            model = LogisticRegression(random_state=42, max_iter=1000)
                        else:
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                        
                        model.fit(X_train_scaled, y_train)
                        
                        y_pred = model.predict(X_test_scaled)
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        
                        st.session_state.model = model
                        st.session_state.scaler = scaler
                        st.session_state.feature_names = feature_cols
                        st.session_state.label_encoder = label_encoder
                        st.session_state.feature_encoders = feature_encoders
                        st.session_state.feature_types = feature_types
                        
                        st.success("Model trained successfully!")
                        
                        st.subheader("Model Performance")
                        
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        with metric_col1:
                            st.metric("Accuracy", f"{accuracy:.2%}")
                        with metric_col2:
                            st.metric("Precision", f"{precision:.2%}")
                        with metric_col3:
                            st.metric("Recall", f"{recall:.2%}")
                        with metric_col4:
                            st.metric("F1 Score", f"{f1:.2%}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Confusion Matrix")
                            cm = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                       xticklabels=label_encoder.classes_,
                                       yticklabels=label_encoder.classes_)
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            ax.set_title('Confusion Matrix')
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            if model_type == "Random Forest":
                                st.subheader("Feature Importance")
                                importance = model.feature_importances_
                                feat_importance = pd.DataFrame({
                                    'Feature': feature_cols,
                                    'Importance': importance
                                }).sort_values('Importance', ascending=True)
                                
                                fig, ax = plt.subplots(figsize=(6, 4))
                                ax.barh(feat_importance['Feature'], feat_importance['Importance'], color='steelblue')
                                ax.set_xlabel('Importance')
                                ax.set_title('Feature Importance')
                                st.pyplot(fig)
                                plt.close()
                            else:
                                st.subheader("Model Coefficients")
                                coef = model.coef_[0]
                                feat_coef = pd.DataFrame({
                                    'Feature': feature_cols,
                                    'Coefficient': coef
                                }).sort_values('Coefficient', ascending=True)
                                
                                fig, ax = plt.subplots(figsize=(6, 4))
                                colors = ['red' if c < 0 else 'green' for c in feat_coef['Coefficient']]
                                ax.barh(feat_coef['Feature'], feat_coef['Coefficient'], color=colors)
                                ax.set_xlabel('Coefficient')
                                ax.set_title('Feature Coefficients')
                                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                                st.pyplot(fig)
                                plt.close()
                        
                    except Exception as e:
                        st.error(f"Error training model: {e}")
        else:
            st.warning("Please select at least one feature column.")

with tab2:
    st.header("Make Predictions")
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the 'Data & Training' tab.")
    else:
        st.success("Model is ready for predictions!")
        
        st.subheader("Enter Values for Prediction")
        
        input_values = {}
        cols = st.columns(min(len(st.session_state.feature_names), 4))
        
        for i, feature in enumerate(st.session_state.feature_names):
            with cols[i % len(cols)]:
                feature_type = st.session_state.feature_types.get(feature, 'numeric')
                if feature_type == 'categorical':
                    encoder = st.session_state.feature_encoders[feature]
                    categories = list(encoder.classes_)
                    selected = st.selectbox(
                        f"{feature}",
                        options=categories,
                        key=f"pred_{feature}"
                    )
                    input_values[feature] = selected
                else:
                    input_values[feature] = st.number_input(
                        f"{feature}",
                        value=0.0,
                        step=0.1,
                        key=f"pred_{feature}"
                    )
        
        if st.button("Predict", type="primary", use_container_width=True):
            try:
                input_df = pd.DataFrame([input_values])
                
                for col in input_df.columns:
                    if st.session_state.feature_types.get(col) == 'categorical':
                        encoder = st.session_state.feature_encoders[col]
                        input_df[col] = encoder.transform(input_df[col].astype(str))
                
                input_scaled = st.session_state.scaler.transform(input_df)
                prediction = st.session_state.model.predict(input_scaled)
                prediction_proba = st.session_state.model.predict_proba(input_scaled)
                
                result = st.session_state.label_encoder.inverse_transform(prediction)[0]
                
                st.subheader("Prediction Result")
                
                if result.lower() == 'pass':
                    st.success(f"### Prediction: {result}")
                else:
                    st.error(f"### Prediction: {result}")
                
                st.subheader("Prediction Confidence")
                col1, col2 = st.columns(2)
                
                for i, class_name in enumerate(st.session_state.label_encoder.classes_):
                    with col1 if i == 0 else col2:
                        prob = prediction_proba[0][i]
                        st.metric(f"{class_name} Probability", f"{prob:.2%}")
                
                fig, ax = plt.subplots(figsize=(6, 3))
                colors = ['#ff6b6b' if c.lower() == 'fail' else '#51cf66' for c in st.session_state.label_encoder.classes_]
                ax.barh(st.session_state.label_encoder.classes_, prediction_proba[0], color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability')
                ax.set_title('Prediction Probabilities')
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")

with tab3:
    st.header("Model Information")
    
    if st.session_state.model is None:
        st.warning("No model trained yet. Please train a model first.")
    else:
        st.subheader("Current Model Details")
        
        model_name = type(st.session_state.model).__name__
        st.info(f"**Model Type:** {model_name}")
        
        st.subheader("Features Used")
        for i, feature in enumerate(st.session_state.feature_names, 1):
            st.write(f"{i}. {feature}")
        
        st.subheader("Target Classes")
        for class_name in st.session_state.label_encoder.classes_:
            st.write(f"- {class_name}")
        
        st.subheader("How to Use")
        st.markdown("""
        1. **Data & Training Tab**: Load your data and train the model
        2. **Prediction Tab**: Enter values for new records to get predictions
        3. **Model Info Tab**: View details about the trained model
        
        **Tips:**
        - Use the sample dataset to test the application
        - Upload your own CSV with a Pass/Fail column
        - Try different models to compare performance
        """)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This application uses machine learning to predict pass/fail outcomes based on input features.

**Supported Models:**
- Logistic Regression
- Random Forest

**Metrics Displayed:**
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
""")
